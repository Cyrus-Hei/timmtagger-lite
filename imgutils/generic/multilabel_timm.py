"""
Multi-Label TIMM Model Module

This module provides functionality for working with multi-label image classification models
trained with TIMM (PyTorch Image Models) and exported to ONNX format. It includes:

1. The MultiLabelTIMMModel class for loading and making predictions with models hosted on Hugging Face Hub
2. Functions for batch prediction and demo interface creation
3. Support for custom thresholds at both category and tag levels
4. Flexible output formatting options for different use cases

The models are expected to be stored on Hugging Face Hub with specific files:

- model.onnx: The ONNX model file
- selected_tags.csv: CSV file containing tag information and categories
- preprocess.json: JSON configuration for image preprocessing
- thresholds.csv: Optional CSV file with recommended thresholds
- categories.json: Category ID and name mapping json file.

This module is designed to work with multi-label classification tasks where images can
belong to multiple categories and have multiple tags within each category.
"""


import json
import os
import warnings
from threading import Lock
from typing import Optional, Literal, Dict, Any, Union

from hbutils.design import SingletonMark


from ..data import ImageTyping, load_image
from ..preprocess import create_pillow_transforms
from ..utils import open_onnx_model
from ..utils import vreplace, ts_lru_cache



__all__ = [
    'MultiLabelTIMMModel',
    'multilabel_timm_predict_batch',
]




FMT_UNSET = SingletonMark('FMT_UNSET')


class MultiLabelTIMMModel:
    """
    A class for working with multi-label image classification models trained with TIMM.

    This class handles loading models from Hugging Face Hub, preprocessing images,
    and making predictions with customizable thresholds.

    :param repo_id: The Hugging Face Hub repository ID containing the model
    :type repo_id: str
    :param hf_token: Optional Hugging Face authentication token for private repositories
    :type hf_token: Optional[str]
    """

    def __init__(self, repo_id: str, hf_token: Optional[str] = None):
        """
        Initialize a MultiLabelTIMMModel.

        :param repo_id: The Hugging Face Hub repository ID containing the model
        :type repo_id: str
        :param hf_token: Optional Hugging Face authentication token for private repositories
        :type hf_token: Optional[str]
        """
        self.repo_id = repo_id
        self._model = None
        self._df_tags = None
        self._preprocess = None
        self._default_category_thresholds = None
        self._hf_token = hf_token
        self._lock = Lock()
        self._category_names = {}
        self._name_to_categories = None

    def _get_hf_token(self) -> Optional[str]:
        """
        Retrieve the Hugging Face authentication token.

        Checks both instance variable and environment for token presence.

        :return: Authentication token if available
        :rtype: Optional[str]
        """
        return self._hf_token or os.environ.get('HF_TOKEN')

    def _get_file(self, filename: str) -> str:
        if os.path.isdir(self.repo_id):
            path = os.path.join(self.repo_id, filename)
            if not os.path.exists(path):
                from huggingface_hub.errors import EntryNotFoundError
                raise EntryNotFoundError(f"Local file {filename} not found in {self.repo_id}")
            return path
        else:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(
                repo_id=self.repo_id,
                repo_type='model',
                filename=filename,
                token=self._get_hf_token(),
            )

    def _open_model(self):
        """
        Load the ONNX model.

        :return: The loaded ONNX model
        :rtype: object
        """
        with self._lock:
            if self._model is None:
                self._model = open_onnx_model(self._get_file('model.onnx'))

        return self._model

    def _open_tags(self):
        """
        Load tag information.

        :return: Dict containing tag information arrays
        :rtype: dict
        """
        with self._lock:
            if self._df_tags is None:
                import csv
                import numpy as np
                with open(self._get_file('selected_tags.csv'), 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)

                self._tag_categories = np.array([int(r['category']) for r in rows], dtype=np.int32)
                self._tag_names = np.array([r['name'] for r in rows], dtype=object)
                if 'best_threshold' in rows[0] and rows[0]['best_threshold'] != '':
                    self._tag_best_thresholds = np.array([float(r['best_threshold']) if r['best_threshold'] != '' else 0.4 for r in rows], dtype=np.float32)
                else:
                    self._tag_best_thresholds = None

                self._df_tags = {
                    'category': self._tag_categories,
                    'name': self._tag_names,
                    'best_threshold': self._tag_best_thresholds,
                }

                with open(self._get_file('categories.json'), 'r') as f:
                    d_category_names = {cate_item['category']: cate_item['name'] for cate_item in json.load(f)}
                    self._name_to_categories = {}
                    for category in sorted(set(self._tag_categories)):
                        self._category_names[category] = d_category_names[category]
                        self._name_to_categories[self._category_names[category]] = category

        return self._df_tags

    def _open_preprocess(self):
        """
        Load preprocessing configuration.

        :return: A tuple of validation and test preprocessing transforms
        :rtype: tuple
        """
        with self._lock:
            if self._preprocess is None:
                with open(self._get_file('preprocess.json'), 'r') as f:
                    data_ = json.load(f)
                    test_trans = create_pillow_transforms(data_['test'])
                    val_trans = create_pillow_transforms(data_['val'])
                    self._preprocess = val_trans, test_trans

        return self._preprocess

    def _open_default_category_thresholds(self):
        """
        Load default category thresholds.

        :return: Dictionary mapping category IDs to threshold values
        :rtype: dict
        """
        with self._lock:
            if self._default_category_thresholds is None:
                from huggingface_hub.errors import EntryNotFoundError
                try:
                    path = self._get_file('thresholds.csv')
                except (EntryNotFoundError,):
                    self._default_category_thresholds = {}
                else:
                    import csv
                    self._default_category_thresholds = {}
                    with open(path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            cate = int(row['category'])
                            threshold = float(row['threshold'])
                            if cate not in self._default_category_thresholds:
                                self._default_category_thresholds[cate] = threshold

        return self._default_category_thresholds



    def _raw_predict_batch(self, images, preprocessor: Literal['test', 'val'] = 'test', batch_size: int = 16):
        model = self._open_model()

        val_trans, test_trans = self._open_preprocess()
        if preprocessor == 'test':
            trans = test_trans
        elif preprocessor == 'val':
            trans = val_trans
        else:
            raise ValueError(
                f'Unknown processor, "test" or "val" expected but {preprocessor!r} found.')

        import numpy as np
        from concurrent.futures import ThreadPoolExecutor

        def _load_and_preprocess(img_item):
            img = load_image(img_item, force_background='white', mode='RGB')
            return trans(img)

        output_names = [output.name for output in model.get_outputs()]
        all_output_values = {name: [] for name in output_names}
        
        num_workers = min(batch_size, os.cpu_count() or 4)

        for i in range(0, len(images), batch_size):
            chunk = images[i:i + batch_size]
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                tensors = list(executor.map(_load_and_preprocess, chunk))
            
            input_ = np.stack(tensors, axis=0)
            output_values = model.run(output_names, {'input': input_})
            for name, value in zip(output_names, output_values):
                all_output_values[name].append(value)

        return {name: np.concatenate(vals, axis=0) for name, vals in all_output_values.items()}

    def predict_batch(self, images, preprocessor: Literal['test', 'val'] = 'test',
                      thresholds: Union[float, Dict[Any, float]] = None, use_tag_thresholds: bool = True,
                      fmt=FMT_UNSET, batch_size: int = 16):
        if not images:
            return []

        df_tags = self._open_tags()
        values_batch = self._raw_predict_batch(images, preprocessor=preprocessor, batch_size=batch_size)
        prediction_batch = values_batch['prediction']

        results = []
        for idx in range(len(images)):
            prediction = prediction_batch[idx]
            tags = {}

            if fmt is FMT_UNSET:
                fmt_val = tuple(self._category_names[category] for category in sorted(set(df_tags['category'].tolist())))
            else:
                fmt_val = fmt

            default_category_thresholds = self._open_default_category_thresholds()
            if 'best_threshold' in self._df_tags and self._df_tags['best_threshold'] is not None:
                default_tag_thresholds = self._df_tags['best_threshold']
            else:
                default_tag_thresholds = None

            values = {
                'prediction': prediction,
            }
            if 'logits' in values_batch:
                values['logits'] = values_batch['logits'][idx]
            if 'embedding' in values_batch:
                values['embedding'] = values_batch['embedding'][idx]

            for category in sorted(set(df_tags['category'].tolist())):
                mask = df_tags['category'] == category
                tag_names = df_tags['name'][mask]
                category_pred = prediction[mask]

                if isinstance(thresholds, float):
                    category_threshold = thresholds
                elif isinstance(thresholds, dict) and \
                        (category in thresholds or self._category_names[category] in thresholds):
                    if category in thresholds:
                        category_threshold = thresholds[category]
                    elif self._category_names[category] in thresholds:
                        category_threshold = thresholds[self._category_names[category]]
                    else:
                        assert False
                elif category in default_category_thresholds:
                    category_threshold = default_category_thresholds[category]
                else:
                    category_threshold = 0.4

                if use_tag_thresholds and default_tag_thresholds is not None:
                    tag_thresholds = default_tag_thresholds[mask]
                    mask = category_pred >= tag_thresholds
                else:
                    mask = category_pred >= category_threshold

                tag_names_filtered = tag_names[mask].tolist()
                category_pred_filtered = category_pred[mask].tolist()
                cate_tags = dict(sorted(zip(tag_names_filtered, category_pred_filtered), key=lambda x: (-x[1], x[0])))
                values[self._category_names[category]] = cate_tags
                tags.update(cate_tags)

            values['tag'] = tags
            results.append(vreplace(fmt_val, values))

        return results



@ts_lru_cache()
def _open_models_for_repo_id(repo_id: str, hf_token: Optional[str] = None) \
        -> MultiLabelTIMMModel:
    """
    Open and cache a MultiLabelTIMMModel for a given repository ID.

    :param repo_id: The Hugging Face Hub repository ID
    :type repo_id: str
    :param hf_token: Optional Hugging Face authentication token
    :type hf_token: Optional[str]

    :return: A cached MultiLabelTIMMModel instance
    :rtype: MultiLabelTIMMModel
    """
    return MultiLabelTIMMModel(
        repo_id=repo_id,
        hf_token=hf_token,
    )



def multilabel_timm_predict_batch(images, repo_id: str,
                                  preprocessor: Literal['test', 'val'] = 'test',
                                  thresholds: Union[float, Dict[Any, float]] = None, use_tag_thresholds: bool = True,
                                  fmt=FMT_UNSET, hf_token: Optional[str] = None, batch_size: int = 16):
    """
    Make predictions for a batch of images using a multi-label TIMM model.
    """
    model = _open_models_for_repo_id(
        repo_id=repo_id,
        hf_token=hf_token,
    )
    return model.predict_batch(
        images=images,
        preprocessor=preprocessor,
        thresholds=thresholds,
        use_tag_thresholds=use_tag_thresholds,
        fmt=fmt,
        batch_size=batch_size,
    )
