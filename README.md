# timmtagger-lite

A lightweight, high-performance anime image tagger extracted from `imgutils`, optimized for parallel batch processing on Windows GPU using **DirectML** (with CPU fallback). 

It uses the `Mooshie/caformer_b36.dbv4-full` model by default and supports both online downloading from Hugging Face and offline loading of local models.

---

## Key Features

1. **DirectML GPU Acceleration**: Native GPU execution out-of-the-box on Windows (both AMD and NVIDIA GPUs) without requiring complex CUDA/cuDNN configuration.
2. **Parallel Batch Inference**: Preprocesses and stacks images to run a single parallel forward pass on the GPU, yielding significantly higher throughput than loop-based sequential processing.
3. **Local/Offline Mode**: Run inference offline by providing a local path to a downloaded model directory.
4. **Pruned & Lightweight**: All non-tagging modules (like OCR, metadata, CCIP, and upscaling) and their heavy requirements (like OpenCV, PyTorch, and torchvision) have been stripped.

---

## Installation

*Note: This project is verified and tested on **Python 3.11** (Windows).*

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Cyrus-Hei/timmtagger-lite.git
   cd timmtagger-lite
   ```

2. **Set up a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This automatically installs `onnxruntime-directml` for GPU execution on Windows.*

---

## Usage

The project includes a command-line script, `batch_inference.py`, to tag all images inside a folder and write the tags to matching `.txt` files.

### 1. Online Execution (Automatic Download)
Run batch inference on a folder of images. It will automatically download the default model (`Mooshie/caformer_b36.dbv4-full`) from Hugging Face on its first run:
```bash
python batch_inference.py <folder_path>
```

### 2. Local / Offline Execution
To run without internet, download the model files from the Hugging Face repository (including `model.onnx`, `selected_tags.csv`, `categories.json`, `preprocess.json`, and `thresholds.csv`) and place them in a folder (e.g. `./local_model`).

Then run:
```bash
python batch_inference.py <folder_path> --model ./local_model
```

### 3. Customize Thresholds
Adjust prediction thresholds using CLI parameters:
```bash
python batch_inference.py <folder_path> --gen-threshold 0.39 --char-threshold 0.47 --rating-threshold 0.39
```

---

## Programmatic API

You can also import prediction functions directly in Python:

```python
from imgutils.generic import multilabel_timm_predict_batch

image_paths = ["image1.png", "image2.jpg"]
model_path = "./local_model" # Or "Mooshie/caformer_b36.dbv4-full"

# Returns a list of predictions for each image in the batch
batch_results = multilabel_timm_predict_batch(
    images=image_paths,
    repo_id=model_path,
    fmt=('general', 'character', 'rating'),
)

for path, (general, character, rating) in zip(image_paths, batch_results):
    print(f"--- Tags for {path} ---")
    print("General:", general)
    print("Character:", character)
    print("Rating:", rating)
```

---

## Acknowledgements

This project is a lightweight, customized extraction focusing exclusively on anime image tagging. I want to acknowledge and thank:
- **[imgutils](https://github.com/deepghs/imgutils)** and its authors (**narugo1992** and **7eu7d7**) for the original, feature-rich repository from which this tagger was extracted.
- **[animetimm](https://huggingface.co/animetimm)** (and un-gater **[Mooshie](https://huggingface.co/Mooshie)**) on Hugging Face for training and open-weighting the multi-label CAFormer models used in this project.

