import os
import argparse
from huggingface_hub import hf_hub_download

def download_model(repo_id: str, local_dir: str, token: str = None):
    os.makedirs(local_dir, exist_ok=True)
    files = ['model.onnx', 'selected_tags.csv', 'preprocess.json', 'categories.json', 'thresholds.csv']
    print(f"Downloading model '{repo_id}' files to '{local_dir}'...")
    for filename in files:
        try:
            print(f"Downloading {filename}...")
            path = hf_hub_download(
                repo_id=repo_id,
                repo_type='model',
                filename=filename,
                token=token,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
            print(f"Saved to: {path}")
        except Exception as e:
            if filename == 'thresholds.csv':
                print(f"Optional file thresholds.csv not found on repo, skipping.")
            else:
                print(f"Error downloading {filename}: {e}")
                raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download multi-label tagger model files from Hugging Face Hub.")
    parser.add_argument('--repo', type=str, default='Mooshie/caformer_b36.dbv4-full', help='Hugging Face model repository ID')
    parser.add_argument('--dir', type=str, default='./local_model', help='Local folder directory to save the files to')
    parser.add_argument('--token', type=str, default=None, help='Optional Hugging Face authentication token')
    args = parser.parse_args()
    download_model(args.repo, args.dir, args.token)
