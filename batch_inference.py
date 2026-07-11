from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Tag anime images in a directory using a multi-label model in batch.")
    parser.add_argument("folder_path", type=str, help="The path to the folder containing images")
    parser.add_argument("--model", type=str, default="Mooshie/caformer_b36.dbv4-full",
                        help="Hugging Face model repo ID or local path to downloaded model directory")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="Optional Hugging Face authentication token")
    parser.add_argument("--gen-threshold", type=float, default=0.39, help="Threshold for general tags")
    parser.add_argument("--char-threshold", type=float, default=0.47, help="Threshold for character tags")
    parser.add_argument("--rating-threshold", type=float, default=0.39, help="Threshold for rating tags")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for model inference to prevent VRAM OOM")
    
    args = parser.parse_args()
    img_folder = Path(args.folder_path)
    if not img_folder.is_dir():
        print(f"Error: The directory '{img_folder}' does not exist.")
        return
    
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
    image_paths = []
    for file_path in img_folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_paths.append(file_path)
            
    if not image_paths:
        print("No images found in the directory.")
        return
        
    print(f"Found {len(image_paths)} images. Starting batch prediction...")
    
    import onnxruntime as ort
    ort.preload_dlls()
    from imgutils.generic import multilabel_timm_predict_batch
    from imgutils.tagging import tags_to_text
    
    batch_results = multilabel_timm_predict_batch(
        [p.resolve() for p in image_paths],
        hf_token=args.hf_token,
        repo_id=args.model,
        fmt=('general', 'character', 'rating'),
        batch_size=args.batch_size,
    )
    
    for file_path, (general, character, rating) in zip(image_paths, batch_results):
        general_t = {k: v for k, v in general.items() if v >= args.gen_threshold}
        char_t = {k: v for k, v in character.items() if v >= args.char_threshold}
        rating_t = {k: v for k, v in rating.items() if v >= args.rating_threshold}
        all_tags = general_t | char_t | rating_t
        text_tags = tags_to_text(all_tags, use_spaces=True)
        txt_file_path = img_folder / f"{file_path.stem}.txt"
        
        try:
            with open(txt_file_path, 'w', encoding='utf-8') as f:
                f.write(text_tags)
            print(f"Created/Overwrote: {txt_file_path.name}")
        except IOError as e:
            print(f"Could not write to {txt_file_path.name}: {e}")
        

if __name__ == "__main__":
    main()