import os
import cv2
import urllib
from tqdm import tqdm

from models_.regular_approach import RegularApproach
from models_.gan_approach_1 import GANApproach_1

def blur_val2017_subset():
    input_dir = "val2017_subset"
    output_dir = "val2017_subset_regular_approach"

    # Create RegularApproach instance with target output dir
    ra = RegularApproach(output_dir=output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all valid image files
    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
    ]

    if len(image_files) == 0:
        print(f"❌ No images found in {input_dir}")
        return

    total_faces = 0
    successful = 0
    failed = 0

    # Process with progress bar
    for file_name in tqdm(image_files, desc="Processing images", unit="img"):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        img = cv2.imread(input_path)
        if img is None:
            print(f"\n⚠️  Skipping unreadable file: {file_name}")
            failed += 1
            continue

        # Process and get face count
        processed_img, face_count = ra.process_img(img, show_count=False)
        
        # Save processed image
        success = cv2.imwrite(output_path, processed_img)
        
        if success:
            total_faces += face_count
            successful += 1
        else:
            print(f"\n⚠️  Failed to save: {file_name}")
            failed += 1
    print(f"\nOutput saved to: {output_dir}")
    print("="*50)

def anonymize_val2017_subset_ver1():
        gan_model_1 = GANApproach_1()

        data_dir = "val2017_subset"
        processed_dir = "val2017_subset_GANs"
        gan_model_1.preprocess(input_path=data_dir, img_size=512, align=True, test_size=0.1,
                shuffle=True, output_dir=processed_dir, num_workers=8)

        # Step 2: Download model if needed
        models_dir = "pre_trained_models"
        os.makedirs(models_dir, exist_ok=True)
        model_file = os.path.join(models_dir, "GANonymization_50.ckpt")
        
        if not os.path.exists(model_file):
            print("Downloading pre-trained model (50 epochs)...")
            model_ckpt = urllib.request.urlretrieve(
                "https://mediastore.rz.uni-augsburg.de/get/Sfle_etB1D/",
                model_file)
            if not os.path.exists(model_ckpt[0]):
                raise RuntimeError(f"Failed to download model to {model_file}")
            print(f"Model downloaded successfully to: {model_ckpt[0]}")
            model_file = model_ckpt[0]
        else:
            print(f"Model already exists at: {model_file}")
        
        # Step 3: Anonymize

        input_dir = os.path.join(processed_dir, "original", "val")
        output_dir = os.path.join(processed_dir, "anonymized_val")
        os.makedirs(output_dir, exist_ok=True)

        gan_model_1.anonymize_directory( 
            model_file=model_file,
            input_directory=input_dir,
            output_directory=output_dir,
            img_size=512,
            align=True,
            device=-1  # Use CPU (-1), change to 0 for GPU
        )
        
if __name__ == "__main__":
    blur_val2017_subset()