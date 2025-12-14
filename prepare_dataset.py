"""
Helper script to prepare and validate your dataset
"""
import os
import shutil
from PIL import Image
import argparse


def validate_images(directory, image_type="image"):
    """Validate that all files in directory are valid images"""
    valid_extensions = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    valid_files = []
    invalid_files = []
    
    if not os.path.exists(directory):
        print(f"[ERROR] Directory does not exist: {directory}")
        return valid_files, invalid_files
    
    files = os.listdir(directory)
    print(f"\nValidating {image_type}s in: {directory}")
    
    for filename in files:
        if filename.lower().endswith(valid_extensions):
            filepath = os.path.join(directory, filename)
            try:
                # Try to open and verify it's a valid image
                img = Image.open(filepath)
                img.verify()
                valid_files.append(filename)
            except Exception as e:
                print(f"[WARNING] Invalid image: {filename} - {str(e)}")
                invalid_files.append(filename)
        else:
            invalid_files.append(filename)
            print(f"[WARNING] Not an image file: {filename}")
    
    print(f"[OK] Found {len(valid_files)} valid {image_type}s")
    if invalid_files:
        print(f"[WARNING] Found {len(invalid_files)} invalid files")
    
    return valid_files, invalid_files


def check_dataset_structure(data_dir):
    """Check if dataset structure is correct"""
    print("=" * 60)
    print("Checking Dataset Structure")
    print("=" * 60)
    
    required_dirs = {
        'train_sketches': os.path.join(data_dir, 'train', 'sketches'),
        'train_images': os.path.join(data_dir, 'train', 'images'),
    }
    
    optional_dirs = {
        'val_sketches': os.path.join(data_dir, 'val', 'sketches'),
        'val_images': os.path.join(data_dir, 'val', 'images'),
    }
    
    # Check required directories
    all_exist = True
    for name, path in required_dirs.items():
        if os.path.exists(path):
            print(f"[OK] {name}: {path}")
        else:
            print(f"[ERROR] {name}: {path} (MISSING)")
            all_exist = False
    
    # Check optional directories
    for name, path in optional_dirs.items():
        if os.path.exists(path):
            print(f"[OK] {name}: {path} (optional)")
        else:
            print(f"[INFO] {name}: {path} (optional, not found)")
    
    if not all_exist:
        print("\n[ERROR] Required directories are missing!")
        print("\nExpected structure:")
        print("dataset/")
        print("  train/")
        print("    sketches/  ← Put your sketch images here")
        print("    images/    ← Put corresponding real images here")
        print("  val/         ← Optional, for validation")
        print("    sketches/")
        print("    images/")
        return False
    
    return True


def validate_pairs(sketch_dir, image_dir):
    """Validate that sketch and image files are paired correctly"""
    print("\n" + "=" * 60)
    print("Validating Sketch-Image Pairs")
    print("=" * 60)
    
    sketch_files, _ = validate_images(sketch_dir, "sketch")
    image_files, _ = validate_images(image_dir, "image")
    
    # Sort to ensure consistent ordering
    sketch_files = sorted(sketch_files)
    image_files = sorted(image_files)
    
    # Check if counts match
    if len(sketch_files) != len(image_files):
        print(f"\n[WARNING] Mismatch in file counts!")
        print(f"   Sketches: {len(sketch_files)}")
        print(f"   Images: {len(image_files)}")
        print("\n   The model expects paired data. Make sure:")
        print("   - Each sketch has a corresponding image")
        print("   - Files are named consistently (e.g., sketch_001.png, image_001.png)")
        return False
    
    print(f"\n[OK] Found {len(sketch_files)} matching pairs")
    
    # Check if filenames match (without extension)
    mismatches = []
    for i, (sketch_file, image_file) in enumerate(zip(sketch_files, image_files)):
        sketch_base = os.path.splitext(sketch_file)[0]
        image_base = os.path.splitext(image_file)[0]
        
        if sketch_base != image_base:
            mismatches.append((sketch_file, image_file))
    
    if mismatches:
        print(f"\n[WARNING] Found {len(mismatches)} filename mismatches:")
        for sketch, image in mismatches[:5]:  # Show first 5
            print(f"   {sketch} <-> {image}")
        if len(mismatches) > 5:
            print(f"   ... and {len(mismatches) - 5} more")
        print("\n   Note: Files don't need matching names, but order matters!")
    else:
        print("[OK] All filenames match (good for pairing)")
    
    return True


def create_sample_structure(base_dir):
    """Create sample directory structure"""
    print("\n" + "=" * 60)
    print("Creating Sample Directory Structure")
    print("=" * 60)
    
    directories = [
        os.path.join(base_dir, 'train', 'sketches'),
        os.path.join(base_dir, 'train', 'images'),
        os.path.join(base_dir, 'val', 'sketches'),
        os.path.join(base_dir, 'val', 'images'),
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"[OK] Created: {directory}")
    
    print(f"\n[OK] Directory structure created at: {base_dir}")
    print("\nNext steps:")
    print("1. Put your sketch images in: dataset/train/sketches/")
    print("2. Put corresponding real images in: dataset/train/images/")
    print("3. Make sure sketch and image files are paired (same order or same names)")
    print("4. Optionally add validation data to dataset/val/")


def main():
    parser = argparse.ArgumentParser(description='Prepare and validate dataset')
    parser.add_argument('--data_dir', type=str, default='./dataset',
                       help='Dataset directory')
    parser.add_argument('--create_structure', action='store_true',
                       help='Create directory structure if it doesn\'t exist')
    parser.add_argument('--validate', action='store_true',
                       help='Validate existing dataset')
    
    args = parser.parse_args()
    
    if args.create_structure:
        create_sample_structure(args.data_dir)
    
    if args.validate or not args.create_structure:
        if check_dataset_structure(args.data_dir):
            train_sketch_dir = os.path.join(args.data_dir, 'train', 'sketches')
            train_image_dir = os.path.join(args.data_dir, 'train', 'images')
            
            if os.path.exists(train_sketch_dir) and os.path.exists(train_image_dir):
                validate_pairs(train_sketch_dir, train_image_dir)
            
            print("\n" + "=" * 60)
            print("Dataset Validation Complete!")
            print("=" * 60)
            print("\nIf everything looks good, you can start training with:")
            print(f"python train.py --data_dir {args.data_dir}")


if __name__ == "__main__":
    main()

