#!/usr/bin/env python3
import sys
import os
import shutil
from PIL import Image, ImageOps
from collections import defaultdict
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import math

# Supported image extensions
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")

# Thread-safe printing
print_lock = Lock()

def thread_safe_print(msg):
    with print_lock:
        print(msg)

def collect_images_by_class(root_path):
    """
    Recursively collect images organized by their deepest parent directory (class).
    Searches at any depth and groups by the immediate parent directory.
    Returns: dict with class_name -> list of image paths
    """
    class_images = defaultdict(list)
    
    if not os.path.isdir(root_path):
        print(f"Error: '{root_path}' is not a directory.")
        return class_images
    
    # Walk through all directories recursively
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(IMG_EXTENSIONS):
                file_path = os.path.join(root, file)
                # Use the immediate parent directory as the class name
                class_name = os.path.basename(root)
                class_images[class_name].append(file_path)
    
    return class_images

def apply_transformations(img):
    """
    Apply all transformations to an image and return a list of transformed images.
    Returns: list of (transform_name, PIL.Image) tuples
    """
    transforms = []
    width, height = img.size
    
    # 1. Flip
    flipped = ImageOps.mirror(img)
    transforms.append(("flip", flipped))
    
    # 2. Rotate
    rotated = img.rotate(45, expand=True)
    transforms.append(("rotate", rotated))
    
    # 3. Crop
    cropped = img.crop((width//4, height//4, 3*width//4, 3*height//4))
    transforms.append(("crop", cropped))
    
    # 4. Shear
    m = -0.5
    xshift = abs(m) * height
    new_width = width + int(round(xshift))
    sheared = img.transform(
        (new_width, height),
        Image.AFFINE,
        (1, m, -xshift if m > 0 else 0, 0, 1, 0),
        resample=Image.BICUBIC
    )
    transforms.append(("shear", sheared))
    
    # 5. Distortion
    coeffs = [1, 0.2, 0, 0.2, 1, 0]
    distorted = img.transform(img.size, Image.AFFINE, coeffs, resample=Image.BICUBIC)
    transforms.append(("distortion", distorted))
    
    return transforms

def process_single_image(args):
    """
    Process a single image: save original and create transforms.
    Returns: (original_path, [transform_paths]) or None on error
    """
    src_path, output_dir, group_id, num_transforms = args
    
    try:
        with Image.open(src_path) as img:
            # Save original
            original_output = os.path.join(output_dir, f"{group_id}_original.jpg")
            img.save(original_output)
            
            # Create transforms
            transforms = apply_transformations(img)
            transform_paths = []
            
            for tname, timg in transforms[:num_transforms]:
                transform_output = os.path.join(output_dir, f"{group_id}_{tname}.jpg")
                timg.save(transform_output)
                transform_paths.append(transform_output)
            
            return (original_output, transform_paths)
    except Exception as e:
        thread_safe_print(f"    Warning: failed to process {src_path}: {e}")
        return None

def augment_class(class_name, image_paths, target_total_images, output_dir, transforms_per_image=5, max_workers=4):
    """
    Augment images in a class to reach target_total_images exactly.
    Uses multithreading for faster processing.
    Returns: list of image groups [(original_path, [transform_paths]), ...]
    """
    class_output_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)
    
    originals = list(image_paths)
    random.shuffle(originals)
    num_originals = len(originals)
    
    # Calculate how many images per group (1 original + N transforms)
    images_per_group = 1 + transforms_per_image
    
    # Calculate how many groups we need
    num_groups_needed = math.ceil(target_total_images / images_per_group)
    
    # If we need fewer groups than we have originals, just sample down
    if num_groups_needed <= num_originals:
        selected_originals = originals[:num_groups_needed]
        # Adjust last group to hit exact target
        full_groups = target_total_images // images_per_group
        remainder = target_total_images % images_per_group
        
        thread_safe_print(f"  Class '{class_name}': Creating {num_groups_needed} groups ({target_total_images} images)")
    else:
        # We need to cycle through originals multiple times
        selected_originals = []
        for i in range(num_groups_needed):
            selected_originals.append(originals[i % num_originals])
        
        full_groups = target_total_images // images_per_group
        remainder = target_total_images % images_per_group
        
        thread_safe_print(f"  Class '{class_name}': Creating {num_groups_needed} groups from {num_originals} originals ({target_total_images} images)")
    
    # Prepare jobs for thread pool
    jobs = []
    for idx, src_path in enumerate(selected_originals):
        base_name = os.path.splitext(os.path.basename(src_path))[0]
        group_id = f"{base_name}_g{idx}"
        
        # Determine how many transforms for this group
        if idx < full_groups:
            num_transforms = transforms_per_image
        elif idx == full_groups:
            # Last partial group (if remainder > 0)
            num_transforms = max(0, remainder - 1)  # -1 because original counts as 1
        else:
            # No more images needed
            break
        
        jobs.append((src_path, class_output_dir, group_id, num_transforms))
    
    # Process images in parallel
    image_groups = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_image, job) for job in jobs]
        
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                image_groups.append(result)
    
    total_images = sum(1 + len(tpaths) for _, tpaths in image_groups)
    thread_safe_print(f"    Created {len(image_groups)} groups ({total_images} images)")
    
    return image_groups

def split_dataset(class_groups, output_base_dir, train_ratio=0.7, eval_ratio=0.15, test_ratio=0.15):
    """
    Split images (not groups) into train/eval/test sets.
    This ensures exact ratio splits by splitting at the IMAGE level.
    """
    train_dir = os.path.join(output_base_dir, "train")
    eval_dir = os.path.join(output_base_dir, "eval")
    test_dir = os.path.join(output_base_dir, "test")
    
    stats = {"train": 0, "eval": 0, "test": 0}
    
    for class_name, groups in class_groups.items():
        # Collect ALL images from all groups into a flat list
        all_images = []
        for original_path, transform_paths in groups:
            all_images.append(original_path)
            all_images.extend(transform_paths)
        
        # Shuffle the flat list
        random.shuffle(all_images)
        
        total_images = len(all_images)
        train_count = int(total_images * train_ratio)
        eval_count = int(total_images * eval_ratio)
        
        train_images = all_images[:train_count]
        eval_images = all_images[train_count:train_count + eval_count]
        test_images = all_images[train_count + eval_count:]
        
        # Create class directories and copy images
        for split_dir, split_images, split_name in [
            (train_dir, train_images, "train"),
            (eval_dir, eval_images, "eval"),
            (test_dir, test_images, "test")
        ]:
            class_split_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_split_dir, exist_ok=True)
            
            for img_path in split_images:
                try:
                    shutil.copy2(img_path, class_split_dir)
                    stats[split_name] += 1
                except Exception as e:
                    print(f"    Warning copying {img_path}: {e}")
        
        print(f"  Class '{class_name}': {len(train_images)} train, {len(eval_images)} eval, {len(test_images)} test")
    
    return stats

def print_statistics(class_images, class_groups, final_stats):
    """Print dataset statistics"""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    print("\nOriginal Dataset:")
    for class_name, images in sorted(class_images.items()):
        print(f"  {class_name}: {len(images)} images")
    print(f"  TOTAL: {sum(len(imgs) for imgs in class_images.values())} images")
    
    print("\nAugmented Dataset (before split):")
    for class_name, groups in sorted(class_groups.items()):
        total_images = sum(1 + len(tpaths) for _, tpaths in groups)
        print(f"  {class_name}: {len(groups)} groups ({total_images} images)")
    
    total_all = sum(final_stats.values())
    print("\nFinal Split:")
    print(f"  Train: {final_stats['train']} images ({100*final_stats['train']/total_all:.1f}%)")
    print(f"  Eval:  {final_stats['eval']} images ({100*final_stats['eval']/total_all:.1f}%)")
    print(f"  Test:  {final_stats['test']} images ({100*final_stats['test']/total_all:.1f}%)")
    print(f"  TOTAL: {total_all} images")
    print("="*60 + "\n")

def main():
    if len(sys.argv) != 2:
        print("Usage: ./augment_and_split.py <input_directory>")
        print("Images will be found recursively at any depth")
        print("Output will be saved to './augmented_directory'")
        return
    
    input_dir = sys.argv[1]
    output_dir = "augmented_directory"
    
    # Create output directory
    if os.path.exists(output_dir):
        response = input(f"Warning: '{output_dir}' already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect images by class (recursively at any depth)
    print("Collecting images recursively by class...")
    class_images = collect_images_by_class(input_dir)
    
    if not class_images:
        print("No images found or invalid directory structure.")
        return
    
    print(f"   Found {len(class_images)} classes")
    
    # Parameters
    transforms_per_image = 5
    max_workers = 8  # Number of threads for parallel processing
    
    # Find the maximum class size
    max_class_size = max(len(images) for images in class_images.values())
    
    # Target: make all classes the same size as the largest class
    images_per_group = 1 + transforms_per_image
    target_images_per_class = max_class_size * images_per_group
    
    print(f"\nTarget images per class: {target_images_per_class}")
    print(f"Using {max_workers} threads for processing")
    
    # Create temporary augmented directory
    temp_augmented_dir = os.path.join(output_dir, "temp_augmented")
    os.makedirs(temp_augmented_dir, exist_ok=True)
    
    # Augment each class
    print("\nAugmenting images...")
    class_groups = {}
    for class_name, image_paths in class_images.items():
        groups = augment_class(
            class_name, 
            image_paths, 
            target_images_per_class, 
            temp_augmented_dir, 
            transforms_per_image=transforms_per_image,
            max_workers=max_workers
        )
        class_groups[class_name] = groups
    
    # Split dataset
    print("\nSplitting dataset into train/eval/test...")
    final_stats = split_dataset(class_groups, output_dir, train_ratio=0.7, eval_ratio=0.15, test_ratio=0.15)
    
    # Clean up temporary directory
    shutil.rmtree(temp_augmented_dir)
    
    # Print statistics
    print_statistics(class_images, class_groups, final_stats)
    
    print(f"Dataset augmentation and splitting complete!")
    print(f"Output saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()