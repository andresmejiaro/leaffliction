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

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")
print_lock = Lock()

def thread_safe_print(msg):
    with print_lock:
        print(msg)

def collect_images_by_class(root_path):
    class_images = defaultdict(list)
    
    if not os.path.isdir(root_path):
        print(f"Error: '{root_path}' is not a directory.")
        return class_images
    
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(IMG_EXTENSIONS):
                file_path = os.path.join(root, file)
                class_name = os.path.basename(root)
                class_images[class_name].append(file_path)
    
    return class_images

def apply_transformations(img):
    transforms = []
    width, height = img.size
    
    flipped = ImageOps.mirror(img)
    transforms.append(("flip", flipped))
    
    rotated = img.rotate(45, expand=True)
    transforms.append(("rotate", rotated))
    
    cropped = img.crop((width//4, height//4, 3*width//4, 3*height//4))
    transforms.append(("crop", cropped))
    
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
            original_output = os.path.join(output_dir, f"{group_id}_original.jpg")
            img.save(original_output)
            
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
    class_output_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)
    
    originals = list(image_paths)
    random.shuffle(originals)
    num_originals = len(originals)
    
    images_per_group = 1 + transforms_per_image
    
    num_groups_needed = math.ceil(target_total_images / images_per_group)
    
    if num_groups_needed <= num_originals:
        selected_originals = originals[:num_groups_needed]
        full_groups = target_total_images // images_per_group
        remainder = target_total_images % images_per_group
        
        thread_safe_print(f"  Class '{class_name}': Creating {num_groups_needed} groups ({target_total_images} images)")
    else:
        selected_originals = []
        for i in range(num_groups_needed):
            selected_originals.append(originals[i % num_originals])
        
        full_groups = target_total_images // images_per_group
        remainder = target_total_images % images_per_group
        
        thread_safe_print(f"  Class '{class_name}': Creating {num_groups_needed} groups from {num_originals} originals ({target_total_images} images)")
    
    jobs = []
    for idx, src_path in enumerate(selected_originals):
        base_name = os.path.splitext(os.path.basename(src_path))[0]
        group_id = f"{base_name}_g{idx}"
        
        if idx < full_groups:
            num_transforms = transforms_per_image
        elif idx == full_groups:
            num_transforms = max(0, remainder - 1)
        else:
            break
        
        jobs.append((src_path, class_output_dir, group_id, num_transforms))
    
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
    train_dir = os.path.join(output_base_dir, "train")
    eval_dir = os.path.join(output_base_dir, "eval")
    test_dir = os.path.join(output_base_dir, "test")
    
    stats = {"train": 0, "eval": 0, "test": 0}
    
    for class_name, groups in class_groups.items():
        random.shuffle(groups)
        
        total_groups = len(groups)
        train_count = int(total_groups * train_ratio)
        eval_count = int(total_groups * eval_ratio)
        
        train_groups = groups[:train_count]
        eval_groups = groups[train_count:train_count + eval_count]
        test_groups = groups[train_count + eval_count:]
        
        for split_dir, split_groups, split_name in [
            (train_dir, train_groups, "train"),
            (eval_dir, eval_groups, "eval"),
            (test_dir, test_groups, "test")
        ]:
            class_split_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_split_dir, exist_ok=True)
            
            for original_path, transform_paths in split_groups:
                try:
                    shutil.copy2(original_path, class_split_dir)
                    stats[split_name] += 1
                except Exception as e:
                    print(f"    Warning copying original {original_path}: {e}")
                
                for transform_path in transform_paths:
                    try:
                        shutil.copy2(transform_path, class_split_dir)
                        stats[split_name] += 1
                    except Exception as e:
                        print(f"    Warning copying transform {transform_path}: {e}")
        
        train_images = sum(1 + len(tpaths) for _, tpaths in train_groups)
        eval_images = sum(1 + len(tpaths) for _, tpaths in eval_groups)
        test_images = sum(1 + len(tpaths) for _, tpaths in test_groups)
        
        print(f"  Class '{class_name}': {len(train_groups)} groups ({train_images} images) train, {len(eval_groups)} groups ({eval_images} images) eval, {len(test_groups)} groups ({test_images} images) test")
    
    return stats

def print_statistics(class_images, class_groups, final_stats):
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
    
    if os.path.exists(output_dir):
        response = input(f"Warning: '{output_dir}' already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Collecting images recursively by class...")
    class_images = collect_images_by_class(input_dir)
    
    if not class_images:
        print("No images found or invalid directory structure.")
        return
    
    print(f"   Found {len(class_images)} classes")
    
    transforms_per_image = 5
    max_workers = 8
    max_class_size = max(len(images) for images in class_images.values())
    images_per_group = 1 + transforms_per_image
    target_images_per_class = max_class_size * images_per_group
    
    print(f"\nTarget images per class: {target_images_per_class}")
    print(f"Using {max_workers} threads for processing")
    temp_augmented_dir = os.path.join(output_dir, "temp_augmented")
    os.makedirs(temp_augmented_dir, exist_ok=True)
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
    
    print("\nSplitting dataset into train/eval/test...")
    final_stats = split_dataset(class_groups, output_dir, train_ratio=0.7, eval_ratio=0.15, test_ratio=0.15)
    shutil.rmtree(temp_augmented_dir)
    print_statistics(class_images, class_groups, final_stats)
    print(f"Dataset augmentation and splitting complete!")
    print(f"Output saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()