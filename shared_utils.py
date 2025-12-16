#!/usr/bin/env python3
import os
import random
import math
import shutil
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from PIL import Image, ImageOps, ImageEnhance
import csv
import polars as pl


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


def apply_flip(img):
    return ImageOps.mirror(img)


def apply_rotation(img):
    return img.rotate(45, expand=True)


def apply_crop(img):
    width, height = img.size
    return img.crop((width//4, height//4, 3*width//4, 3*height//4))


def apply_shear(img):
    width, height = img.size
    m = -0.5
    xshift = abs(m) * height
    new_width = width + int(round(xshift))
    sheared = img.transform(
        (new_width, height),
        Image.AFFINE,
        (1, m, -xshift if m > 0 else 0, 0, 1, 0),
        resample=Image.BICUBIC
    )
    return sheared


def apply_distortion(img):
    coeffs = [1, 0.2, 0, 0.2, 1, 0]
    return img.transform(img.size, Image.AFFINE, coeffs, resample=Image.BICUBIC)


def apply_brightness(img):
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(1.3)


def get_augmentation_transforms():
    return [
        ("flip", apply_flip),
        ("rotate", apply_rotation),
        ("crop", apply_crop),
        ("shear", apply_shear),
        ("distortion", apply_distortion)
    ]


def process_single_image_augmentation(args):
    src_path, output_dir, group_id, num_transforms = args
    
    try:
        with Image.open(src_path) as img:
            original_output = os.path.join(output_dir, f"{group_id}_original.jpg")
            img.save(original_output)
            
            transforms = get_augmentation_transforms()
            transform_paths = []
            
            for tname, tfunc in transforms[:num_transforms]:
                timg = tfunc(img)
                transform_output = os.path.join(
                    output_dir, f"{group_id}_{tname}.jpg"
                )
                timg.save(transform_output)
                transform_paths.append(transform_output)
            
            return (original_output, transform_paths)
    except Exception as e:
        thread_safe_print(f"Warning: failed to process {src_path}: {e}")
        return None


def augment_class(class_name, image_paths, target_total_images,
                   output_dir, transforms_per_image=5, max_workers=4):
    class_output_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)
    
    originals = list(image_paths)
    random.shuffle(originals)
    num_originals = len(originals)
    
    images_per_group = 1 + transforms_per_image
    num_groups_needed = math.ceil(target_total_images / images_per_group)
    
    if num_groups_needed <= num_originals:
        selected_originals = originals[:num_groups_needed]
    else:
        selected_originals = []
        for i in range(num_groups_needed):
            selected_originals.append(originals[i % num_originals])
    
    full_groups = target_total_images // images_per_group
    remainder = target_total_images % images_per_group
    
    thread_safe_print(
        f"  Class '{class_name}': Creating {num_groups_needed} groups "
        f"({target_total_images} images)"
    )
    
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
        futures = [
            executor.submit(process_single_image_augmentation, job)
            for job in jobs
        ]
        
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                image_groups.append(result)
    
    total_images = sum(1 + len(tpaths) for _, tpaths in image_groups)
    thread_safe_print(
        f"    Created {len(image_groups)} groups ({total_images} images)"
    )
    
    return image_groups


def split_dataset(class_groups, output_base_dir,
                   train_ratio=0.7, eval_ratio=0.15, test_ratio=0.15):
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
                    print(f"Warning copying original {original_path}: {e}")
                
                for transform_path in transform_paths:
                    try:
                        shutil.copy2(transform_path, class_split_dir)
                        stats[split_name] += 1
                    except Exception as e:
                        print(f"Warning copying transform {transform_path}: {e}")
    
    return stats


def _process_path(p: Path, exts: set):
    if p.is_file() and p.suffix.lower() in exts:
        parts = p.parts
        split = None
        for part in parts:
            if part in ['train', 'eval', 'test']:
                split = part
                break
        
        group_stem = p.stem.split("_g")[0] if "_g" in p.stem else p.stem.split("_")[0]
        
        return {
            "path": str(p.resolve()),
            "name": p.name,
            "class": p.parent.name,
            "stem": p.stem,
            "group": group_stem,
            "split": split if split else "unknown",
        }
    return None


def build_csv_from_directory(root_dir, output_csv, max_workers=8):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
    paths = list(Path(root_dir).rglob("*"))
    rows = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(lambda p: _process_path(p, exts), paths):
            if result:
                rows.append(result)
    
    if not rows:
        raise ValueError(f"No images found in {root_dir}")
    
    df = pl.from_records(rows)
    df.write_csv(output_csv)
    print(f"CSV saved to {output_csv}")
    return df
