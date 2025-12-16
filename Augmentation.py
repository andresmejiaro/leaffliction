#!/usr/bin/env python3
import argparse
import os
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


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
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(1.3)


def get_augmentations():
    return [
        ("flip", apply_flip),
        ("rotate", apply_rotation),
        ("crop", apply_crop),
        ("shear", apply_shear),
        ("distortion", apply_distortion),
        ("brightness", apply_brightness)
    ]


def augment_image(image_path, output_dir=None):
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return []
    
    img = Image.open(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    augmentations = get_augmentations()
    augmented_images = []
    saved_paths = []
    
    for aug_name, aug_func in augmentations:
        aug_img = aug_func(img)
        output_path = os.path.join(output_dir, f"{base_name}_{aug_name}.jpg")
        aug_img.save(output_path)
        augmented_images.append((aug_name, aug_img))
        saved_paths.append(output_path)
        print(f"Saved: {output_path}")
    
    return augmented_images, saved_paths


def display_augmentations(original_path, augmented_images):
    original_img = Image.open(original_path)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Data Augmentations: {os.path.basename(original_path)}")
    
    for idx, (aug_name, aug_img) in enumerate(augmented_images):
        row = idx // 3
        col = idx % 3
        axes[row, col].imshow(aug_img)
        axes[row, col].set_title(aug_name.capitalize())
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Apply 6 data augmentations to an image"
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to the image file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory (default: same as input image)"
    )
    
    args = parser.parse_args()
    
    print(f"Augmenting image: {args.image}")
    augmented_images, saved_paths = augment_image(args.image, args.output)
    
    if augmented_images:
        print(f"\nCreated {len(augmented_images)} augmented images")
        display_augmentations(args.image, augmented_images)


if __name__ == "__main__":
    main()
