#!/usr/bin/env python3
import argparse
import os
from PIL import ImageEnhance
from PIL import Image, ImageOps, UnidentifiedImageError
import matplotlib.pyplot as plt
from typing import Callable
import sys


def apply_flip(img: Image.Image) -> Image.Image:
    """
    Return a mirrored (horizontal) copy of the input image.

    Args:
        img: A PIL Image to be flipped

    Returns:
        A new PIL flipped image
    """
    return ImageOps.mirror(img)


def apply_rotation(img: Image.Image) -> Image.Image:
    """
    Return a 45 degree counterclockwise rotated copy of the input image.

    Args:
        img: A PIL Image to be rotated

    Returns:
        A new PIL rotated image
    """
    return img.rotate(45, expand=True)


def apply_crop(img: Image.Image) -> Image.Image:
    """
    Return a cropped copy of the input image. (the center 1/4th of the image)

    Args:
        img: A PIL Image to be cropped

    Returns:
        A new PIL cropped image
    """
    width, height = img.size
    return img.crop((width//4, height//4, 3*width//4, 3*height//4))


def apply_shear(img: Image.Image) -> Image.Image:
    """
    Return a sheared copy of the input image.
    The shear is equation is done by
    x' = 1*x - 0.5*y + xshift
    y' = 0*x + 1*y + 0 (y unaffected)


    Args:
        img: A PIL Image to be sheared

    Returns:
        A new PIL sheared image
    """
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


def apply_distortion(img: Image.Image) -> Image.Image:
    """
    Return a distorted copy of the input image.
    The distorted equation is done by
    x' = x + 0.2*y
    y' = 0.2*x + y

    Args:
        img: A PIL Image to be distorted

    Returns:
        A new PIL distorted image
    """
    coeffs = [1, 0.2, 0, 0.2, 1, 0]
    return img.transform(img.size,
                         Image.AFFINE, coeffs,
                         resample=Image.BICUBIC)


def apply_brightness(img: Image.Image) -> Image.Image:
    """
    Return brighter copy of the input image.

    Brightness increase 30%
    Args:
        img: A PIL Image

    Returns:
        A new PIL image
    """
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(1.3)


def get_augmentations() -> list[
        tuple[str, Callable[[Image.Image], Image.Image]]]:
    """
    Return the list of augmentations of the pipeline.

    Args:
        None

    Returns:
        A list of tuples with name and function
    """
    return [
        ("flip", apply_flip),
        ("rotate", apply_rotation),
        ("crop", apply_crop),
        ("shear", apply_shear),
        ("distortion", apply_distortion),
        ("brightness", apply_brightness)
    ]


def augment_image(image_path: str, output_dir: str | None = None) -> tuple[
        list[tuple[str, Image.Image]], list[str]]:
    """
    Creates an augmentation of a single file given by the
    get_augmentations function.


    :param image_path: Description
    :type image_path: str
    :param output_dir: Description
    :type output_dir: str | None
    :return: Description
    :rtype: tuple[list[tuple[str, Image.Image]], list[str]]
    """
    try:
        with Image.open(image_path) as img:
            img = img.copy()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Image file not found {image_path}") from e
    except PermissionError as e:
        raise PermissionError(
            f"No permission to read image file {image_path}") from e
    except UnidentifiedImageError as e:
        raise ValueError(f"File is not a valid Image {image_path}") from e

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    if output_dir is None:
        output_dir = os.path.dirname(image_path)

    os.makedirs(output_dir, exist_ok=True)

    augmentations = get_augmentations()
    augmented_images = []
    saved_paths = []

    for aug_name, aug_func in augmentations:
        aug_img = aug_func(img)
        output_path = os.path.join(output_dir, f"{base_name}_{aug_name}.JPG")
        aug_img.save(output_path)
        augmented_images.append((aug_name, aug_img))
        saved_paths.append(output_path)
        print(f"Saved: {output_path}")

    return augmented_images, saved_paths


def display_augmentations(original_path: str, augmented_images: list[
        tuple[str, Image.Image]]):
    """
    Docstring for display_augmentations
    Show augmentation in screen


    :param original_path: Description
    :type original_path: str
    :param augmented_images: Description
    :type augmented_images: list[tuple[str, Image.Image]]
    """
    try:
        with Image.open(original_path) as original_img:
            original_img = original_img.copy()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Image file not found {original_path}") from e
    except PermissionError as e:
        raise PermissionError(
            f"No permission to read image file {original_path}") from e
    except UnidentifiedImageError as e:
        raise ValueError(f"File is not a valid Image {original_path}") from e
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
    """
    Entry point for the image augmentation CLI.

    Parses command-line arguments, applies a fixed set of data augmentations
    to the input image, saves the augmented images to disk, and optionally
    displays the results. All file and image-related errors are handled
    gracefully to prevent unhandled exceptions.
    """
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
    try:
        augmented_images, _ = augment_image(args.image, args.output)
    except FileNotFoundError as e:
        print(f"File {e} not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error {e} occurred")
        sys.exit(1)

    if augmented_images:
        print(f"\nCreated {len(augmented_images)} augmented images")
        try:
            display_augmentations(args.image, augmented_images)
        except FileNotFoundError as e:
            print(f"File {e} not found")
            sys.exit(1)
        except Exception as e:
            print(f"Error {e} occurred")
            sys.exit(1)


if __name__ == "__main__":
    main()
