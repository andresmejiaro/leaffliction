import argparse
import sys

from PIL import UnidentifiedImageError

from leaffliction.Augmentation import augment_image, display_augmentations


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
        print(f"Image file not found {args.image}") 
        sys.exit(1)
    except PermissionError as e:
        print(f"No permission to read image file {args.image}") 
        sys.exit(1)
    except UnidentifiedImageError as e:
        print(f"File is not a valid Image {args.image}")
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
