import argparse
import os


from leaffliction.distribution import collect_images_by_class, print_statistics, visualize_distribution


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize image distribution by class"
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing images organized by class"
    )
    
    args = parser.parse_args()
    
    directory_name = os.path.basename(os.path.normpath(args.directory))
    
    print(f"Scanning directory: {args.directory}")
    class_images = collect_images_by_class(args.directory)
    
    if not class_images:
        print("No images found.")
        return
    
    print(f"Found {len(class_images)} classes")
    print_statistics(class_images)
    visualize_distribution(class_images, directory_name)


if __name__ == "__main__":
    main()
