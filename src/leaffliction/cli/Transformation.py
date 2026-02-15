#!/usr/bin/env python3
import argparse
import os
import sys

from plantcv import plantcv as pcv

from leaffliction.Transformation import process_directory, process_single_image, visualize_results


pcv.params.debug = None

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply plant transformations to images"
    )
    parser.add_argument(
        "image",
        type=str,
        nargs="?",
        help="Path to single image file"
    )
    parser.add_argument(
        "-src", "--source",
        type=str,
        help="Source directory containing images"
    )
    parser.add_argument(
        "-dst", "--destination",
        type=str,
        help="Destination directory for transformed images"
    )
    
    args = parser.parse_args()
    
    if args.source and args.destination:
        if not os.path.isdir(args.source):
            print(f"Error: Source directory '{args.source}' not found.")
            sys.exit(1)
        print(f"Processing directory: {args.source}")
        process_directory(args.source, args.destination)
        print(f"Transformations saved to: {args.destination}")
    
    elif args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image file '{args.image}' not found.")
            sys.exit(1)
        print(f"Processing image: {args.image}")
        outputs = process_single_image(args.image)
        print("Displaying results...")
        visualize_results(outputs)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
