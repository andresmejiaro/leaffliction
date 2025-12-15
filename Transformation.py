#!/usr/bin/env python3

from plantcv import plantcv as pcv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

pcv.params.debug = None


def apply_gaussian_blur(img):
    return pcv.gaussian_blur(img=img, ksize=(5, 5), sigma_x=0)


def create_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([25, 40, 40])
    upper = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = pcv.fill(mask, size=50)
    mask = pcv.erode(mask, ksize=3, i=1)
    mask = pcv.dilate(mask, ksize=3, i=1)
    return mask


def extract_roi_objects(img, mask):
    roi = pcv.roi.rectangle(img=img, x=0, y=0, h=mask.shape[0], w=mask.shape[1])
    
    filtered_mask = pcv.roi.filter(mask=mask, roi=roi, roi_type="partial")
    
    contours, _ = cv2.findContours(
        filtered_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    roi_vis = img.copy()
    if len(contours) > 0:
        cv2.drawContours(roi_vis, contours, -1, (0, 255, 0), 2)
    
    return roi_vis, contours, filtered_mask


def analyze_shape(img, contours, mask):
    if len(contours) > 0:
        analysis_img = pcv.analyze.size(img=img, labeled_mask=mask, n_labels=1, label="plant")
        return analysis_img
    return img.copy()


def analyze_landmarks(img, mask):
    if cv2.countNonZero(mask) > 0:
        try:
            pl_img = pcv.morphology.check_cycles(mask=mask)
            return img.copy()
        except:
            return img.copy()
    return img.copy()


def create_masked_color(img, mask):
    if cv2.countNonZero(mask) > 0:
        try:
            pcv.analyze.color(rgb_img=img, mask=mask, colorspaces="hsv", label="plant")
            return cv2.bitwise_and(img, img, mask=mask)
        except:
            return cv2.bitwise_and(img, img, mask=mask)
    return img.copy()


def visualize_results(outputs):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for ax, (title, image) in zip(axes.flat, outputs.items()):
        if len(image.shape) == 2:
            ax.imshow(image, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()


def process_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        sys.exit(1)
    
    print(f"Processing image: {image_path}")
    
    img, _, _ = pcv.readimage(image_path)
    outputs = {}
    
    print("Applying transformations...")
    
    blur = apply_gaussian_blur(img)
    outputs["1. Gaussian Blur"] = blur
    
    mask = create_mask(blur)
    outputs["2. Mask"] = mask
    
    roi_vis, contours, filtered_mask = extract_roi_objects(img, mask)
    outputs["3. ROI Objects"] = roi_vis
    
    analysis_img = analyze_shape(img, contours, filtered_mask)
    outputs["4. Analyze Object"] = analysis_img
    
    pl_img = analyze_landmarks(img, filtered_mask)
    outputs["5. Pseudo-landmarks"] = pl_img
    
    masked_color = create_masked_color(img, filtered_mask)
    outputs["6. Masked Color"] = masked_color
    
    print("Displaying results...")
    visualize_results(outputs)


def main():
    if len(sys.argv) != 2:
        print("Usage: python Transformation.py <image_path>")
        print("Example: python Transformation.py input.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    process_image(image_path)


if __name__ == "__main__":
    main()
