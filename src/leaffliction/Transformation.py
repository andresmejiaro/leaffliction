import os

import cv2
import numpy as np
import plotly.graph_objects as go
from plantcv import plantcv as pcv
from plotly.subplots import make_subplots


def apply_gaussian_blur(img: np.ndarray) -> np.ndarray:
    return pcv.gaussian_blur(img=img, ksize=(5, 5), sigma_x=0)


def create_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower = np.array([25, 40, 40])
    upper = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = pcv.fill(mask, size=50)
    mask = pcv.erode(mask, ksize=3, i=1)
    mask = pcv.dilate(mask, ksize=3, i=1)
    return mask


def extract_roi_objects(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    roi = pcv.roi.rectangle(
        img=img, x=0, y=0, h=mask.shape[0], w=mask.shape[1]
    )
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


def analyze_shape(img: np.ndarray, contours: list[np.ndarray], mask: np.ndarray) -> np.ndarray:
    if len(contours) > 0:
        analysis_img = pcv.analyze.size(
            img=img, labeled_mask=mask, n_labels=1, label="plant"
        )
        return analysis_img
    return img.copy()


def _contour_center(contour: np.ndarray) -> tuple[int, int] | None:
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return cx, cy


def _draw_landmark(
    image: np.ndarray,
    point: tuple[int, int],
    label: str,
    color: tuple[int, int, int],
) -> None:
    cv2.circle(image, point, 6, color, -1)
    cv2.putText(
        image,
        label,
        (point[0] + 8, point[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def analyze_landmarks(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if cv2.countNonZero(mask) == 0:
        return img.copy()

    landmark_img = img.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return landmark_img

    contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(landmark_img, [contour], -1, (255, 255, 0), 2)

    leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
    rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
    topmost = tuple(contour[contour[:, :, 1].argmin()][0])
    bottommost = tuple(contour[contour[:, :, 1].argmax()][0])

    for label, point in {
        "L": leftmost,
        "R": rightmost,
        "T": topmost,
        "B": bottommost,
    }.items():
        _draw_landmark(landmark_img, point, label, (0, 255, 255))

    center = _contour_center(contour)
    if center is not None:
        _draw_landmark(landmark_img, center, "C", (255, 255, 255))

    try:
        skeleton = pcv.morphology.skeletonize(mask=mask)
        pruned_skeleton, _, _ = pcv.morphology.prune(skeleton, size=5, mask=mask)
        if cv2.countNonZero(pruned_skeleton) > 0:
            skeleton = pruned_skeleton
        tip_mask = pcv.morphology.find_tips(skeleton)

        landmark_img[skeleton > 0] = (0, 255, 0)
        tip_contours, _ = cv2.findContours(
            tip_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for tip_index, tip_contour in enumerate(tip_contours, start=1):
            tip_center = _contour_center(tip_contour)
            if tip_center is None:
                continue
            _draw_landmark(landmark_img, tip_center, f"P{tip_index}", (255, 0, 0))
    except Exception:
        pass

    return landmark_img


def create_masked_color(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if cv2.countNonZero(mask) > 0:
        try:
            pcv.analyze.color(
                rgb_img=img, mask=mask, colorspaces="hsv", label="plant"
            )
            return cv2.bitwise_and(img, img, mask=mask)
        except Exception:
            return cv2.bitwise_and(img, img, mask=mask)
    return img.copy()


def process_single_image(image_path: str) -> dict[str, np.ndarray]:
    img, _, _ = pcv.readimage(image_path)
    outputs = {}
    
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
    
    return outputs


def visualize_results(outputs: dict[str, np.ndarray]) -> None:
    subplot_titles = [title for title, _ in list(outputs.items())[:6]]
    fig = make_subplots(rows=2, cols=3, subplot_titles=subplot_titles)

    for idx, (_, image) in enumerate(list(outputs.items())[:6]):
        row = idx // 3 + 1
        col = idx % 3 + 1
        if image.ndim == 2:
            binary_mask = (image > 0).astype(np.uint8) * 255
            display_image = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
        else:
            display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fig.add_trace(go.Image(z=display_image), row=row, col=col)
        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, row=row, col=col)

    fig.update_layout(
        title="Transformations",
        margin=dict(l=0, r=0, t=60, b=0),
        height=800,
        width=1200,
    )
    fig.show()


def save_transformations(outputs: dict[str, np.ndarray], output_dir: str, base_name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    for title, image in outputs.items():
        transform_name = title.split(". ")[1].lower().replace(" ", "_")
        output_path = os.path.join(
            output_dir, f"{base_name}_{transform_name}.jpg"
        )
        cv2.imwrite(output_path, image)
        print(f"Saved: {output_path}")


def process_directory(src_dir: str, dst_dir: str) -> None:
    img_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")
    
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith(img_extensions):
                image_path = os.path.join(root, file)
                print(f"Processing: {image_path}")
                
                try:
                    outputs = process_single_image(image_path)
                    base_name = os.path.splitext(file)[0]
                    save_transformations(outputs, dst_dir, base_name)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
