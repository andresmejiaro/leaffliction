from plantcv import plantcv as pcv
import cv2
import matplotlib.pyplot as plt

pcv.params.debug = None  # we control visualization manually

# --------------------------------
# Read image
# --------------------------------
img, _, _ = pcv.readimage("input.jpg")

outputs = {}

# ==============================================
# 1. Gaussian Blur
# ==============================================
blur = pcv.gaussian_blur(img=img, ksize=(5, 5), sigma_x=0)
outputs["Gaussian Blur"] = blur

# ==============================================
# 2. Mask (binary)
# ==============================================
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

lower = (25, 40, 40)
upper = (85, 255, 255)

mask = cv2.inRange(hsv, lower, upper)
mask = pcv.fill(mask, size=50)
outputs["Mask"] = mask

# ==============================================
# 3. ROI Objects (visual)
# ==============================================
contours, hierarchy = pcv.find_objects(img, mask)

roi, roi_hierarchy = pcv.roi.rectangle(
    img=img,
    x=0,
    y=0,
    h=mask.shape[0],
    w=mask.shape[1]
)

roi_objects, _, kept_mask, _ = pcv.roi_objects(
    img=img,
    roi_contour=roi,
    roi_hierarchy=roi_hierarchy,
    object_contours=contours,
    obj_hierarchy=hierarchy,
    roi_type="partial"
)

roi_vis = pcv.visualize_objects(
    img=img,
    objects=roi_objects,
    mask=kept_mask
)
outputs["ROI Objects"] = roi_vis

# ==============================================
# 4. Analyze Object (shape traits)
# ==============================================
analysis_img = pcv.analyze_object(
    img=img,
    obj=roi_objects[0],
    mask=kept_mask
)
outputs["Analyze Object"] = analysis_img

# ==============================================
# 5. Pseudo-landmarks
# ==============================================
pl_img = pcv.analyze_pseudolandmarks(
    img=img,
    obj=roi_objects[0],
    mask=kept_mask,
    label="plant"
)
outputs["Pseudo-landmarks"] = pl_img

# ==============================================
# 6. Color Histogram (visual)
# ==============================================
pcv.analyze_color(
    img=img,
    mask=kept_mask,
    colorspaces="hsv",
    label="plant"
)

hist_img = pcv.visualize_color_histograms(
    label="plant",
    colorspaces="hsv"
)
outputs["Color Histogram"] = hist_img

# ==============================================
# Visualization grid
# ==============================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for ax, (title, image) in zip(axes.flat, outputs.items()):
    if len(image.shape) == 2:
        ax.imshow(image, cmap="gray")
    else:
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()

'''
agaussian bluer
mask
roi objects
analyze object
pseudolandmarks
color histogram
'''
