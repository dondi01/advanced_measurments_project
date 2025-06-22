import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent

def get_orientation_angle_and_rectangle(contour):
    """Get the orientation angle and bounding rectangle of a contour."""
    rect = cv2.minAreaRect(contour)
    (center_x, center_y), (width, height), angle = rect

    # Normalize angle to [-90, 90)
    if width < height:
        main_axis_angle = angle
    else:
        main_axis_angle = angle + 90

    main_axis_angle = main_axis_angle % 180
    if main_axis_angle >= 90:
        main_axis_angle -= 180

    center = (int(center_x), int(center_y))
    return main_axis_angle, center, rect

def find_best_alignment_angle(base_angle, test_angle):
    """Find the axis (0°, 90°, -90°) that minimizes the total rotation for both masks."""
    possible_axes = [0, 90, -90]  # Axes to align to
    best_axis = None
    min_total_rotation = float('inf')

    for axis in possible_axes:
        # Compute the total rotation required to align both masks to this axis
        total_rotation = abs(base_angle - axis) + abs(test_angle - axis)
        if total_rotation < min_total_rotation:
            min_total_rotation = total_rotation
            best_axis = axis

    return best_axis

def get_main_object_contour(contours, image_shape, area_thresh=0.9):
    """Get the main object contour based on area filtering."""
    img_area = image_shape[0] * image_shape[1]
    filtered = [c for c in contours if cv2.contourArea(c) < area_thresh * img_area]
    if not filtered:
        return None
    return max(filtered, key=cv2.contourArea)

def preprocess(image_path):
    """Preprocess the image to extract contours and thresholded mask."""
    image = cv2.imread(image_path)
    lightened = cv2.convertScaleAbs(image, alpha=1, beta=100)
    gray = cv2.cvtColor(lightened, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return image, contours, thresh

def align_image_to_angle(img, contours, target_angle):
    """Align the image to a specific target angle and center the rectangle."""
    main_contour = get_main_object_contour(contours, img.shape)
    if main_contour is None or len(main_contour) < 5:
        return img, None, None

    # Get the orientation angle and bounding rectangle
    angle, rect_center, rect = get_orientation_angle_and_rectangle(main_contour)

    # Rotate the image to align with the target angle
    rotation_angle = angle - target_angle
    (h, w) = img.shape[:2]
    img_center = (w // 2, h // 2)

    # Compute the rotation matrix
    M_rot = cv2.getRotationMatrix2D(rect_center, rotation_angle, 1.0)

    # Compute the translation to center the rectangle
    dx = img_center[0] - int(rect_center[0])
    dy = img_center[1] - int(rect_center[1])

    # Combine rotation and translation into a single matrix
    M_rot[0, 2] += dx
    M_rot[1, 2] += dy

    # Apply the combined transformation
    aligned_img = cv2.warpAffine(img, M_rot, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

    return aligned_img, rect, main_contour

def center_crop(img, target_shape):
    """Center crop the image to the target shape."""
    h, w = img.shape[:2]
    th, tw = target_shape
    y1 = max((h - th) // 2, 0)
    x1 = max((w - tw) // 2, 0)
    y2 = y1 + th
    x2 = x1 + tw
    return img[y1:y2, x1:x2]

# Start timing
start = time.time()

# Paths to your images
base_path = str(project_root / "Reconstructed" / "green_ok.png")
test_path = str(project_root / "dataset_piccoli" / "dezoommata_green_cut.png")

# Preprocess both images
base_img, base_contours, base_thresh = preprocess(base_path)
test_img, test_contours, test_thresh = preprocess(test_path)

# Compute the angles of the main contours
base_angle, _, _ = get_orientation_angle_and_rectangle(get_main_object_contour(base_contours, base_thresh.shape))
test_angle, _, _ = get_orientation_angle_and_rectangle(get_main_object_contour(test_contours, test_thresh.shape))

# Find the best alignment axis to minimize total rotation
best_axis = find_best_alignment_angle(base_angle, test_angle)
print(f"Best alignment axis: {best_axis+90}°")

# Align both masks to the best axis
aligned_base_thresh, base_rect, base_main_contour = align_image_to_angle(base_thresh, base_contours, best_axis)
aligned_test_thresh, test_rect, test_main_contour = align_image_to_angle(test_thresh, test_contours, best_axis)

# Use main object rectangles for scaling
if base_rect is not None and test_rect is not None:
    # Get width and height (sorted so width <= height)
    base_w, base_h = sorted(base_rect[1])
    test_w, test_h = sorted(test_rect[1])

    # Compute scaling ratios
    ratio_w = base_w / test_w if test_w != 0 else 1
    ratio_h = base_h / test_h if test_h != 0 else 1

    # Resize test mask to match base mask's object size
    new_w = int(aligned_test_thresh.shape[1] * ratio_w)
    new_h = int(aligned_test_thresh.shape[0] * ratio_h)
    aligned_test_thresh = cv2.resize(aligned_test_thresh, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Center crop or pad to match base mask size
    target_shape = aligned_base_thresh.shape[:2]
    if aligned_test_thresh.shape[0] > target_shape[0] or aligned_test_thresh.shape[1] > target_shape[1]:
        aligned_test_thresh = center_crop(aligned_test_thresh, target_shape)
    elif aligned_test_thresh.shape[0] < target_shape[0] or aligned_test_thresh.shape[1] < target_shape[1]:
        pad_vert = (target_shape[0] - aligned_test_thresh.shape[0]) // 2
        pad_horz = (target_shape[1] - aligned_test_thresh.shape[1]) // 2
        aligned_test_thresh = cv2.copyMakeBorder(
            aligned_test_thresh,
            pad_vert, target_shape[0] - aligned_test_thresh.shape[0] - pad_vert,
            pad_horz, target_shape[1] - aligned_test_thresh.shape[1] - pad_horz,
            cv2.BORDER_CONSTANT, value=0
        )

# Ensure the masks have the same size
if aligned_base_thresh.shape != aligned_test_thresh.shape:
    min_h = min(aligned_base_thresh.shape[0], aligned_test_thresh.shape[0])
    min_w = min(aligned_base_thresh.shape[1], aligned_test_thresh.shape[1])
    aligned_base_thresh = center_crop(aligned_base_thresh, (min_h, min_w))
    aligned_test_thresh = center_crop(aligned_test_thresh, (min_h, min_w))

# Compute absolute difference between the two masks
diff_mask = cv2.absdiff(aligned_base_thresh, aligned_test_thresh)

# Optional: clean up small noise
kernel = np.ones((21, 21), np.uint8)
diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)

#contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# min_area = 200  # Adjust as needed
# clean_mask = np.zeros_like(diff_mask)
# for cnt in contours:
#     if cv2.contourArea(cnt) > min_area:
#         cv2.drawContours(clean_mask, [cnt], -1, 255, -1)
# #diff_mask = clean_mask
print(f"Execution time: {time.time() - start:.2f} seconds")
print("diff mask has %d non zero pixels out of %d" % (np.count_nonzero(diff_mask), diff_mask.size))

# Plot the aligned base mask, aligned test mask, and difference mask
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(aligned_base_thresh, cmap='gray')
plt.title("Aligned Base Mask")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(aligned_test_thresh, cmap='gray')
plt.title("Aligned Test Mask")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(diff_mask, cmap='gray')
plt.title("Difference Mask")
plt.axis('off')

plt.tight_layout()
plt.show()

# Overlay the two masks to visualize differences
overlay = np.zeros((aligned_base_thresh.shape[0], aligned_base_thresh.shape[1], 3), dtype=np.uint8)

# Red for base mask, green for test mask, yellow for overlap
overlay[(aligned_base_thresh == 0) & (aligned_test_thresh > 0)] = [255, 0, 0]  # Red: Base mask only
overlay[(aligned_test_thresh == 0) & (aligned_base_thresh > 0)] = [0, 255, 0]  # Green: Test mask only
overlay[(aligned_base_thresh == aligned_test_thresh)] = [255, 255, 0]  # Yellow: Overlap

# Plot the overlay
plt.figure(figsize=(7, 7))
plt.imshow(overlay)
plt.title("Overlay of Base and Test Masks")
plt.axis('off')
plt.show()