import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent

def get_orientation_angle_and_rectangle(contour):
    rect = cv2.minAreaRect(contour)
    (center_x, center_y), (width, height), angle = rect
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
    possible_axes = [0, 90, -90]
    best_axis = None
    min_total_rotation = float('inf')
    for axis in possible_axes:
        total_rotation = abs(base_angle - axis) + abs(test_angle - axis)
        if total_rotation < min_total_rotation:
            min_total_rotation = total_rotation
            best_axis = axis
    return best_axis

def get_main_object_contour(contours, image_shape, area_thresh=0.9):
    img_area = image_shape[0] * image_shape[1]
    filtered = [c for c in contours if cv2.contourArea(c) < area_thresh * img_area]
    if not filtered:
        return None
    return max(filtered, key=cv2.contourArea)

def preprocess_for_canny(image_path):
    image = cv2.imread(image_path)
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # Standard threshold for contour finding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Canny mask as in Prove.py
    canny = cv2.Canny(blurred, 10, 50)
    
    return canny, contours, gray

def align_image_to_angle(img, contours, target_angle):
    main_contour = get_main_object_contour(contours, img.shape)
    if main_contour is None or len(main_contour) < 5:
        return img, None, None
    angle, rect_center, rect = get_orientation_angle_and_rectangle(main_contour)
    rotation_angle = angle - target_angle
    (h, w) = img.shape[:2]
    img_center = (w // 2, h // 2)
    M_rot = cv2.getRotationMatrix2D(rect_center, rotation_angle, 1.0)
    dx = img_center[0] - int(rect_center[0])
    dy = img_center[1] - int(rect_center[1])
    M_rot[0, 2] += dx
    M_rot[1, 2] += dy
    aligned_img = cv2.warpAffine(img, M_rot, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    return aligned_img, rect, main_contour

def center_crop(img, target_shape):
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
base_path = str(project_root / "Schematics" / "green.png")
test_path = str(project_root / "Reconstructed" / "green_ok.png")

# Preprocess both images for Canny and contours

base_canny, base_contours, base_gray = preprocess_for_canny(base_path)
test_canny, test_contours, test_gray = preprocess_for_canny(test_path)

# Compute the angles of the main contours
base_angle, _, base_rect = get_orientation_angle_and_rectangle(get_main_object_contour(base_contours, base_gray.shape))
test_angle, _, test_rect = get_orientation_angle_and_rectangle(get_main_object_contour(test_contours, test_gray.shape))

# Find the best alignment axis to minimize total rotation
best_axis = find_best_alignment_angle(base_angle, test_angle)
print(f"Best alignment axis: {best_axis+90}°")

# Align both Canny masks to the best axis
aligned_base_canny, base_rect, base_main_contour = align_image_to_angle(base_canny, base_contours, best_axis)
aligned_test_canny, test_rect, test_main_contour = align_image_to_angle(test_canny, test_contours, best_axis)

# Use main object rectangles for scaling
if base_rect is not None and test_rect is not None:
    base_w, base_h = sorted(base_rect[1])
    test_w, test_h = sorted(test_rect[1])
    ratio_w = base_w / test_w if test_w != 0 else 1
    ratio_h = base_h / test_h if test_h != 0 else 1
    new_w = int(aligned_test_canny.shape[1] * ratio_w)
    new_h = int(aligned_test_canny.shape[0] * ratio_h)
    aligned_test_canny = cv2.resize(aligned_test_canny, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    target_shape = aligned_base_canny.shape[:2]
    if aligned_test_canny.shape[0] > target_shape[0] or aligned_test_canny.shape[1] > target_shape[1]:
        aligned_test_canny = center_crop(aligned_test_canny, target_shape)
    elif aligned_test_canny.shape[0] < target_shape[0] or aligned_test_canny.shape[1] < target_shape[1]:
        pad_vert = (target_shape[0] - aligned_test_canny.shape[0]) // 2
        pad_horz = (target_shape[1] - aligned_test_canny.shape[1]) // 2
        aligned_test_canny = cv2.copyMakeBorder(
            aligned_test_canny,
            pad_vert, target_shape[0] - aligned_test_canny.shape[0] - pad_vert,
            pad_horz, target_shape[1] - aligned_test_canny.shape[1] - pad_horz,
            cv2.BORDER_CONSTANT, value=0
        )

# Ensure the masks have the same size
if aligned_base_canny.shape != aligned_test_canny.shape:
    min_h = min(aligned_base_canny.shape[0], aligned_test_canny.shape[0])
    min_w = min(aligned_base_canny.shape[1], aligned_test_canny.shape[1])
    aligned_base_canny = center_crop(aligned_base_canny, (min_h, min_w))
    aligned_test_canny = center_crop(aligned_test_canny, (min_h, min_w))

# Dilate each mask
kernel = np.ones((9,9), np.uint8)
dil_base = cv2.dilate(aligned_base_canny, kernel, iterations=1)
dil_test = cv2.dilate(aligned_test_canny, kernel, iterations=1)

# "Missed" = base edge not matched by test (even with band)
missed = (aligned_base_canny > 0) & (dil_test == 0)
# "Extra" = test edge not matched by base (even with band)
extra = (aligned_test_canny > 0) & (dil_base == 0)
# "Matched" = edge in base or test, and found in the other's band
matched = ((aligned_base_canny > 0) & (dil_test > 0)) | ((aligned_test_canny > 0) & (dil_base > 0))

print("Missed:", np.count_nonzero(missed))
print("Extra:", np.count_nonzero(extra))
print("Matched:", np.count_nonzero(matched))

# Visualization
overlay = np.zeros((aligned_base_canny.shape[0], aligned_base_canny.shape[1], 3), dtype=np.uint8)
overlay[missed] = [255, 0, 0]   # Red: missed (base only)
overlay[extra] = [0, 255, 0]    # Green: extra (test only)
overlay[matched] = [255, 255, 0] # Yellow: matched

plt.figure(figsize=(7, 7))
plt.imshow(overlay)
plt.title("Fuzzy Edge Match Overlay")
plt.axis('off')
plt.show()

# Plot only the not matched pixels (missed and extra) as a diff mask
diff_fuzzy = np.zeros_like(aligned_base_canny)
diff_fuzzy[missed | extra] = 1  # Use 1 for binary mask

diff_fuzzy = cv2.GaussianBlur(diff_fuzzy.astype(np.float32), (3, 3), 0)

diff_fuzzy_sub = cv2.dilate(diff_fuzzy.astype(np.uint8), kernel, iterations=1)
#diff_fuzzy_sub = cv2.medianBlur(diff_fuzzy_sub, 5)

plt.figure(figsize=(7, 7))
plt.imshow(diff_fuzzy, cmap='grey')
plt.title("Not Matched Pixel Density (Neighborhood Count)")
plt.axis('off')
plt.show()

plt.figure(figsize=(10, 10))

# Convert base and test images to color if they are grayscale
if len(base_gray.shape) == 2:
    base_color = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2RGB)
else:
    base_color = base_gray.copy()
if len(test_gray.shape) == 2:
    test_color = cv2.cvtColor(test_gray, cv2.COLOR_GRAY2RGB)
else:
    test_color = test_gray.copy()

# Overlay differences on the test image (red for missed, green for extra)
diff_overlay = test_color.copy()
diff_overlay = cv2.normalize(diff_overlay, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Highlight missed (red) and extra (green) pixels
missed_mask = (missed > 0)
extra_mask = (extra > 0)
diff_overlay[missed_mask] = [255, 0, 0]   # Red for missed
# If missed and extra overlap, show as yellow
diff_overlay[missed_mask & extra_mask] = [255, 255, 0]
diff_overlay[extra_mask & ~missed_mask] = [0, 255, 0]   # Green for extra only

plt.imshow(diff_overlay)
plt.title("Differences Highlighted on Test Image\nRed: Missed, Green: Extra, Yellow: Both")
plt.axis('off')
plt.show()