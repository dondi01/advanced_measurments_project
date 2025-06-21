import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

def get_orientation_angle_and_rectangle(contour):
    rect = cv2.minAreaRect(contour)
    (center_x, center_y), (width, height), angle = rect

    if width < height:
        main_axis_angle = angle
    else:
        main_axis_angle = angle + 90

    # Normalize to [-90, 90)
    main_axis_angle = main_axis_angle % 180
    if main_axis_angle >= 90:
        main_axis_angle -= 180
    
    print(main_axis_angle)
    center = (int(center_x), int(center_y))
    return main_axis_angle, center, rect

def get_main_object_contour(contours, image_shape, area_thresh=0.9):
    img_area = image_shape[0] * image_shape[1]
    filtered = [c for c in contours if cv2.contourArea(c) < area_thresh * img_area]
    if not filtered:
        return None
    return max(filtered, key=cv2.contourArea)

def preprocess(image_path):
    image = cv2.imread(image_path)
    lightened = cv2.convertScaleAbs(image, alpha=1, beta=100)
    gray = cv2.cvtColor(lightened, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21,21), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return image, contours, thresh

def align_image_to_zero(img, contours):
    main_contour = get_main_object_contour(contours, img.shape)
    if main_contour is None or len(main_contour) < 5:
        return img, None, None
    angle, rect_center, rect = get_orientation_angle_and_rectangle(main_contour)
    M_rot = cv2.getRotationMatrix2D(rect_center, angle, 1.0)
    (h, w) = img.shape[:2]
    rotated_img = cv2.warpAffine(img, M_rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    (h, w) = img.shape[:2]
    img_center = (w // 2, h // 2)
    dx = img_center[0] - rect_center[0]
    dy = img_center[1] - rect_center[1]
    M_trans = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned_img = cv2.warpAffine(rotated_img, M_trans, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return aligned_img, rect, main_contour

def center_crop(img, target_shape):
    h, w = img.shape[:2]
    th, tw = target_shape
    y1 = max((h - th) // 2, 0)
    x1 = max((w - tw) // 2, 0)
    y2 = y1 + th
    x2 = x1 + tw
    return img[y1:y2, x1:x2]

start=time.time()
# Paths to your images
base_path = "C:/Users/franc/Desktop/Scuola/Measurement/advanced_measurments_project/Reconstructed/green_OK.png"
test_path = "C:/Users/franc/Desktop/Scuola/Measurement/advanced_measurments_project/Reconstructed/green_buco_in_piu.png"

# Preprocess both images
base_img, base_contours, base_thresh = preprocess(base_path)
test_img, test_contours, test_thresh = preprocess(test_path)

# Align both masks to zero orientation and center
aligned_base_thresh, base_rect, base_main_contour = align_image_to_zero(base_thresh, base_contours)
aligned_test_thresh, test_rect, test_main_contour = align_image_to_zero(test_thresh, test_contours)

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
kernel = np.ones((3, 3), np.uint8)
diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)
kernel = np.ones((5, 5), np.uint8)
diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)

contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_area = 200  # Adjust as needed
clean_mask = np.zeros_like(diff_mask)
for cnt in contours:
    if cv2.contourArea(cnt) > min_area:
        cv2.drawContours(clean_mask, [cnt], -1, 255, -1)
diff_mask = clean_mask
print(f"Execution time: {time.time() - start:.2f} seconds")

# Plot the difference mask
plt.figure(figsize=(7, 6))
plt.imshow(diff_mask, cmap='gray')
plt.title('Difference Mask (White = Difference)')
plt.axis('off')
plt.show()

# Optionally, overlay the difference on the test image (red)
# Align the color test image
aligned_test_img, _, _ = align_image_to_zero(test_img, test_contours)

# Resize the color image using the same scaling as the mask
if base_rect is not None and test_rect is not None:
    aligned_test_img = cv2.resize(aligned_test_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # Center crop or pad to match base mask size
    if aligned_test_img.shape[0] > target_shape[0] or aligned_test_img.shape[1] > target_shape[1]:
        aligned_test_img = center_crop(aligned_test_img, target_shape)
    elif aligned_test_img.shape[0] < target_shape[0] or aligned_test_img.shape[1] < target_shape[1]:
        pad_vert = (target_shape[0] - aligned_test_img.shape[0]) // 2
        pad_horz = (target_shape[1] - aligned_test_img.shape[1]) // 2
        aligned_test_img = cv2.copyMakeBorder(
            aligned_test_img,
            pad_vert, target_shape[0] - aligned_test_img.shape[0] - pad_vert,
            pad_horz, target_shape[1] - aligned_test_img.shape[1] - pad_horz,
            cv2.BORDER_CONSTANT, value=0
        )




highlight = aligned_test_img.copy()
if highlight.shape[:2] != diff_mask.shape[:2]:
    highlight = cv2.resize(highlight, (diff_mask.shape[1], diff_mask.shape[0]), interpolation=cv2.INTER_LINEAR)
highlight[diff_mask > 0] = [0, 0, 255]

plt.figure(figsize=(7, 6))
plt.imshow(cv2.cvtColor(highlight, cv2.COLOR_BGR2RGB))
plt.title('Differences Highlighted on Test Image')
plt.axis('off')
plt.show()