import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_orientation_angle_and_rectangle(contour):
    rect = cv2.minAreaRect(contour)
    (center_x, center_y), (width, height), angle = rect
    # Normalize angle for symmetry: [0, 90)
    angle = angle % 180
    if angle < 0:
        angle += 180
    if angle >= 90:
        angle -= 90
    center = (int(center_x), int(center_y))
    return angle, center, rect

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
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return image, contours, thresh

def align_image_to_zero(img, contours, angle_threshold=5):
    main_contour = get_main_object_contour(contours, img.shape)
    if main_contour is None or len(main_contour) < 5:
        return img
    angle, rect_center, rect = get_orientation_angle_and_rectangle(main_contour)
    if abs(angle) < angle_threshold:
        rotated_img = img.copy()
    else:
        M_rot = cv2.getRotationMatrix2D(rect_center, -90+angle, 1.0)
        (h, w) = img.shape[:2]
        rotated_img = cv2.warpAffine(img, M_rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    (h, w) = img.shape[:2]
    img_center = (w // 2, h // 2)
    dx = img_center[0] - rect_center[0]
    dy = img_center[1] - rect_center[1]
    M_trans = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned_img = cv2.warpAffine(rotated_img, M_trans, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return aligned_img

# Paths to your images
base_path = "C:/Users/franc/Desktop/Scuola/Measurement/advanced_measurments_project/Reconstructed/green_OK.png"
test_path = "C:/Users/franc/Desktop/Scuola/Measurement/advanced_measurments_project/Reconstructed/green_buco_in_piu.png"

# Preprocess both images
base_img, base_contours, base_thresh = preprocess(base_path)
test_img, test_contours, test_thresh = preprocess(test_path)

# Align both images to zero orientation and center (only if needed)
aligned_base_img = align_image_to_zero(base_img, base_contours)
aligned_base_thresh = align_image_to_zero(base_thresh, base_contours)
aligned_test_img = align_image_to_zero(test_img, test_contours)
aligned_test_thresh = align_image_to_zero(test_thresh, test_contours)

# Recompute contours for aligned images
aligned_base_contours, _ = cv2.findContours(aligned_base_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
aligned_test_contours, _ = cv2.findContours(aligned_test_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Compare: extra in test (red)
result_img = aligned_test_img.copy()
for t_cnt in aligned_test_contours:
    match_found = False
    for b_cnt in aligned_base_contours:
        similarity = cv2.matchShapes(t_cnt, b_cnt, cv2.CONTOURS_MATCH_I1, 0.0)
        if similarity < 0.15:
            match_found = True
            break
    if not match_found and cv2.contourArea(t_cnt) > 50:
        cv2.drawContours(result_img, [t_cnt], -1, (0, 0, 255), 2)  # Red

# Compare: missing in test (blue)
missing_img = aligned_base_img.copy()
for b_cnt in aligned_base_contours:
    match_found = False
    for t_cnt in aligned_test_contours:
        similarity = cv2.matchShapes(b_cnt, t_cnt, cv2.CONTOURS_MATCH_I1, 0.0)
        if similarity < 0.1:
            match_found = True
            break
    if not match_found and cv2.contourArea(b_cnt) > 50:
        cv2.drawContours(missing_img, [b_cnt], -1, (255, 0, 0), 2)  # Blue

# Plot all contours for both images
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
cv2.drawContours(aligned_base_img, aligned_base_contours, -1, (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(aligned_base_img, cv2.COLOR_BGR2RGB))
plt.title('Base Image (Aligned) - All Contours')
plt.axis('off')

plt.subplot(1, 2, 2)
cv2.drawContours(aligned_test_img, aligned_test_contours, -1, (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(aligned_test_img, cv2.COLOR_BGR2RGB))
plt.title('Test Image (Aligned) - All Contours')
plt.axis('off')
plt.show()

# Show the result image with extra contours in test (red)
plt.figure(figsize=(7, 6))
plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.title('Extra Contours in Test (Red)')
plt.axis('off')
plt.show()

# Show the missing contours in test (blue)
plt.figure(figsize=(7, 6))
plt.imshow(cv2.cvtColor(missing_img, cv2.COLOR_BGR2RGB))
plt.title('Missing Contours in Test (Blue)')
plt.axis('off')
plt.show()
