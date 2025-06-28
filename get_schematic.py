import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
import paths
project_root = Path(__file__).resolve().parent

# Function to find a rectangle that approximates the main body, to find its orientation
# and center, useful to align the image to zero orientation
def get_orientation_angle_and_rectangle(contour):
    rect = cv2.minAreaRect(contour)
    (center_x, center_y), (width, height), angle = rect

    # Adjust the angle to align the main axis horizontally
    if width >= height:  # If the width is greater, align horizontally
        main_axis_angle = angle
    else:  # If the height is greater, rotate by 90 degrees to align horizontally
        main_axis_angle = angle + 90

    # Normalize to [-90, 90)
    main_axis_angle = main_axis_angle % 180
    if main_axis_angle >= 90:
        main_axis_angle -= 180

    center = ((center_x), (center_y))
    return main_axis_angle, center, rect

# Function to get the contour of the main body, it works by finding the longest one,
# excluding the ones that are too big (more than 90% of the pic), to avoid the background
def get_main_object_contour(contours, image_shape, area_thresh=0.99):
    img_area = image_shape[0] * image_shape[1]
    filtered = [c for c in contours if cv2.contourArea(c) < area_thresh * img_area]
    if not filtered:
        return None
    return max(filtered, key=cv2.contourArea)

# Takes the image path, reads the image, lightens it, converts it to grayscale,
# blurs it, and thresholds it to create a binary mask.
def preprocess(image_path):
    image = cv2.imread(image_path)
    lightened = cv2.convertScaleAbs(image, alpha=1, beta=100)
    gray = cv2.cvtColor(lightened, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return image, contours, thresh

def align_image_to_least_rotation(img, contours):
    """Align the image to the axis that involves the least rotation."""
    main_contour = get_main_object_contour(contours, img.shape)
    if main_contour is None or len(main_contour) < 5:
        return img, None, None

    # Get the orientation angle and bounding rectangle
    angle, rect_center, rect = get_orientation_angle_and_rectangle(main_contour)

    # Define possible axes to align to
    possible_axes = [0, 90, -90]

    # Find the axis that requires the least rotation
    best_axis = min(possible_axes, key=lambda axis: abs(angle - axis))
    rotation_angle = angle - best_axis

    # Compute the rotation matrix
    M_rot = cv2.getRotationMatrix2D(rect_center, rotation_angle, 1.0)
    (h, w) = img.shape[:2]
    rotated_img = cv2.warpAffine(img, M_rot, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

    # Translate the rectangle to the center of the image
    img_center = (w // 2, h // 2)
    dx = img_center[0] - int(rect_center[0])
    dy = img_center[1] - int(rect_center[1])
    M_trans = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned_img = cv2.warpAffine(rotated_img, M_trans, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

    return aligned_img, rect, main_contour

# Function to center crop the image to match the target shape
def center_crop(img, target_shape):
    h, w = img.shape[:2]
    th, tw = target_shape
    y1 = max((h - th) // 2, 0)
    x1 = max((w - tw) // 2, 0)
    y2 = y1 + th
    x2 = x1 + tw
    return img[y1:y2, x1:x2]

# Function to center pad the image to match the target shape
def center_pad(img, target_shape):
    pad_vert = (target_shape[0] - img.shape[0]) // 2
    pad_horz = (target_shape[1] - img.shape[1]) // 2
    img = cv2.copyMakeBorder(
        img,
        pad_vert, target_shape[0] - img.shape[0] - pad_vert,
        pad_horz, target_shape[1] - img.shape[1] - pad_horz,
        cv2.BORDER_CONSTANT, value=0
    )
    return img

# Start processing
start = time.time()

# Path to your image
_, base_shape_path, _, recomposed_path =paths.define_files("parmareggio_ok", project_root)  # Path to the base schematic image

# Preprocess the image
base_img, base_contours, base_thresh = preprocess(recomposed_path)

# Align the mask to zero orientation
aligned_base_thresh, base_rect, base_main_contour = align_image_to_least_rotation(base_thresh, base_contours)

# Plot the results
plt.figure(figsize=(10, 6))

# Original mask
plt.subplot(1, 2, 1)
plt.imshow(base_thresh, cmap='gray')
plt.title('Original Mask')
plt.axis('off')

# Aligned mask
plt.subplot(1, 2, 2)
plt.imshow(aligned_base_thresh, cmap='gray')
plt.title('Aligned Mask')
plt.axis('off')

plt.tight_layout()
plt.show()

cv2.imwrite(base_shape_path, aligned_base_thresh)
print(f"Execution time: {time.time() - start:.2f} seconds")