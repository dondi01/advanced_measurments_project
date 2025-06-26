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

def preprocess(image):
    """Preprocess the image to extract contours and thresholded mask."""
    lightened = cv2.convertScaleAbs(image, alpha=1, beta=100)
    try:
        gray = cv2.cvtColor(lightened, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        gray=lightened
    blurred = cv2.GaussianBlur(gray, (31, 31), 0)
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

def compare_and_plot_masks(base_img, test_img, show_plots=False):
    """Compare and plot masks from base and test images (images as arrays). The test mask is always adapted to the base mask, never the other way around."""
    start = time.time()

    aligned_base_thresh=base_img
    # Preprocess both images (now pass arrays, not paths)
    _, base_contours, _ = preprocess(base_img)
    test_img_proc, test_contours, test_thresh = preprocess(test_img)

    # Compute the angles of the main contours
    base_angle, _, _ = get_orientation_angle_and_rectangle(get_main_object_contour(base_contours, aligned_base_thresh.shape))
    test_angle, _, _ = get_orientation_angle_and_rectangle(get_main_object_contour(test_contours, test_thresh.shape))

    # Find the best alignment axis to minimize total rotation
    best_axis = find_best_alignment_angle(base_angle, test_angle)

    # Align only the test mask to the base mask's orientation and center
    aligned_test_thresh, test_rect, test_main_contour = align_image_to_angle(test_thresh, test_contours, best_axis)

    # Use main object rectangles for scaling (adapt test to base)
    base_rect = cv2.minAreaRect(get_main_object_contour(base_contours, aligned_base_thresh.shape))
    if test_rect is not None and base_rect is not None:
        base_w, base_h = sorted(base_rect[1])
        test_w, test_h = sorted(test_rect[1])
        ratio_w = base_w / test_w if test_w != 0 else 1
        ratio_h = base_h / test_h if test_h != 0 else 1
        new_w = int(aligned_test_thresh.shape[1] * ratio_w)
        new_h = int(aligned_test_thresh.shape[0] * ratio_h)
        aligned_test_thresh = cv2.resize(aligned_test_thresh, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
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
        target_shape = aligned_base_thresh.shape[:2]
        aligned_test_thresh = center_crop(aligned_test_thresh, target_shape)
        # If test mask is still smaller, pad it
        pad_vert = max(target_shape[0] - aligned_test_thresh.shape[0], 0)
        pad_horz = max(target_shape[1] - aligned_test_thresh.shape[1], 0)
        if pad_vert > 0 or pad_horz > 0:
            aligned_test_thresh = cv2.copyMakeBorder(
                aligned_test_thresh,
                pad_vert // 2, pad_vert - pad_vert // 2,
                pad_horz // 2, pad_horz - pad_horz // 2,
                cv2.BORDER_CONSTANT, value=0
            )
        # If test mask is too big, crop again
        if aligned_test_thresh.shape != target_shape:
            aligned_test_thresh = center_crop(aligned_test_thresh, target_shape)

    # Compute absolute difference between the two masks
    diff_mask = cv2.absdiff(aligned_base_thresh, aligned_test_thresh)

    # Optional: clean up small noise
    kernel = np.ones((21, 21), np.uint8)
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)

    print(f"Execution time of compare masks: {time.time() - start:.2f} seconds")
    if show_plots:
        # Plot the aligned base mask, aligned test mask, and difference mask
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(aligned_base_thresh, cmap='gray')
        plt.title("Aligned Base Mask")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(aligned_test_thresh, cmap='gray')
        plt.title("Aligned Test Mask (Adapted)")
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

    if show_plots:
        # Plot the overlay
        plt.figure(figsize=(7, 7))
        plt.imshow(overlay)
        plt.title("Overlay of Base and Test Masks")
        plt.axis('off')
        plt.show()

    return aligned_test_thresh, aligned_base_thresh, diff_mask, overlay

if __name__ == "__main__":
    # Paths to your images
    base_path = str(project_root / "Schematics" / "green.png")
    test_path = str(project_root / "Reconstructed" / "green_buco_in_piu.png")
    base = cv2.imread(base_path)
    test = cv2.imread(test_path)
    compare_and_plot_masks(base, test)