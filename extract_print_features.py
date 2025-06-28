import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
import Treshold_compare_masks as tcm
import paths
project_root = Path(__file__).resolve().parent

def preprocess_for_canny(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    try:
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        gray = blurred

    # Standard threshold for contour finding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Canny mask as in Prove.py
    canny = cv2.Canny(blurred,0,50, apertureSize=3)
    
    return canny, contours, gray

#It compares the schematic image with the picture image,
#and returns a fuzzy edge diff mask. This returns a good estimation
#of prints, bends and scratches.
def extract_print(schematic_img, picture_img, show_plots=True):
    start = time.time()
    # Preprocess both images for Canny and contours
    base_canny, base_contours, base_gray = preprocess_for_canny(schematic_img)
    test_canny, test_contours, test_gray = preprocess_for_canny(picture_img)
    # Compute the angles of the main contours
    base_angle, _, base_rect = tcm.get_orientation_angle_and_rectangle(tcm.get_main_object_contour(base_contours, base_gray.shape))
    test_angle, _, test_rect = tcm.get_orientation_angle_and_rectangle(tcm.get_main_object_contour(test_contours, test_gray.shape))
    # Find the best alignment axis to minimize total rotation
    best_axis = base_angle

    # Align both Canny masks to the best axis
    aligned_base_canny, base_rect, base_main_contour = tcm.align_image_to_angle(base_canny, base_contours, best_axis)
    aligned_test_canny, test_rect, test_main_contour = tcm.align_image_to_angle(test_canny, test_contours, best_axis)
    # Use tcm's robust scaling and resizing logic
    target_shape = aligned_base_canny.shape[:2]
    aligned_test_canny = tcm.rescale_and_resize_mask(aligned_test_canny, test_rect, base_rect, target_shape,pad_value=0)
    # Ensure the masks have the same size
    if aligned_base_canny.shape != aligned_test_canny.shape:
        min_h = min(aligned_base_canny.shape[0], aligned_test_canny.shape[0])
        min_w = min(aligned_base_canny.shape[1], aligned_test_canny.shape[1])
        aligned_base_canny = tcm.center_crop(aligned_base_canny, (min_h, min_w))
        aligned_test_canny = tcm.center_crop(aligned_test_canny, (min_h, min_w))
    # Dilate each mask
    kernel = np.ones((21,21), np.uint8)
    dil_base = cv2.dilate(aligned_base_canny, kernel, iterations=1)
    dil_test = cv2.dilate(aligned_test_canny, kernel, iterations=1)
    # "Missed" = base edge not matched by test (even with band)
    missed = (aligned_base_canny > 0) & (dil_test == 0)
    # "Extra" = test edge not matched by base (even with band)
    extra = (aligned_test_canny > 0) & (dil_base == 0)
    # "Matched" = edge in base or test, and found in the other's band
    matched = ((aligned_base_canny > 0) & (dil_test > 0)) | ((aligned_test_canny > 0) & (dil_base > 0))
    if show_plots:
        overlay = np.zeros((aligned_base_canny.shape[0], aligned_base_canny.shape[1], 3), dtype=np.uint8)
        overlay[missed] = [255, 0, 0]
        overlay[extra] = [0, 255, 0]
        overlay[matched] = [255, 255, 0]
        plt.figure(figsize=(7, 7))
        plt.imshow(overlay)
        plt.title("Fuzzy Edge Match Overlay")
        plt.axis('off')
        plt.show()
    # Plot only the not matched pixels (missed and extra) as a diff mask
    diff_fuzzy = np.zeros_like(aligned_base_canny)
    diff_fuzzy[missed | extra] = 1
    diff_fuzzy = cv2.GaussianBlur(diff_fuzzy.astype(np.float32), (3, 3), 0)
    if show_plots:
        plt.figure(figsize=(7, 7))
        plt.imshow(diff_fuzzy, cmap='grey')
        plt.title("Not Matched Pixel Density (Neighborhood Count)")
        plt.axis('off')
        plt.show()
    
    print(f"Execution time of extract_print: {time.time() - start:.2f} seconds")
    return diff_fuzzy


if __name__ == "__main__":

    scorre_path, base_shape_path, base_print_path, recomposed_path = paths.define_files("green_ok", project_root)  # Paths to the base and test images

    base_img=cv2.imread(base_shape_path)  # Load the base image for comparison
    test_img=cv2.imread(recomposed_path)  # Load the test image
    res=extract_print(base_img, test_img, show_plots=True)
    #cv2.imwrite(base_print_path, res * 255)  # Save the result as a binary mask