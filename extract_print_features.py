import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
import Treshold_compare_masks as tcm
import paths
import functions_th as th
project_root = Path(__file__).resolve().parent

def fuzzy_diff_mask(base_mask, test_mask, show_plots=False):
    # Dilate each mask
    kernel = np.ones((21,21), np.uint8)
    dil_base = cv2.dilate(base_mask, kernel, iterations=1)
    dil_test = cv2.dilate(test_mask, kernel, iterations=1)
    # "Missed" = base edge not matched by test (even with band)
    missed = (base_mask > 0) & (dil_test == 0)
    # "Extra" = test edge not matched by base (even with band)
    extra = (test_mask > 0) & (dil_base == 0)
    # "Matched" = edge in base or test, and found in the other's band
    matched = ((base_mask > 0) & (dil_test > 0)) | ((test_mask > 0) & (dil_base > 0))
    return matched, missed, extra

def preprocess_for_canny(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    try:
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        gray = blurred

    # Standard threshold for contour finding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Canny mask as in Prove.py
    canny = cv2.Canny(blurred,0,60)
    contours= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    return canny, contours

#It compares the schematic image with the picture image,
#and returns a fuzzy edge diff mask. This returns a good estimation
#of prints, bends and scratches.
def extract_print(schematic_img, picture_img, show_plots=True):
    start = time.time()
    # Preprocess both images for Canny and contours
    aligned_base_canny, base_contours = preprocess_for_canny(schematic_img)
    aligned_test_canny, test_contours = preprocess_for_canny(picture_img)
    test_rect = cv2.minAreaRect(th.get_main_object_contour(test_contours, aligned_test_canny.shape))  # Ensure we have a rectangle for alignment
    base_rect = cv2.minAreaRect(th.get_main_object_contour(base_contours, aligned_base_canny.shape))  # Ensure we have a rectangle for alignment
    aligned_test_canny = th.rescale_and_resize_mask(aligned_test_canny, test_rect, base_rect, aligned_base_canny,pad_value=0)
    # Ensure the masks have the same size
    if aligned_base_canny.shape != aligned_test_canny.shape:
        aligned_test_canny=th.match_size(aligned_base_canny, aligned_test_canny,pad_value=0)
   
    matched,missed,extra=fuzzy_diff_mask(aligned_base_canny, aligned_test_canny, show_plots=show_plots)
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
    
    #print(f"Execution time of extract_print: {time.time() - start:.2f} seconds")
    return diff_fuzzy


if __name__ == "__main__":

    scorre_path, base_shape_path, base_print_path, recomposed_path = paths.define_files("green_ok", project_root)  # Paths to the base and test images

    base_img=cv2.imread(base_shape_path)  # Load the base image for comparison
    test_img=cv2.imread(recomposed_path)  # Load the test image
    res=extract_print(base_img, test_img, show_plots=True)
    cv2.imwrite(base_print_path, res * 255)  # Save the result as a binary mask