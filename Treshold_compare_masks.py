import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
import paths
import functions_th as th

project_root = Path(__file__).resolve().parent

def compute_overlay(base_mask, test_mask):
     # Overlay the two masks to visualize differences
    overlay = np.zeros((base_mask.shape[0], base_mask.shape[1], 3), dtype=np.uint8)

    # Red for base mask, green for test mask, yellow for overlap
    overlay[(base_mask == 0) & (test_mask > 0)] = [255, 0, 0]  # Red: Base mask only
    overlay[(test_mask == 0) & (base_mask > 0)] = [0, 255, 0]  # Green: Test mask only
    overlay[(base_mask == test_mask)] = [255, 255, 0]  # Yellow: Overlap
    return overlay


def compare_and_plot_masks(base_img, test_img, show_plots=False):
    #The base one is already imported as aligned
    aligned_base_thresh=base_img

    _, aligned_test_thresh = th.preprocess(test_img)

    # Rescale and resize the test mask to match the base mask's rectangle and shape
    aligned_test_thresh = th.rescale_and_resize_mask(aligned_mask=aligned_test_thresh, target_img=aligned_base_thresh)
    
    # Ensure the masks have the same size, or absdiff will fail
    #It is needed sometimes, probably due to rounding
    if aligned_base_thresh.shape != aligned_test_thresh.shape:
        
        aligned_test_thresh=th.match_size(aligned_base_thresh, aligned_test_thresh,pad_value=255)

    # Compute absolute difference between the two masks
    diff_mask = cv2.absdiff(aligned_base_thresh, aligned_test_thresh)

    #clean up small noise
    kernel = np.ones((11,11), np.uint8)
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)

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

        overlay=compute_overlay(aligned_base_thresh, aligned_test_thresh)
        # Plot the overlay
        plt.figure(figsize=(7, 7))
        plt.imshow(overlay)
        plt.title("Overlay of Base and Test Masks")
        plt.axis('off')
        plt.show()

    return aligned_test_thresh, aligned_base_thresh, diff_mask

if __name__ == "__main__":
    scorre_path, base_shape_path, base_print_path, recomposed_path= paths.define_files("parmareggio", project_root)  # Paths to the base and test images

    base = cv2.imread(base_shape_path,cv2.IMREAD_GRAYSCALE)
    test = cv2.imread(recomposed_path)
    compare_and_plot_masks(base, test, show_plots=True)