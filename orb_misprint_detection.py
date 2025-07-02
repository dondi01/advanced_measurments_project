import cv2
import matplotlib.pyplot as plt
import numpy as np
import paths
from pathlib import Path
from Treshold_compare_masks import preprocess, align_image_to_angle, rescale_and_resize_mask, center_crop, center_pad
from Treshold_compare_masks import get_orientation_angle_and_rectangle, get_main_object_contour
import time
import get_schematic as gs
root = Path(__file__).resolve().parent

def detect_differences_with_orb(base, test, show_plots=True):
    """
    Detect differences between two images using ORB feature detection.

    Args:
        base_path (str): Path to the base image (reference).
        test_path (str): Path to the test image.
        show_plots (bool): Whether to display plots or not.

    Returns:
        None: Displays the differences.
    """
    # Load the images
    #base = cv2.imread(base_path, cv2.IMREAD_COLOR)
    #test = cv2.imread(test_path, cv2.IMREAD_COLOR)

    # Preprocess both images to extract contours
    _, base_contours, _ = preprocess(base)
    _, test_contours, _ = preprocess(test)
    # Align both images to angle 0 or 90 based on the longest side of their rectangles
    base_angle, _, base_rect = get_orientation_angle_and_rectangle(get_main_object_contour(base_contours, base.shape))
    test_angle, _, test_rect = get_orientation_angle_and_rectangle(get_main_object_contour(test_contours, test.shape))
    base_rect=cv2.minAreaRect(get_main_object_contour(base_contours,base.shape))  # Ensure we have a rectangle for alignment
    
    # Determine alignment angle (0 or 90 degrees)   
    aligned_base = gs.align_image_to_least_rotation(base, base_contours)[0]
    aligned_test = gs.align_image_to_least_rotation(test, test_contours)[0]
    

    # Rescale and resize the aligned test image to match the aligned base image
    aligned_test = rescale_and_resize_mask(aligned_test, test_rect, base_rect, aligned_base.shape[:2], pad_value=0)

    # Ensure the aligned test image has the same size as the aligned base image
    if aligned_test.shape != aligned_base.shape:
        aligned_test = center_crop(aligned_test, aligned_base.shape[:2])
        if aligned_test.shape != aligned_base.shape:
            aligned_test = center_pad(aligned_test, aligned_base.shape[:2])
    
    # Convert images to grayscale
    gray_base = cv2.equalizeHist(cv2.cvtColor(aligned_base, cv2.COLOR_BGR2GRAY))
    gray_test = cv2.equalizeHist(cv2.cvtColor(aligned_test, cv2.COLOR_BGR2GRAY))
    patch_size =31

    # Initialize ORB detector
    orb = cv2.ORB_create(600, 1.1, 3, int(patch_size), 0, 2, cv2.ORB_HARRIS_SCORE, patch_size)

    # Detect keypoints and descriptors
    kp_base, des_base = orb.detectAndCompute(gray_base, None)
    kp_test, des_test = orb.detectAndCompute(gray_test, None)

    # Match features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_base, des_test)

    # Filter matches by spatial distance
    max_spatial_distance = 10  # You can tune this value
    filtered_matches = []
    for match in matches:
        pt_base = kp_base[match.queryIdx].pt
        pt_test = kp_test[match.trainIdx].pt
        spatial_dist = np.linalg.norm(np.array(pt_base) - np.array(pt_test))
        if spatial_dist < max_spatial_distance:
            filtered_matches.append(match)

    # Sort filtered matches by descriptor distance
    before_filtered_matches = sorted(filtered_matches, key=lambda x: x.distance)

    # Highlight differences based on keypoints
    min_diff_detected=5
    diff_img = aligned_test.copy()
    filtered_matches = []
    for match in before_filtered_matches:
        pt_base = kp_base[match.queryIdx].pt
        pt_test = kp_test[match.trainIdx].pt
        distance = np.linalg.norm(np.array(pt_base) - np.array(pt_test))
        if distance > min_diff_detected:  # Threshold for difference
            cv2.circle(diff_img, (int(pt_test[0]), int(pt_test[1])), 5, (0, 0, 255), -1)
            filtered_matches.append(match)

    # Display the differences
    if show_plots:
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(diff_img, cv2.COLOR_BGR2RGB))
        plt.title("Highlighted Differences (on test image)")
        plt.axis("off")
        plt.show()

    # Overlay: plot both images and draw lines between matched keypoints with large distance
    if show_plots:
        overlay = cv2.addWeighted(aligned_base, 0.5, aligned_test, 0.5, 0)
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        for match in filtered_matches:
            pt_base = kp_base[match.queryIdx].pt
            pt_test = kp_test[match.trainIdx].pt
            distance = np.linalg.norm(np.array(pt_base) - np.array(pt_test))
            if distance > min_diff_detected:
                plt.plot([pt_base[0], pt_test[0]], [pt_base[1], pt_test[1]], color='red', linewidth=1)
                plt.scatter([pt_base[0], pt_test[0]], [pt_base[1], pt_test[1]], color='yellow', s=10)
        plt.title("Overlay: Differences as Red Lines (Base to Test)")
        plt.axis("off")
        plt.show()

    # After detecting keypoints:
    if show_plots:
        img_with_keypoints = cv2.drawKeypoints(aligned_test, kp_test, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
        plt.title("ORB Keypoints in Test Image")
        plt.axis("off")
        plt.show()

    # Extract and plot patches around keypoints in the test image
    # (plots are commented out, so no change needed)
    return filtered_matches, kp_base, kp_test, aligned_base, aligned_test

if __name__ == "__main__":
    # Example usage
    _,_,_,test=paths.define_files("parmareggio", root)
    _,_,_,base=paths.define_files("parmareggio_ok", root)
    test= cv2.imread(test, cv2.IMREAD_COLOR)
    base= cv2.imread(base, cv2.IMREAD_COLOR)
    detect_differences_with_orb(base, test)