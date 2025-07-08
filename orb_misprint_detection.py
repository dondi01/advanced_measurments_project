import cv2
import matplotlib.pyplot as plt
import numpy as np
import paths
from pathlib import Path
import get_schematic as gs
import functions_th as th
import extract_print_features as epf
root = Path(__file__).resolve().parent


class detect_misprint:

    def __init__(self):
        # Use AKAZE for binary descriptors
        self.detector = cv2.AKAZE_create()
        # FLANN parameters for binary descriptors (LSH)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def detect_differences_with_orb(self, aligned_base, aligned_test, base_schematic,test_schematic,show_plots=True, min_kp_size=10):

        # Rescale and resize the aligned test image to match the aligned base image
        aligned_test = th.rescale_and_resize_mask(aligned_mask=aligned_test, target_img=aligned_base, pad_value=0)
        
        # Ensure the aligned test image has the same size as the aligned base image
        if aligned_test.shape != aligned_base.shape:
            aligned_test = th.match_size(aligned_base, aligned_test, pad_value=[0, 0, 0])
        
        # og_base = aligned_test.copy()
        # og_test = aligned_base.copy()
        
        aligned_base=epf.extract_print(aligned_base, base_schematic,show_plots=False)
        aligned_test=epf.extract_print(aligned_test,test_schematic, show_plots=False)
        
        og_base = aligned_test.copy()
        og_test = aligned_base.copy()
        # Detect keypoints and descriptors using AKAZE
        kp_base, des_base = self.detector.detectAndCompute(aligned_base, None)
        kp_test, des_test = self.detector.detectAndCompute(aligned_test, None)

        # Filter keypoints by minimum size
        if kp_base is not None and des_base is not None:
            kp_base_filtered = [kp for kp in kp_base if kp.size >= min_kp_size]
            des_base_filtered = des_base[[i for i, kp in enumerate(kp_base) if kp.size >= min_kp_size]]
        else:
            kp_base_filtered, des_base_filtered = [], None
        if kp_test is not None and des_test is not None:
            kp_test_filtered = [kp for kp in kp_test if kp.size >= min_kp_size]
            des_test_filtered = des_test[[i for i, kp in enumerate(kp_test) if kp.size >= min_kp_size]]
        else:
            kp_test_filtered, des_test_filtered = [], None

        # FLANN matcher requires descriptors to be of type np.uint8
        if des_base_filtered is not None and des_base_filtered.dtype != np.uint8:
            des_base_filtered = np.uint8(des_base_filtered)
        if des_test_filtered is not None and des_test_filtered.dtype != np.uint8:
            des_test_filtered = np.uint8(des_test_filtered)

        # Use KNN match with Lowe's ratio test for stricter matching
        if des_base_filtered is None or des_test_filtered is None or len(kp_base_filtered) == 0 or len(kp_test_filtered) == 0:
            return []

        knn_matches = self.matcher.knnMatch(des_base_filtered, des_test_filtered, k=2)
        ratio_thresh = 0.7  # Standard value for Lowe's ratio test
        good_matches = []
        for knn in knn_matches:
            if len(knn) == 2:
                m, n = knn
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
                

        # Filter matches by spatial distance
        max_spatial_distance = 30  # You can tune this value
        filtered_matches = []
        for match in good_matches:
            pt_base = kp_base_filtered[match.queryIdx].pt
            pt_test = kp_test_filtered[match.trainIdx].pt
            spatial_dist = np.linalg.norm(np.array(pt_base) - np.array(pt_test))
            if spatial_dist < max_spatial_distance:
                filtered_matches.append(match)

        # Sort filtered matches by descriptor distance
        before_filtered_matches = sorted(filtered_matches, key=lambda x: x.distance)

        # Highlight differences based on keypoints
        min_diff_detected = 10
        diff_img = aligned_test.copy()
        filtered_matches = []
        for match in before_filtered_matches:
            pt_base = kp_base_filtered[match.queryIdx].pt
            pt_test = kp_test_filtered[match.trainIdx].pt
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
            overlay = cv2.addWeighted(og_base, 0.5, og_test, 0.5, 0)
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            for match in filtered_matches:
                pt_base = kp_base_filtered[match.queryIdx].pt
                pt_test = kp_test_filtered[match.trainIdx].pt
                distance = np.linalg.norm(np.array(pt_base) - np.array(pt_test))
                if distance > min_diff_detected:
                    plt.plot([pt_base[0], pt_test[0]], [pt_base[1], pt_test[1]], color='red', linewidth=1)
                    plt.scatter([pt_base[0], pt_test[0]], [pt_base[1], pt_test[1]], color='yellow', s=10)
            plt.title("Overlay: Differences as Red Lines (Base to Test)")
            plt.axis("off")
            plt.show()

            # After detecting keypoints:
            img_with_keypoints_test = cv2.drawKeypoints(og_test, kp_test_filtered, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(img_with_keypoints_test, cv2.COLOR_BGR2RGB))
            plt.title("AKAZE Keypoints in Test Image (Filtered by Size)")
            plt.axis("off")
            plt.show()

            img_with_keypoints_base = cv2.drawKeypoints(og_base, kp_base_filtered, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(img_with_keypoints_base, cv2.COLOR_BGR2RGB))
            plt.title("AKAZE Keypoints in Base Image (Filtered by Size)")
            plt.axis("off")
            plt.show()

        # Extract and plot patches around keypoints in the test image
        # (plots are commented out, so no change needed)
        return filtered_matches

if __name__ == "__main__":
    # Example usage
    _,test_schematic_path,_,test=paths.define_files("green_lettere_disallineate", root)
    _,base_schematic_path,_,base=paths.define_files("green_ok", root)
    test= cv2.imread(test, cv2.IMREAD_COLOR)
    base= cv2.imread(base, cv2.IMREAD_COLOR)
    base_schematic= cv2.imread(base_schematic_path, cv2.IMREAD_COLOR)
    test_schematic= cv2.imread(test_schematic_path, cv2.IMREAD_COLOR)
    detect_misprint().detect_differences_with_orb(base, test, base_schematic, test_schematic)