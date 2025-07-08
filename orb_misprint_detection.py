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
        self.detector = cv2.AKAZE_create(threshold=1e-9, nOctaves=4, nOctaveLayers=4, diffusivity=cv2.KAZE_DIFF_PM_G2)
        # FLANN parameters for binary descriptors (LSH)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def detect_differences_with_orb(self, aligned_base, aligned_test, base_schematic,test_schematic,show_plots=True):

        # Rescale and resize the aligned test image to match the aligned base image
        aligned_test = th.rescale_and_resize_mask(aligned_mask=aligned_test, target_img=aligned_base, pad_value=0)
        og_base = aligned_base.copy()
        og_test = aligned_test.copy()

        # Ensure the aligned test image has the same size as the aligned base image
        if aligned_test.shape != aligned_base.shape:
            aligned_test = th.match_size(aligned_base, aligned_test, pad_value=[0, 0, 0])
        
        #aligned_base=epf.extract_print(aligned_base, base_schematic,show_plots=False)
        #aligned_test=epf.extract_print(aligned_test,test_schematic, show_plots=False)

        # Detect keypoints and descriptors using AKAZE
        kp_base, des_base = self.detector.detectAndCompute(aligned_base.astype(np.uint8), None)
        kp_test, des_test = self.detector.detectAndCompute(aligned_test.astype(np.uint8), None)

        good_matches=self.matcher.match(des_base, des_test)

        # Filter matches by spatial distance
        max_spatial_distance = 1000  # You can tune this value
        filtered_matches = []
        for match in good_matches:
            pt_base = kp_base[match.queryIdx].pt
            pt_test = kp_test[match.trainIdx].pt
            spatial_dist = np.linalg.norm(np.array(pt_base) - np.array(pt_test))
            if spatial_dist < max_spatial_distance:
                filtered_matches.append(match)

        # Sort filtered matches by descriptor distance
        before_filtered_matches = sorted(filtered_matches, key=lambda x: x.distance)[:10]

        # Highlight differences based on keypoints
        min_diff_detected = 0
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
            overlay = cv2.addWeighted(og_base, 0.5, og_test, 0.5, 0)
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
            img_with_keypoints_test = cv2.drawKeypoints(og_test, kp_test, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(img_with_keypoints_test, cv2.COLOR_BGR2RGB))
            plt.title("AKAZE Keypoints in Test Image (Filtered by Size)")
            plt.axis("off")

            img_with_keypoints_base = cv2.drawKeypoints(og_base, kp_base, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            plt.subplot(1, 2, 2)
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
    base_schematic= cv2.imread(base_schematic_path,cv2.IMREAD_GRAYSCALE) 
    test_schematic= cv2.imread(test_schematic_path,cv2.IMREAD_GRAYSCALE)
    
    detect_misprint().detect_differences_with_orb(base, test, base_schematic, test_schematic)