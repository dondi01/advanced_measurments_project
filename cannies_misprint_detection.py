import numpy as np
import matplotlib.pyplot as plt
import cv2
import extract_print_features as epf
import functions_th as th
import paths
from pathlib import Path
import time

#@profile
def patch_based_misprint_detection(base, test, base_schematic, test_schematic, patch_size=11, edge_thresh=0, show_plots=False):
    """
    Compare two images using patch-based difference on their extracted print edges.
    Returns the coordinates of far_points (unmatched edge pixels in test).
    If show_plots is True, displays diagnostic plots.
    """
    test = th.rescale_and_resize_mask(aligned_mask=test, target_img=base, pad_value=0)
    if base.shape != test.shape:
        test = th.match_size(base, test, pad_value=[0, 0, 0])
    og_base = base.copy()
    og_test = test.copy()
    aligned_base = epf.extract_print(base, base_schematic, show_plots=False)
    aligned_test = epf.extract_print(test, test_schematic, show_plots=False)
    keypoints_test = np.column_stack(np.where(aligned_test > edge_thresh))
    half_patch = patch_size // 2
    # Only keep edge pixels that are not too close to the border
    def valid_points(points, img_shape, half_patch):
        return [pt for pt in points if
                half_patch <= pt[0] < img_shape[0] - half_patch and
                half_patch <= pt[1] < img_shape[1] - half_patch]
    keypoints_test = valid_points(keypoints_test, aligned_test.shape, half_patch)
    keypoints_test = np.array(keypoints_test)
    diff_threshold = patch_size**2/6 # Tune this value
    far_points = []
    for y, x in keypoints_test:
        patch_test = aligned_test[y-half_patch:y+half_patch+1, x-half_patch:x+half_patch+1]
        patch_base = aligned_base[y-half_patch:y+half_patch+1, x-half_patch:x+half_patch+1]
        sum_test = np.sum(patch_test)
        sum_base = np.sum(patch_base)
        if abs(sum_test - sum_base) > diff_threshold:
            far_points.append([y, x])
    far_points = np.array(far_points)
    if show_plots:
        plt.imshow(aligned_test, cmap='gray')
        if len(far_points) > 0:
            plt.scatter(far_points[:, 1], far_points[:, 0], s=1, c='yellow')
        plt.title('Misalligned pixels (yellow)')
        plt.axis('off')
        plt.tight_layout(pad=5)
        plt.show()
        # Overlay the two images
        # If both og_base and og_test are grayscale, convert them to color before blending

        overlay = cv2.addWeighted(og_base, 0.5, og_test, 0.5, 0)
        # Convert BGR to RGB for matplotlib
        #overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 8))
        plt.imshow(overlay)
        if len(far_points) > 0:
            plt.scatter(far_points[:, 1], far_points[:, 0], s=1, c='yellow')
        plt.title('Overlay with unmatched test edge pixels (yellow)')
        plt.axis('off')
        plt.show()
    return far_points

if __name__ == "__main__":
    # Example usage
    project_root = Path(__file__).parent
    scorre_path, base_shape_path, base_print_path, recomposed_path = paths.define_files("green_lettere_disallineate", project_root)
    scorre_path_ok, base_shape_path_ok, base_print_path_ok, recomposed_path_ok = paths.define_files("green_ok", project_root)
    base_img = cv2.imread(recomposed_path_ok)
    test_img = cv2.imread(recomposed_path)
    
    base_schematic = cv2.imread(base_print_path_ok, cv2.IMREAD_GRAYSCALE)
    _,test_schematic =th.preprocess(test_img)
    start=time.time()
    far_points = patch_based_misprint_detection(base_img, test_img, base_schematic, test_schematic, show_plots=True)
    end=time.time()
    print(f"Execution time: {end - start:.2f} seconds")
