import numpy as np
import matplotlib.pyplot as plt
import cv2
import extract_print_features as epf
import paths
from pathlib import Path
import functions_th as th
root= Path(__file__).resolve().parent
_,test_schematic_path,_,test=paths.define_files("green_lettere_disallineate", root)
_,base_schematic_path,_,base=paths.define_files("green_ok", root)
test= cv2.imread(test, cv2.IMREAD_COLOR)
base= cv2.imread(base, cv2.IMREAD_COLOR)
base_schematic= cv2.imread(base_schematic_path,cv2.IMREAD_GRAYSCALE) 
test_schematic= cv2.imread(test_schematic_path,cv2.IMREAD_GRAYSCALE)
test=th.rescale_and_resize_mask(aligned_mask=test, target_img=base, pad_value=0)
test=th.match_size(base, test, pad_value=[0, 0, 0])
og_base= base.copy()
og_test= test.copy()
aligned_base= epf.extract_print(base, base_schematic, show_plots=False)
aligned_test= epf.extract_print(test, test_schematic, show_plots=False)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(aligned_base, cmap='gray')
plt.title('Aligned Base Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(aligned_test, cmap='gray')
plt.title('Aligned Test Image')
plt.axis('off')
plt.show()

# After you have your edge images (aligned_base, aligned_test)
edge_thresh = 0 # or 1 if your edges are binary

# Get coordinates of edge pixels
keypoints_test = np.column_stack(np.where(aligned_test > edge_thresh))

patch_size = 11  # Must be odd
half_patch = patch_size // 2

# Only keep edge pixels that are not too close to the border
def valid_points(points, img_shape, half_patch):
    return [pt for pt in points if
            half_patch <= pt[0] < img_shape[0] - half_patch and
            half_patch <= pt[1] < img_shape[1] - half_patch]

keypoints_test = valid_points(keypoints_test, aligned_test.shape, half_patch)
keypoints_test = np.array(keypoints_test)

diff_threshold = patch_size**2/6 # Tune this value
count_diff_threshold = patch_size**2/5  # Tune this value
far_points = []
for y, x in keypoints_test:
    patch_test = aligned_test[y-half_patch:y+half_patch+1, x-half_patch:x+half_patch+1]
    patch_base = aligned_base[y-half_patch:y+half_patch+1, x-half_patch:x+half_patch+1]
    sum_test = np.sum(patch_test)
    sum_base = np.sum(patch_base)
    count_test = np.count_nonzero(patch_test > edge_thresh)
    count_base = np.count_nonzero(patch_base > edge_thresh)
    if abs(sum_test - sum_base) > diff_threshold and abs(count_test - count_base) > count_diff_threshold:
        far_points.append([y, x])

far_points = np.array(far_points)
plt.imshow(aligned_test, cmap='gray')
if len(far_points) > 0:
    plt.scatter(far_points[:, 1], far_points[:, 0], s=1, c='yellow')
plt.title('Test edge pixels with different patch sum from base')
plt.show()

# Overlay the two images
overlay = cv2.addWeighted(og_base, 0.5, og_test, 0.5, 0)
plt.figure(figsize=(8, 8))
plt.imshow(overlay, cmap='gray')
if len(far_points) > 0:
    plt.scatter(far_points[:, 1], far_points[:, 0], s=1, c='yellow')
plt.title('Overlay with unmatched test edge pixels (yellow)')
plt.axis('off')
plt.show()