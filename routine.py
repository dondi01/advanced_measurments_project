import Panorama_Switching as panorama
import cv2
import numpy as np
from pathlib import Path    
import matplotlib.pyplot as plt
import scipy.io
import glob
import Treshold_compare_masks as tcm
import compare_prints_with_masks as cpm
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
import paths
import cannies_misprint_detection as cmd
import functions_th as th
from sklearn.cluster import DBSCAN
import time

class Routine:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent
        self.panorama = panorama.reassembler()
        self.tcm = tcm  # If Treshold_compare_masks is now a class, use: tcm.TresholdCompareMasks()
        self.cpm = cpm  # If compare_prints_with_masks is now a class, use: cpm.ComparePrintsWithMasks()
        self.cmd = cmd  # Instantiate the class here
#    @profile
    def run_full_analysis(self, frames, base_shape, base_print, base_image, show_plots=False,recomposed=None):
        if recomposed is None:
            recomposed = self.panorama.run_panorama_pipeline(frames, show_plots=False, save_path=None)
        
        # Compare masks (holes) 
        test_mask, _, holes = self.tcm.compare_and_plot_masks(base_shape, recomposed, show_plots=False)
        
        contours, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_defect_area = 50
        found_hole = False
        contours = [con for con in contours if cv2.contourArea(con) < 0.9 * recomposed.size]
        if contours is not None:
            for cnt in contours:
                if cv2.contourArea(cnt) > min_defect_area:
                    found_hole = True
                    break

        # Compare prints (scratches)
        scratches = self.cpm.compare_prints_with_masks(base_print, recomposed, test_mask, show_plots=False)
        scratches = skeletonize(scratches)
        labeled = label(scratches)
        min_length = 50
        filtered_skeleton = np.zeros_like(scratches, dtype=np.uint8)
        for region in regionprops(labeled):
            if region.area >= min_length:
                for coord in region.coords:
                    filtered_skeleton[coord[0], coord[1]] = 1

        found_scratch = filtered_skeleton.sum() >= 200

        # Cannies misprint detection using the class instance
        aligned_img = th.rescale_and_resize_mask(aligned_mask=recomposed, target_img=base_shape, pad_value=0)
        aligned_base_img = th.rescale_and_resize_mask(aligned_mask=base_image, target_img=base_shape, pad_value=0)
        far_points = self.cmd.patch_based_misprint_detection(aligned_base_img, aligned_img, base_shape, test_mask, show_plots=False)

        # --- Cluster analysis for misprint detection ---
        found_disaligned = False
        if len(far_points) > 0:
            # DBSCAN clustering
            clustering = DBSCAN(eps=5, min_samples=3).fit(far_points)
            labels = clustering.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters > 0:
                cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]
                avg_cluster_size = np.mean(cluster_sizes)
                # Threshold: average cluster size must be above 10 pixels (tune as needed)
                if avg_cluster_size > 10:
                    found_disaligned = True
        # --- End cluster analysis ---

        # Plot if requested
        if show_plots:
            # Holes
            hole_contours, _ = cv2.findContours(holes.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            holes_to_plot = []
            if hole_contours is not None:
                for cnt in hole_contours:
                    if cv2.contourArea(cnt) > min_defect_area:
                        holes_to_plot.append(cnt)
            # Lines
            lines_to_plot = []
            for region in regionprops(label(filtered_skeleton)):
                if region.area >= min_length:
                    for coord in region.coords:
                        lines_to_plot.append((coord[1], coord[0]))
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
            for cnt in holes_to_plot:
                cnt = cnt.reshape(-1, 2)
                plt.plot(cnt[:, 0], cnt[:, 1], color='red', linewidth=2, label='Hole' if 'Hole' not in plt.gca().get_legend_handles_labels()[1] else "")
            if lines_to_plot:
                xs, ys = zip(*lines_to_plot)
                plt.scatter(xs, ys, color='yellow', s=1, label='Line')
            if len(far_points) > 0:
                plt.scatter(far_points[:, 1], far_points[:, 0], color='blue', s=2, label='Cannie diff')
            
            plt.title("Aligned Image with Holes, Lines, and Cannie Differences")
            plt.axis('off')
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            plt.show()
        
        return found_hole, found_scratch, found_disaligned

if __name__ == "__main__":
    routine = Routine()
    scorre_path, base_shape_path, base_print_path, recomposed_path = paths.define_files("green_buco_in_piu", routine.project_root)
    base_image_path = paths.define_files("green_ok", routine.project_root)[3]
    image_files = sorted(
        glob.glob(scorre_path),
        key=th.alphanum_key)
    frames = [cv2.imread(f, cv2.IMREAD_COLOR) for f in image_files]
    frames = [f for f in frames if f is not None and f.shape == frames[0].shape]
    base_shape = cv2.imread(base_shape_path, cv2.IMREAD_GRAYSCALE)
    base_print = cv2.imread(base_print_path, cv2.IMREAD_GRAYSCALE)
    base_image=cv2.imread(base_image_path,cv2.IMREAD_COLOR)
    start=time.time()
    result = routine.run_full_analysis(
        frames=frames,
        base_shape=base_shape,
        base_print=base_print,
        base_image=base_image,
        show_plots=True,
        recomposed=cv2.imread(recomposed_path, cv2.IMREAD_COLOR)
    )
    end=time.time()
    print(f"Execution time: {end - start:.2f} seconds")
    print(f"Defect detected: {result}")