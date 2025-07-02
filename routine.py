import Panorama_Switching as panorama
import cv2
import numpy as np
from pathlib import Path    
import matplotlib.pyplot as plt
import scipy.io
import glob
import time
import Treshold_compare_masks as tcm
import compare_prints_with_masks as cpm
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
import paths
import tensorflow as tf
import random
import orb_misprint_detection as omd

project_root = Path(__file__).resolve().parent

#To do all the operations before calling the panorama funciton
def call_panorama_pipeline(folder_path,show_plots=False):
    #Load calibration data from the .mat file
    mat = scipy.io.loadmat(project_root / 'dataset_medi' / 'TARATURA' / 'medium_dataset_taratura.mat')
    camera_matrix = mat['K']
    dist_coeffs = mat['dist']
    #Load the images from the specified folder path
    #This sorts the images by their alphanumeric order
    #and reads them into a list of frames. glob.glob take *.png and finds 
    # all the pngs
    image_files = sorted(
        glob.glob(folder_path),
        key=panorama.alphanum_key)
    #Read the images into frames, and filter out any None or mismatched shape frames
    frames = [cv2.imread(f, cv2.IMREAD_COLOR) for f in image_files]
    frames = [f for f in frames if f is not None and f.shape == frames[0].shape]
    #initialize ORB and BFMatcher, the first execution always takes longer,
    #so we run it on the first frame to warm up
    orb = cv2.ORB_create(200)
    orb.detectAndCompute(frames[0], None)
    #Also initialize the BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #Call it and compute execution time
    start = time.time()
    res = panorama.run_panorama_pipeline(frames, orb, bf, camera_matrix, dist_coeffs, show_plots,save_path=None)
    #print("Execution time of panorama switching is:", time.time() - start)
    return res

@profile
def run_full_analysis(recomposed, base_shape, base_print, base_image,show_plots=True):
    """
    Run the full analysis pipeline.
    Returns True if a defect (scratch or hole) is detected, False otherwise.
    """
    # Load images
    #recomposed = cv2.imread(recomposed_path, cv2.IMREAD_COLOR)
    #base_shape = cv2.imread(base_shape_path, cv2.IMREAD_GRAYSCALE)
    #base_print = cv2.imread(base_print_path, cv2.IMREAD_GRAYSCALE)

    # Compare masks (holes)
    test_mask, _, holes, _ = tcm.compare_and_plot_masks(base_shape, recomposed, show_plots=False)
    
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
    scratches = cpm.compare_prints_with_masks(base_print, recomposed, test_mask, show_plots=False)
    scratches = skeletonize(scratches)
    labeled = label(scratches)
    min_length = 50
    filtered_skeleton = np.zeros_like(scratches, dtype=np.uint8)
    for region in regionprops(labeled):
        if region.area >= min_length:
            for coord in region.coords:
                filtered_skeleton[coord[0], coord[1]] = 1
    found_scratch = filtered_skeleton.sum() >= 200

    # ORB misprint detection
    filtered_matches, kp_base, kp_test, aligned_base, aligned_test = omd.detect_differences_with_orb(base_image, recomposed, show_plots=False)

    # Realign for plotting
    _, recomposed_contours, _ = tcm.preprocess(recomposed)
    _, base_contours, _ = tcm.preprocess(base_shape)
    base_angle, _, base_rect = tcm.get_orientation_angle_and_rectangle(tcm.get_main_object_contour(base_contours, base_shape.shape))
    recomposed_angle, recomposed_center, recomposed_rect = tcm.get_orientation_angle_and_rectangle(tcm.get_main_object_contour(recomposed_contours, recomposed.shape))
    aligned_img, recomposed_rect, _ = tcm.align_image_to_angle(recomposed, recomposed_contours, base_angle, (recomposed_angle, recomposed_center, recomposed_rect))
    aligned_img = tcm.rescale_and_resize_mask(aligned_img, recomposed_rect, base_rect, base_shape.shape[:2], pad_value=0)
    if aligned_img.shape != base_shape.shape:
        aligned_img = tcm.center_crop(aligned_img, base_shape.shape[:2])
        if aligned_img.shape != base_shape.shape:
            aligned_img = tcm.center_pad(aligned_img, base_shape.shape[:2], pad_value=0)

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
        for match in filtered_matches:
            pt_test = kp_test[match.trainIdx].pt
            plt.scatter(pt_test[0], pt_test[1], color='blue', s=10, label='ORB diff' if 'ORB diff' not in plt.gca().get_legend_handles_labels()[1] else "")
        
        plt.title("Aligned Image with Holes, Lines, and ORB Differences")
        plt.axis('off')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.show()
    
    found_disaligned = False
    if len(filtered_matches) > 3:
        found_disaligned = True

    return found_hole, found_scratch, found_disaligned

if __name__ == "__main__":
    # Example usage
    project_root = Path(__file__).resolve().parent
    scorre_path, base_shape_path, base_print_path, recomposed_path = paths.define_files("green_lettere_disallineate", project_root)
    base_shape= cv2.imread(base_shape_path, cv2.IMREAD_GRAYSCALE)
    test = cv2.imread(recomposed_path)
    base_print= cv2.imread(base_print_path,cv2.IMREAD_GRAYSCALE)
    base_image=cv2.imread(paths.define_files("green_ok",project_root)[3], cv2.IMREAD_COLOR)
    result = run_full_analysis(
        recomposed=test,
        base_shape=base_shape,
        base_print=base_print,
        show_plots=False,
        base_image=base_image
    )
    print(f"Defect detected: {result}")