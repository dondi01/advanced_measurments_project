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
    print("Execution time of panorama switching is:", time.time() - start)
    return res


if __name__ == "__main__":
    
    #Set the folder path for the images to be processed
    scorre_path, base_shape_path, base_print_path, recomposed_path = paths.define_files("nappies_misprint", project_root)
    torecompose = False  # Set to True if you want to recompute the panorama, False to use an existing image
    
    if torecompose:

        recomposed=call_panorama_pipeline(scorre_path,show_plots=True)  # Call the panorama pipeline function
        print("Panorama pipeline completed successfully.") #Just a message to indicate the process is done, 
                                                           #Can be removed
    else:
        #If you do not want to recompute the panorama, load the existing image
        recomposed=cv2.imread(recomposed_path, cv2.IMREAD_COLOR)  # Load the recomposed panorama image
    #Load the ok schematic image for comparison
    base_shape=cv2.imread(base_shape_path,cv2.IMREAD_GRAYSCALE)  # Load the base image for comparison

    #Compute the difference mask between the recomposed panorama and the base shape
    #to find the bigger holes in the panorama
    test_mask,_,holes,_=tcm.compare_and_plot_masks(base_shape, recomposed,show_plots=True)  # Call the threshold comparison function

    contours, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Check if the contours found in the difference mask are significant 
    #enough to be considered defects
    min_defect_area = 50  # pixels, tune as needed
    found = False
    contours=[con for con in contours if cv2.contourArea(con) < 0.9 * recomposed.size]  # Filter contours based on area threshold
    if contours is not None:
        for cnt in contours:
            if cv2.contourArea(cnt) > min_defect_area:
                found = True
                break
    if found:
        print("Defect detected!")
    else:
        print("No defect detected.")
        print("Comparison of masks completed successfully")
    
    #Take the prints schematic and compare it with the recomposed panorama
    #to find smaller defects
    base_print=cv2.imread(base_print_path,cv2.IMREAD_GRAYSCALE)  # Load the base print mask
    scratches=cpm.compare_prints_with_masks(base_print,recomposed, test_mask, show_plots=True)

    #Skeletonize the difference mask to clear out some noise
    #and to prepare it for further analysis
    scratches=skeletonize(scratches)

    #Find connected components in the skeletonized mask
    labeled = label(scratches)

    #Measure length of each defect (region), to make sure you are only considering
    #significant defects
    min_length = 50  # Set your threshold
    filtered_skeleton = np.zeros_like(scratches, dtype=np.uint8)
    
    #For each defect found in the skeletonized mask, check if it is significant,
    #and if so, add it to the filtered skeleton mask
    for region in regionprops(labeled):
        if region.area >= min_length:
            # region.coords gives the (row, col) coordinates of the skeleton pixels
            for coord in region.coords:
                filtered_skeleton[coord[0], coord[1]] = 1
    #If there is no significant defect found in the skeletonized mask,
    #then the carton is ok
    if filtered_skeleton.sum() < 500:
        print("No significant defects found in the skeletonized mask.")
    else:
        print(f"Found {filtered_skeleton.sum()} significant defects in the skeletonized mask.")
    
    #Plot the skeletonized difference mask for debugging
    plt.imshow(filtered_skeleton, cmap='gray')
    plt.title("Skeletonized Difference Mask")
    plt.axis('off')
    plt.show()
    print("Comparison of prints with masks completed successfully.")

     
 # --- Realign the initial image using the same procedure as in Treshold_compare_masks.py ---Add commentMore actions
    # Preprocess the recomposed image to get contours
    _, recomposed_contours, _ = tcm.preprocess(recomposed)
    # Preprocess the base_shape to get contours and angle
    _, base_contours, _ = tcm.preprocess(base_shape)
    base_angle, _, base_rect = tcm.get_orientation_angle_and_rectangle(tcm.get_main_object_contour(base_contours, base_shape.shape))
    recomposed_angle, recomposed_center, recomposed_rect = tcm.get_orientation_angle_and_rectangle(tcm.get_main_object_contour(recomposed_contours, recomposed.shape))
    # Align recomposed image to base_shape orientation and center
    aligned_img, recomposed_rect, _ = tcm.align_image_to_angle(recomposed, recomposed_contours, base_angle, (recomposed_angle, recomposed_center, recomposed_rect))
    # Rescale and resize to match base_shape's rectangle and shape
    aligned_img = tcm.rescale_and_resize_mask(aligned_img, recomposed_rect, base_rect, base_shape.shape[:2],pad_value=0)
    # Crop or pad if needed
    if aligned_img.shape != base_shape.shape:
        aligned_img = tcm.center_crop(aligned_img, base_shape.shape[:2])
        if aligned_img.shape != base_shape.shape:
            aligned_img = tcm.center_pad(aligned_img, base_shape.shape[:2], pad_value=0)

    # --- Plot holes and lines on the aligned image ---
    # Find holes (contours) that surpass min_defect_area
    hole_contours, _ = cv2.findContours(holes.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    holes_to_plot = []
    if hole_contours is not None:
        for cnt in hole_contours:
            if cv2.contourArea(cnt) > min_defect_area:
                holes_to_plot.append(cnt)

    # Find lines (defects) that surpass min_length
    lines_to_plot = []
    for region in regionprops(label(filtered_skeleton)):
        if region.area >= min_length:
            for coord in region.coords:
                lines_to_plot.append((coord[1], coord[0]))  # (x, y)

    # Plot the aligned image with holes and lines
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
    # Plot holes
    for cnt in holes_to_plot:
        cnt = cnt.reshape(-1, 2)
        plt.plot(cnt[:, 0], cnt[:, 1], color='red', linewidth=2, label='Hole' if 'Hole' not in plt.gca().get_legend_handles_labels()[1] else "")
    # Plot lines
    if lines_to_plot:
        xs, ys = zip(*lines_to_plot)
        plt.scatter(xs, ys, color='yellow', s=1, label='Line')
    plt.title("Aligned Image with Detected Holes and Lines")
    plt.axis('off')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()
    print("Final visualization with holes and lines completed.")