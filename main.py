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

project_root = Path(__file__).resolve().parent

#To do all the operations before calling the panorama funciton
def call_panorama_pipeline(folder_path):
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
    res = panorama.run_panorama_pipeline(frames, orb, bf, camera_matrix, dist_coeffs, show_plots=False)
    print("Execution time of panorama switching is:", time.time() - start)
    return res




if __name__ == "__main__":
    #folder_path = str(project_root / "dataset_piccoli" / "Scorre_verde" / "Lettere_disallineate" / "*.png")
    
    #Set the folder path for the images to be processed
    folder_path = str(project_root / "dataset_medi" / "Scorre_Parmareggio_no" / "*.png")
    torecompose = False  # Set to True if you want to recompute the panorama, False to use an existing image
    
    if torecompose:

        recomposed=call_panorama_pipeline(folder_path)  # Call the panorama pipeline function
        print("Panorama pipeline completed successfully.") #Just a message to indicate the process is done, 
                                                           #Can be removed
    else:
        #If you do not want to recompute the panorama, load the existing image
        recomposed=cv2.imread(str(project_root / "dataset_piccoli" / "dezoommata_green.png"))
        #cv2.imread(str(project_root / "dataset_piccoli" / "dezoommata_green_cut.png"))

    #Load the ok schematic image for comparison
    base_shape=cv2.imread(str(project_root / 'Schematics' / 'shapes' /'green.png'),cv2.IMREAD_GRAYSCALE)  # Load the base image for comparison

    #Compute the difference mask between the recomposed panorama and the base shape
    #to find the bigger holes in the panorama
    test_mask,_,diff_mask,_=tcm.compare_and_plot_masks(base_shape, recomposed,show_plots=True)  # Call the threshold comparison function
    contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #Check if the contours found in the difference mask are significant 
    #enough to be considered defects
    min_defect_area = 50  # pixels, tune as needed
    found = False
    contours=tcm.get_main_object_contour(contours, diff_mask.shape, area_thresh=0.9)
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
    base_print=cv2.imread(str(project_root / 'Schematics' / 'prints' /'green.png'),cv2.IMREAD_GRAYSCALE)  # Load the base print mask
    diff_mask=cpm.compare_prints_with_masks(base_print,recomposed, test_mask, show_plots=True)

    #Skeletonize the difference mask to clear out some noise
    #and to prepare it for further analysis
    diff_mask=skeletonize(diff_mask)

    #Find connected components in the skeletonized mask
    labeled = label(diff_mask)

    #Measure length of each defect (region), to make sure you are only considering
    #significant defects
    min_length = 50  # Set your threshold
    filtered_skeleton = np.zeros_like(diff_mask, dtype=np.uint8)
    
    #For each defect found in the skeletonized mask, check if it is significant,
    #and if so, add it to the filtered skeleton mask
    for region in regionprops(labeled):
        if region.area >= min_length:
            # region.coords gives the (row, col) coordinates of the skeleton pixels
            for coord in region.coords:
                filtered_skeleton[coord[0], coord[1]] = 1
    #If there is no significant defect found in the skeletonized mask,
    #then the carton is ok
    if filtered_skeleton.sum() == 0:
        print("No significant defects found in the skeletonized mask.")
    else:
        print(f"Found {filtered_skeleton.sum()} significant defects in the skeletonized mask.")
    
    #Plot the skeletonized difference mask for debugging
    plt.imshow(filtered_skeleton, cmap='gray')
    plt.title("Skeletonized Difference Mask")
    plt.axis('off')
    plt.show()
    print("Comparison of prints with masks completed successfully.")


