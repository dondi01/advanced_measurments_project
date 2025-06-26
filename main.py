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

project_root = Path(__file__).resolve().parent

def call_panorama_pipeline(folder_path):
    mat = scipy.io.loadmat(project_root / 'dataset_medi' / 'TARATURA' / 'medium_dataset_taratura.mat')
    camera_matrix = mat['K']
    dist_coeffs = mat['dist']
    image_files = sorted(
        glob.glob(folder_path),
        key=panorama.alphanum_key)
    frames = [cv2.imread(f, cv2.IMREAD_COLOR) for f in image_files]
    frames = [f for f in frames if f is not None and f.shape == frames[0].shape]
    orb = cv2.ORB_create(200)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    orb.detectAndCompute(frames[0], None)
    start = time.time()
    res = panorama.run_panorama_pipeline(frames, orb, bf, camera_matrix, dist_coeffs, show_plots=False)
    print("Execution time of panorama switching is:", time.time() - start)
    return res


if __name__ == "__main__":
    #folder_path = str(project_root / "dataset_piccoli" / "Scorre_verde" / "Lettere_disallineate" / "*.png")
    folder_path = str(project_root / "dataset_piccoli" / "Scorre_verde" / "Lettere_disallineate" / "*.png")
    torecompose = False  # Set to True if you want to recompute the panorama, False to use an existing image
    
    if torecompose:
        recomposed=call_panorama_pipeline(folder_path)  # Call the panorama pipeline function
        print("Panorama pipeline completed successfully.")
    else:
        recomposed=cv2.imread(str(project_root / "dataset_piccoli" / "dezoommata_green_cut.png"))

        #cv2.imread(str(project_root / "dataset_piccoli" / "dezoommata_green_cut.png"))

    base_shape=cv2.imread(str(project_root / 'Schematics' / 'shapes' /'green.png'),cv2.IMREAD_GRAYSCALE)  # Load the base image for comparison

    test_mask,base_shape_,_,_=tcm.compare_and_plot_masks(base_shape, recomposed,show_plots=False)  # Call the threshold comparison function
    print("Comparison of masks completed successfully")
    print("base_shape shape:", base_shape.shape)
    print("base_shape_ shape:", base_shape_.shape)
    
    base_print=cv2.imread(str(project_root / 'Schematics' / 'prints' /'green.png'),cv2.IMREAD_GRAYSCALE)  # Load the base print mask
    cpm.compare_prints_with_masks(base_print,recomposed, test_mask, show_plots=True)
    print("Comparison of prints with masks completed successfully.")


