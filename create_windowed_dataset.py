import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
import sys
sys.path.append('./functions')
import ml_functions as mlfn
project_root = Path(__file__).resolve().parent

faulty_source_path = str( project_root / 'ml_datasets' / 'carton_baseline' / 'faulty')
healthy_source_path = str( project_root / 'ml_datasets' / 'carton_baseline' / 'healthy')
faulty_output_path = str( project_root / 'ml_datasets' / 'carton_windowed' / 'faulty')
healthy_output_path = str( project_root / 'ml_datasets' / 'carton_windowed' / 'healthy')

window_size = (512, 512)  # Size of the sliding window
stride = 250  # Step size for the sliding window

# rename files in the output directories to ensure consistency (a continous index is employed for non ambiguous naming)
image_index = 0
image_index = mlfn.rename_files_in_dataset(faulty_source_path, 'faulty', image_index)
image_index = mlfn.rename_files_in_dataset(healthy_source_path, 'healthy', image_index)


# Generate windowed datasets for both faulty and healthy images
mlfn.generate_windowed_dataset(faulty_source_path, window_size, stride, faulty_output_path)
mlfn.generate_windowed_dataset(healthy_source_path, window_size, stride, healthy_output_path)


#DEBUG OPTIONS

#test_img = cv2.imread(str(project_root / 'ml_datasets' / 'carton_baseline'/ 'faulty' / '2022-06-22_21_36_28_051.bmp'))
#test_img = preprocess_image(test_img, window_size, stride)
#generate_windows(0, test_img, window_size, stride, faulty_output_path)

