import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
import sys
sys.path.append('./functions')
import ml_functions as mlfn
project_root = Path(__file__).resolve().parent


#dataset1_faulty_path = str( project_root / 'ml_datasets' / 'archive_2' / 'faulty')
#dataset1_healthy_path = str( project_root / 'ml_datasets' / 'archive_2' / 'healthy')
#dataset2_faulty_path = str( project_root / 'ml_datasets' / 'carton_baseline' / 'faulty')
#dataset2_healthy_path = str( project_root / 'ml_datasets' / 'carton_baseline' / 'healthy')
dataset3_faulty_path = str( project_root / 'ml_datasets' / 'our_dataset' / 'faulty')
dataset3_healthy_path = str( project_root / 'ml_datasets' / 'our_dataset' / 'healthy')
input_healthy_path = str( project_root / 'ml_datasets' / 'multiple_datasets' / 'input_folder' / 'healthy')
input_faulty_path = str( project_root / 'ml_datasets' / 'multiple_datasets' / 'input_folder' / 'faulty')

window_healthy_path = str( project_root / 'ml_datasets' / 'multiple_datasets' / 'working_folder' / 'healthy')
window_faulty_path = str( project_root / 'ml_datasets' / 'multiple_datasets' / 'working_folder' / 'faulty')


window_size = (224, 224)  # Size of the sliding window
stride = 56  # Step size for the sliding window
border_type = 'reflect'  

# Empties out folders
mlfn.empty_directory(input_healthy_path)
mlfn.empty_directory(input_faulty_path)
mlfn.empty_directory(window_healthy_path)
mlfn.empty_directory(window_faulty_path)

# Copy contents from the source folders to the destination folders

#mlfn.copy_folder_contents(dataset1_healthy_path, input_healthy_path)
#mlfn.center_crop_folder(dataset2_healthy_path, input_healthy_path, (1750, 1750))
mlfn.copy_folder_contents(dataset3_healthy_path, input_healthy_path)

#mlfn.copy_folder_contents(dataset1_faulty_path, input_faulty_path)
#mlfn.center_crop_folder(dataset2_faulty_path, input_faulty_path, (1750, 1750))
mlfn.copy_folder_contents(dataset3_faulty_path, input_faulty_path)

# Generate windowed datasets

image_index = 0
image_index = mlfn.rename_files_in_dataset(input_faulty_path, 'faulty', image_index)
image_index = mlfn.rename_files_in_dataset(input_healthy_path, 'healthy', image_index)
mlfn.generate_windowed_dataset(input_faulty_path, window_size, stride, border_type, window_faulty_path)
mlfn.generate_windowed_dataset(input_healthy_path, window_size, stride, border_type, window_healthy_path)


