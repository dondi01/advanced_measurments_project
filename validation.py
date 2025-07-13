import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import scipy.io
import time
from pathlib import Path
import paths
import functions_th as th
import  Panorama_Switching as ps
import sys
sys.path.append('./functions')
import ml_functions as mlfn
project_root = Path(__file__).resolve().parent


def process_images(case):
    p_s=ps.reassembler()
    mat = scipy.io.loadmat(p_s.project_root / 'dataset_medi' / 'TARATURA' / 'medium_dataset_taratura.mat')
    camera_matrix = mat['K']
    dist_coeffs = mat['dist']
    filepath, base_shape, _, recomposed_path = paths.define_files(case,p_s.project_root)
    image_files = sorted(
        glob.glob(filepath),
        key=th.alphanum_key)
    frames = [cv2.imread(f, cv2.IMREAD_COLOR) for f in image_files]
    frames = [f for f in frames if f is not None and f.shape == frames[0].shape]
    p_s.camera_matrix = camera_matrix
    p_s.dist_coeffs = dist_coeffs
    res = p_s.run_panorama_pipeline(frames,save_path=None,show_plots=False)
    return res

window_size = (100, 100)
stride = 50
output_path_reference = str(project_root / "validation_data" / "windows_reference")
output_path_test = str(project_root / "validation_data" / "windows_test")
mlfn.empty_directory(output_path_reference)
mlfn.empty_directory(output_path_test)
reference = process_images('green_ok')
test = process_images('green_buco_in_meno')
#test = mlfn.pad_image_for_sliding_window(image, window_size, stride, border_type) 
mlfn.generate_windows(0, reference, window_size, stride, output_path_reference)
mlfn.generate_windows(0, test, window_size, stride, output_path_test)



plt.figure(figsize=(10, 8))
if reference.ndim == 2:  # grayscale
    plt.imshow(reference, cmap='gray')
else:  # color
    plt.imshow(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
plt.title("Panorama Result")
plt.axis('off')
plt.show()