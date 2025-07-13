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
import routine as rt
import csv

project_root = Path(__file__).resolve().parent


def process_images(case, load_panorama=False):
    p_s=ps.reassembler()
    mat = scipy.io.loadmat(p_s.project_root / 'dataset_medi' / 'TARATURA' / 'medium_dataset_taratura.mat')
    camera_matrix = mat['K']
    dist_coeffs = mat['dist']
    filepath, base_shape, base_print, recomposed_path = paths.define_files(case,p_s.project_root)
    image_files = sorted(
        glob.glob(filepath),
        key=th.alphanum_key)
    frames = [cv2.imread(f, cv2.IMREAD_COLOR) for f in image_files]
    frames = [f for f in frames if f is not None and f.shape == frames[0].shape]
    p_s.camera_matrix = camera_matrix
    p_s.dist_coeffs = dist_coeffs
    res = p_s.run_panorama_pipeline(frames,save_path=None,show_plots=False)
    base_shape_img = cv2.imread(base_shape, cv2.IMREAD_COLOR)
    base_print_img = cv2.imread(base_print, cv2.IMREAD_COLOR)
    if load_panorama:
        recomposed_img = cv2.imread(recomposed_path, cv2.IMREAD_COLOR)
        if recomposed_img is None:
            print(f"Error: Panorama image {recomposed_path} not found.")
            return None, None, None
        res = recomposed_img
    return res, base_shape_img, base_print_img

window_size = (200, 200)
stride = 100
output_path_schematic_shape = str(project_root / "validation_data" / "windows_reference_shape")
output_path_schematic_print = str(project_root / "validation_data" / "windows_reference_print")
output_path_test_shape = str(project_root / "validation_data" / "windows_test_shape")
output_path_test = str(project_root / "validation_data" / "windows_test")

healthy_path = str(project_root / "validation_data" / "classified_healthy")
faulty_path = str(project_root / "validation_data" / "classified_faulty")
# plt.figure(figsize=(10, 8))
# if schematic_print.ndim == 2:  # grayscale
#     plt.imshow(schematic_print, cmap='gray')
# else:  # color
#     plt.imshow(cv2.cvtColor(schematic_print, cv2.COLOR_BGR2RGB))
# plt.title("Panorama Result")
# plt.axis('off')
# plt.show()
mlfn.empty_directory(output_path_schematic_print)
mlfn.empty_directory(output_path_schematic_shape)
mlfn.empty_directory(output_path_test_shape)
mlfn.empty_directory(output_path_test)
mlfn.empty_directory(healthy_path)
mlfn.empty_directory(faulty_path)

aligned_base_thresh = {}
for i, case in enumerate(['green_ok', 'green_lettere_disallineate', 'green_buco_in_piu', 'green_buco_in_meno']):
    
    if case == 'green_ok':
        test, schematic, schematic_print = process_images(case, load_panorama=True)
    else:
        test, schematic, schematic_print = process_images(case)
    
    test = th.rescale_and_resize_mask(aligned_mask=test, target_img=schematic, pad_value=0)
    if schematic.shape != test.shape:
        test = th.match_size(schematic, test, pad_value=[0, 0, 0])
    # Generate windows
    _, aligned_base_thresh[i] = th.preprocess(test)
    mlfn.generate_windows(i, schematic, window_size, stride, output_path_schematic_shape)
    mlfn.generate_windows(i, schematic_print, window_size, stride, output_path_schematic_print)
    mlfn.generate_windows(i, aligned_base_thresh[i], window_size, stride, output_path_test_shape)
    mlfn.generate_windows(i, test, window_size, stride, output_path_test)
 


# classification 
routine = rt.Routine()
classification_results = {}
for i, window_path in enumerate(Path(output_path_test).iterdir()):
    filename = window_path.name
    window_path = Path(output_path_test) / filename
    window_shape_path = Path(output_path_test_shape) / filename
    schematic_shape_path = Path(output_path_schematic_shape) / filename
    schematic_print_path = Path(output_path_schematic_print) / filename
    #i, _, _ = mlfn.parse_windowed_filename_py(window_path)
    # loading windows
    window_test = cv2.imread(str(window_path), cv2.IMREAD_COLOR)
    window_test_shape = cv2.imread(str(window_shape_path), cv2.IMREAD_GRAYSCALE)
    window_schematic_shape = cv2.imread(str(schematic_shape_path), cv2.IMREAD_GRAYSCALE)
    window_schematic_print = cv2.imread(str(schematic_print_path), cv2.IMREAD_GRAYSCALE)

    if window_test is None or window_schematic_shape is None or window_schematic_print is None:
        print(f"Missing window for {filename}, skipping.")
        continue
    
    found_hole, found_scratch, found_disaligned = routine.run_full_analysis(
        frames=None,  
        base_shape=window_schematic_shape,
        base_print=window_schematic_print,
        base_image=window_test,
        show_plots=False,
        recomposed=window_test,
        skip_resizing=True,
        aligned_base_thresh = window_test_shape
    )

    if found_hole or found_scratch or found_disaligned:
        cv2.imwrite(Path(faulty_path) / filename, window_test)
    else:
        cv2.imwrite(Path(healthy_path) / filename, window_test)

    classification_results[filename] = {
        'found_hole': found_hole,
        'found_scratch': found_scratch,
        'found_disaligned': found_disaligned
    }

print("Classification Results:")
        
count_hole = sum(res['found_hole'] for res in classification_results.values())
count_scratch = sum(res['found_scratch'] for res in classification_results.values())
count_disaligned = sum(res['found_disaligned'] for res in classification_results.values())

print(f"Total windows: {len(classification_results)}")
print(f"Windows with holes: {count_hole}")
print(f"Windows with scratches: {count_scratch}")
print(f"Windows with disalignment: {count_disaligned}")

with open("classification_results.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write header
    writer.writerow(["filename", "found_hole", "found_scratch", "found_disaligned"])
    # Write data rows
    for filename, results in classification_results.items():
        writer.writerow([
            filename,
            int(results['found_hole']),
            int(results['found_scratch']),
            int(results['found_disaligned'])
        ])


# filename = 'window_0_at_100_100.png'  # Example filename, replace with actual logic to get filenames
# window_path = Path(output_path_test) / filename
# window_shape_path = Path(output_path_test_shape) / filename
# schematic_shape_path = Path(output_path_schematic_shape) / filename
# schematic_print_path = Path(output_path_schematic_print) / filename
# i, _, _ = mlfn.parse_windowed_filename_py(window_path)
# # loading windows
# window_test = cv2.imread(str(window_path), cv2.IMREAD_COLOR)
# window_test_shape = cv2.imread(str(window_shape_path), cv2.IMREAD_GRAYSCALE)
# window_schematic_shape = cv2.imread(str(schematic_shape_path), cv2.IMREAD_GRAYSCALE)
# window_schematic_print = cv2.imread(str(schematic_print_path), cv2.IMREAD_GRAYSCALE)


# found_hole, found_scratch, found_disaligned = routine.run_full_analysis(
#     frames=None,  # or '' if your method expects it
#     base_shape=window_schematic_shape,
#     base_print=window_schematic_print,
#     base_image=window_test,
#     show_plots=True,
#     recomposed=window_test,
#     skip_resizing=True,
#     aligned_base_thresh = window_test_shape
# )
# plt.figure(figsize=(18, 6))

# # First subplot: schematic
# plt.subplot(1, 3, 1)
# if schematic_used.ndim == 2:
#     plt.imshow(schematic_used, cmap='gray')
# else:
#     plt.imshow(cv2.cvtColor(schematic_used, cv2.COLOR_BGR2RGB))
# plt.title("Schematic")
# plt.axis('off')

# # Second subplot: test
# plt.subplot(1, 3, 2)
# if test.ndim == 2:
#     plt.imshow(test, cmap='gray')
# else:
#     plt.imshow(cv2.cvtColor(test, cv2.COLOR_BGR2RGB))
# plt.title("Test")
# plt.axis('off')

# # Third subplot: overlay
# plt.subplot(1, 3, 3)
# if schematic_used.ndim == 2:
#     plt.imshow(schematic_used, cmap='gray', alpha=0.7)
# else:
#     plt.imshow(cv2.cvtColor(schematic_used, cv2.COLOR_BGR2RGB), alpha=0.7)
# if test.ndim == 2:
#     plt.imshow(test, cmap='hot', alpha=0.3)
# else:
#     plt.imshow(cv2.cvtColor(test, cv2.COLOR_BGR2RGB), alpha=0.3)
# plt.title("Overlay")
# plt.axis('off')

# plt.tight_layout()
# plt.show()