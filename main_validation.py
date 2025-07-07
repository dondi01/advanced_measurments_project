import os
import cv2
import routine
import paths
from pathlib import Path
import time
import matplotlib.pyplot as plt
# Path to your dataset
DATASET_ROOT = r"C:/Users/franc/Desktop/Scuola/Measurement/Carton Defect Dataset_cut"
CATEGORIES = ["Normal","Breakage", "Color",  "Scratch"]

# Counters
wrong = 0
correct = 0

def get_base_images():
    _, base_shape_path, base_print_path, base_image_path = paths.define_files("external_ok", Path(__file__).resolve().parent)
    base_shape = cv2.imread(base_shape_path, cv2.IMREAD_GRAYSCALE)
    base_print = cv2.imread(base_print_path, cv2.IMREAD_GRAYSCALE)
    base_image = cv2.imread(base_image_path, cv2.IMREAD_COLOR)
    return base_shape, base_print, base_image

if __name__ == "__main__":
    base_shape, base_print, base_image = get_base_images()
    total = 0
    times = []
    for category in CATEGORIES:
        folder = os.path.join(DATASET_ROOT, category)
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                continue
            img_path = os.path.join(folder, fname)
            print(f"Processing {category}/{fname}")
            recomposed = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            recomposed=cv2.equalizeHist(recomposed)  # Optional: enhance contrast for better thresholding
            if recomposed is None:
                print(f"Could not read {img_path}, skipping.")
                continue
            # Downsample image to 50% size
            recomposed = cv2.resize(recomposed, (recomposed.shape[1] // 4, recomposed.shape[0] // 4), interpolation=cv2.INTER_AREA)
            start_time = time.time()
            result = routine.run_full_analysis(
                recomposed=recomposed,
                base_shape=base_shape,
                base_print=base_print,
                base_image=base_image,
                show_plots=False
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
            print(f"Execution time: {elapsed:.3f} seconds")
            is_fault = (category != "Normal")
            if any(result) != is_fault:
                print(f"WRONG: {category}/{fname} - Detected: {result}, Should be: {is_fault}")
                wrong += 1
            else:
                correct += 1
                print(f"CORRECT: {category}/{fname} - Detected: {result}, Should be: {is_fault}")
            total += 1
    print(f"Total images: {total}")
    print(f"Wrong detections: {wrong}")
    print(f"Accuracy: {(total-wrong)/total*100:.2f}%")
    if times:
        avg_time = sum(times) / len(times)
        print(f"Average execution time per image: {avg_time:.3f} seconds")