import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
import paths
import functions_th as th
project_root = Path(__file__).resolve().parent



if __name__ == "__main__":
    # Start processing
    start = time.time()

    # Path to your image
    _, base_shape_path, _, recomposed_path =paths.define_files("green_ok", project_root)  # Path to the base schematic image

    recomposed = cv2.imread(recomposed_path, cv2.IMREAD_GRAYSCALE)
    # Preprocess the image
    base_contours, aligned_base_thresh = th.preprocess(recomposed)

    # Plot the results
    plt.figure(figsize=(10, 6))

    # Aligned mask
    plt.imshow(aligned_base_thresh, cmap='gray')
    plt.title('Aligned Mask')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    cv2.imwrite(base_shape_path, aligned_base_thresh)
    print(f"Execution time: {time.time() - start:.2f} seconds")