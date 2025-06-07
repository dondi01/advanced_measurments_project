import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import scipy.io
import re

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]

# 1. Load distortion matrix from .mat file
mat = scipy.io.loadmat("C:/Users/franc/Desktop/Scuola/Measurement/advanced_measurments_project/dataset_medi/TARATURA/medium_dataset_taratura.mat")
camera_matrix = mat['K']
dist_coeffs = mat['dist']

# 2. Load all images
image_files = sorted(
    glob.glob("C:/Users/franc/Desktop/Scuola/Measurement/advanced_measurments_project/dataset_medi/Scorre_nappies/*.png"),
    key=alphanum_key)
frames = [cv2.imread(f, cv2.IMREAD_COLOR) for f in image_files]
frames = [f for f in frames if f is not None and f.shape == frames[0].shape]

# 3. Undistort all images
frames = [
    cv2.undistort(f, camera_matrix, dist_coeffs) for f in frames
]

# 4. ORB feature matcher setup
orb = cv2.ORB_create(1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# --- Compute offsets for alignment ---
offsets = [0]
for i in range(1, len(frames)):
    img1 = frames[i-1]
    img2 = frames[i]
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        print(f"Skipping frame {i} due to insufficient features.")
        offsets.append(offsets[-1])
        continue

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:10]

    # Visualize matches (optional)
    # match_img = cv2.drawMatches(
    #     img1, kp1, img2, kp2, good_matches, None,
    #     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    # )
    # plt.figure(figsize=(15, 8))
    # plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    # plt.title(f"Matches between frame {i-1} and {i}")
    # plt.axis('off')
    # plt.show()

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,2)
    dxs = dst_pts[:,0] - src_pts[:,0]
    median_dx = np.median(dxs)

    # NEGATE the shift to align the carton (undo its movement)
    current_offset = offsets[-1] - median_dx
    offsets.append(current_offset)
    print(f"Frame {i}: carton moved {median_dx:.2f} px right, so shift image {(-median_dx):.2f} px left (offset: {current_offset:.2f})")

# Normalize offsets so the minimum is zero
min_offset = int(min(offsets))
offsets = [int(o - min_offset) for o in offsets]

# --- Blend images into panorama ---
H, W, C = frames[0].shape
panorama_width = max(offsets) + W
panorama_height = H

panorama_sum = np.zeros((panorama_height, panorama_width, C), dtype=np.float32)
panorama_count = np.zeros((panorama_height, panorama_width), dtype=np.float32)

for i, img in enumerate(frames):
    x_start = offsets[i]
    x_end = x_start + W
    mask = np.any(img > 10, axis=2)
    for c in range(C):
        panorama_sum[:, x_start:x_end, c] += img[..., c] * mask
    panorama_count[:, x_start:x_end] += mask

# Avoid division by zero
panorama_count[panorama_count == 0] = 1
panorama_avg = (panorama_sum / panorama_count[..., None]).astype(np.uint8)

# Crop to non-empty region (optional)
gray_panorama = cv2.cvtColor(panorama_avg, cv2.COLOR_BGR2GRAY)
coords = cv2.findNonZero((gray_panorama > 0).astype(np.uint8))
x, y, w, h = cv2.boundingRect(coords)
panorama_cropped = panorama_avg[y:y+h, x:x+w]

# Show the result
plt.figure(figsize=(15, 8))
plt.imshow(cv2.cvtColor(panorama_cropped, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Reconstructed Subject")
plt.show()