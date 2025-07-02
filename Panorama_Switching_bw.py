import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import scipy.io
import re
import time

# Define the file path to the images
filepath = "C:/Users/franc/Desktop/Scuola/Measurement/advanced_measurments_project/dataset_medi/Scorre_nappies/*.png"

# Function to convert strings to integers where possible, useful to order files
def tryint(s):
    try:
        return int(s)
    except:
        return s

# Function to split strings into alphanumeric parts for sorting, also considers case where it's not just the number
# but string + number
def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]

#@profile
def main(frames, orb, bf) -> np.ndarray:

    # Define image size (grayscale, so only H, W)
    H, W = frames[0].shape

    # Initialize transforms matrix list with identity matrix
    transforms = [np.eye(3, dtype=np.float32)]  # 3x3 identity for homogenous coordinates

    # --- Optimized: Precompute keypoints and descriptors for the first image ---
    kp_prev, des_prev = orb.detectAndCompute(frames[0], None)

    # Estimate transforms between frames
    for i in range(1, len(frames)):
        # Take current image
        img_curr = frames[i]
        kp_curr, des_curr = orb.detectAndCompute(img_curr, None)

        # If no features are found, skip this frame
        if des_prev is None or des_curr is None or len(kp_prev) < 4 or len(kp_curr) < 4:
            print(f"Skipping frame {i} due to insufficient features.")
            transforms.append(transforms[-1].copy())
            kp_prev, des_prev = kp_curr, des_curr  # Move forward
            continue
        # Match features that are common to both images
        matches = bf.match(des_prev, des_curr)
        matches = sorted(matches, key=lambda x: x.distance)
        # Extract the 10 best matches (they are ordered by distance, so the first ones are the best)
        good_matches = matches[:10]

        # Extract the matched keypoints from both images
        src_pts = np.float32([kp_prev[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([kp_curr[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        # Estimate affine transform
        M, inliers = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC)
        if M is None:
            print(f"Skipping frame {i} due to failed affine estimation.")
            transforms.append(transforms[-1].copy())
            kp_prev, des_prev = kp_curr, des_curr  # Move forward
            continue

        # Affine transformations only need a 2x3 matrix, they represent a linear transformation
        # that includes shearing, rotation, translation and scaling. To combine multiple ones we need
        # to apply this transformation into a 3x3 matrix
        M33 = np.eye(3, dtype=np.float32)
        M33[:2, :3] = M

        # Compose with previous transform through matrix multiplication and add to the list
        cumulative = transforms[-1] @ M33
        transforms.append(cumulative)

        # Move forward: current becomes previous for next iteration
        kp_prev, des_prev = kp_curr, des_curr

    # --- Compute panorama size ---
    # This defines the corners of the images in homogeneous coordinates
    corners = np.array([
        [0, 0, 1],
        [W, 0, 1],
        [W, H, 1],
        [0, H, 1]
    ], dtype=np.float32)

    # Put each transform in the reference system of the panorama
    all_corners = []
    for T in transforms:
        warped = (T @ corners.T).T
        all_corners.append(warped[:, :2])

    # Stack all corners and find the min and max coordinates to define the panorama size
    all_corners = np.vstack(all_corners)
    x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)

    # Calculate panorama dimensions
    panorama_width = x_max - x_min
    panorama_height = y_max - y_min

    # Initialize panorama and weight maps, the weights are used to blend the images together
    panorama = np.zeros((panorama_height, panorama_width), dtype=np.float32)
    weight = np.zeros((panorama_height, panorama_width), dtype=np.float32)

    # Warp (rototranslate+shearing) images and blend them into the panorama
    for i, img in enumerate(frames):
        # Set the transformation matrix with the computed boundaries of the panorama
        offset_M = transforms[i].copy()
        offset_M[0, 2] -= x_min
        offset_M[1, 2] -= y_min

        # Compute the warping of the image
        warped = cv2.warpAffine(img, offset_M[:2], (panorama_width, panorama_height))
        mask = cv2.warpAffine((img > 20).astype(np.float32), offset_M[:2], (panorama_width, panorama_height))

        # Feathering for smooth blending
        feather = cv2.blur(mask, (201, 201))  # Try (31, 31) or (51, 51) for speed
        feather = np.clip(feather, 1e-3, 1.0)

        # Add the warped image to the panorama, weighted by the feathering mask
        panorama += warped * feather
        weight += feather

    # Avoid division by zero in normalization
    weight[weight == 0] = 1

    # Normalize on the weight map to avoid overexposure
    panorama /= weight
    panorama = np.clip(panorama, 0, 255).astype(np.uint8)

    # Crop to non-empty region
    coords = cv2.findNonZero((panorama > 0).astype(np.uint8))
    x, y, w, h = cv2.boundingRect(coords)
    panorama_cropped = panorama[y:y+h, x:x+w]
    return panorama_cropped

# 1. Load distortion matrix from .mat file
mat = scipy.io.loadmat("C:/Users/franc/Desktop/Scuola/Measurement/advanced_measurments_project/dataset_medi/TARATURA/medium_dataset_taratura.mat")
camera_matrix = mat['K']
dist_coeffs = mat['dist']

# 2. Load all images and convert to grayscale
image_files = sorted(
    glob.glob(filepath),
    key=alphanum_key)
frames = [cv2.imread(f, cv2.IMREAD_COLOR) for f in image_files]
frames = [f for f in frames if f is not None and f.shape == frames[0].shape]
frames = [cv2.undistort(f, camera_matrix, dist_coeffs) for f in frames]
gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

# ORB feature matcher setup
orb = cv2.ORB_create(1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
orb.detectAndCompute(gray_frames[0], None)  # Initialize ORB

start = time.time()
res = main(gray_frames, orb, bf)
print("Execution time is:", time.time() - start)

# Show the result
plt.figure(figsize=(15, 8))
plt.imshow(res, cmap='gray')
plt.axis('off')
plt.title("Reconstructed Subject (Grayscale Feather Blend)")
plt.show()