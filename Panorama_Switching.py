import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import scipy.io
import re
from scipy.ndimage import distance_transform_edt
import time
from numba import njit, prange
from pathlib import Path

project_root = Path(__file__).resolve().parent

#"C:/Users/franc/Desktop/Scuola/Measurement/advanced_measurments_project/dataset_medi/Scorre_marrone/*.png"
#"C:/Users/franc/Desktop/Scuola/Measurement/advanced_measurments_project/dataset_medi/Scorre_parmareggio_no/*.png"
#"C:/Users/franc/Desktop/Scuola/Measurement/advanced_measurments_project/dataset_medi/Scorre_nappies/*.png"
#"C:/Users/franc/Desktop/Scuola/Measurement/advanced_measurments_project/dataset_piccoli/Scorre_verde/Buco_in_meno/*.png"

#Define the file path to the images
filepath = str(project_root / 'dataset_piccoli' / 'Scorre_verde' / 'Buco_in_meno' / '*.png')

#filepath="C:/Users/franc/Desktop/Scuola/Measurement/advanced_measurments_project/dataset_piccoli/Scorre_verde/Buco_in_meno/*.png"




# Function to convert strings to integers where possible, useful to order files
def tryint(s):
    try:
        return int(s)
    except:
        return s

#@njit(fastmath=True,parallel=True)
def apply_feather(feather, warped, panorama, C) -> np.ndarray:
    for c in range(C):
        panorama[..., c] += warped[..., c] * feather
    return panorama
    

# Function to split strings into alphanumeric parts for sorting, also considers case where it's not just the number
#but string + number
def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]

#@profile
def main(frames, orb, bf) -> np.ndarray:

    frames=frames[::2]
    # 3. Undistort all images
    frames = [cv2.undistort(f, camera_matrix, dist_coeffs) for f in frames]
    crop_w, crop_h = 1556, 1052
    H, W, C = frames[0].shape
    center_x, center_y = W // 2, H // 2
    x1 = center_x - crop_w // 2
    y1 = center_y - crop_h // 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    frames = [f[y1:y2, x1:x2] for f in frames]

    #Define image size
    H, W, C = frames[0].shape

    #Initialize transforms matrix list with identity matrix
    transforms = [np.eye(3, dtype=np.float32)]  # 3x3 identity for homogenous coordinates

    # --- Optimized: Precompute keypoints and descriptors for the first image ---
    gray_prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    kp_prev, des_prev = orb.detectAndCompute(gray_prev, None)

    # Estimate transforms between frames
    for i in range(1, len(frames)):
        #Take current image
        img_curr = frames[i]
        gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)
        kp_curr, des_curr = orb.detectAndCompute(gray_curr, None)

        #If no features are found, skip this frame
        if des_prev is None or des_curr is None or len(kp_prev) < 4 or len(kp_curr) < 4:
            print(f"Skipping frame {i} due to insufficient features.")
            transforms.append(transforms[-1].copy())
            kp_prev, des_prev = kp_curr, des_curr  # Move forward
            continue
        #Match features that are common to both images
        matches = bf.match(des_prev, des_curr)
        matches = sorted(matches, key=lambda x: x.distance)
        #Extract the 10 best matches(they are ordered by distance, 
        # so the first ones are the best)
        good_matches = matches[:10]

        #Extract the matched keypoints from both images, queryIdx are for img1,
        # trainIdx are for img2. It's a for cycle that goes through the matches
        src_pts = np.float32([kp_prev[m.queryIdx].pt for m in good_matches]).reshape(-1,2)
        dst_pts = np.float32([kp_curr[m.trainIdx].pt for m in good_matches]).reshape(-1,2)

        # Estimate affine transform, check how much the subject has moved, to understand
        #how to align the next image
        M, inliers = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC)
        # If M is None, it means the estimation failed, so we skip this frame
        if M is None:
            print(f"Skipping frame {i} due to failed affine estimation.")
            transforms.append(transforms[-1].copy())
            kp_prev, des_prev = kp_curr, des_curr  # Move forward
            continue

        #Affine transformations only need a 2x3 matrix, they represent a linear transformation
        #that includes shearing, rotation, translation and scaling. To combine multiple ones we need
        #to apply this transformation into  a 3x3 matrix
        M33 = np.eye(3, dtype=np.float32)
        M33[:2, :3] = M

        # Compose with previous transform through matrix multiplication and added to the list,
        # so that we can apply the cumulative transformation to the next image
        cumulative = transforms[-1] @ M33
        transforms.append(cumulative)

        # Move forward: current becomes previous for next iteration
        kp_prev, des_prev = kp_curr, des_curr

    # --- Compute panorama size ---
    #This defines the corners of the images in homogeneous coordinates, the same of the affine
    #transormations, it's a reference system where the top-left corner is (0,0) 
    # and the bottom-right corner is (W,H), where W and H are the width and height of the images
    corners = np.array([
        [0, 0, 1],
        [W, 0, 1],
        [W, H, 1],
        [0, H, 1]
    ], dtype=np.float32)

    #Put each transform in the reference system of the panorama
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
    panorama = np.zeros((panorama_height, panorama_width, C), dtype=np.float32)
    weight = np.zeros((panorama_height, panorama_width), dtype=np.float32)

    # Warp (rototranslate+shearing) images and blend them into the panorama
    for i, img in enumerate(frames):
        #For each point in each image we apply the transformation

        #We set the transformation matrix with the computed buondaries of the panorama
        offset_M = transforms[i].copy()
        offset_M[0,2] -= x_min
        offset_M[1,2] -= y_min

        #We compute the warping of the image
        warped = cv2.warpAffine(img, offset_M[:2], (panorama_width, panorama_height))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.warpAffine((gray_img > 20).astype(np.float32), offset_M[:2], (panorama_width, panorama_height))
        # mask = cv2.warpAffine((img[...,0] > 20).astype(np.float32), offset_M[:2], (panorama_width, panorama_height))


        # To avoid lines in the final result we blend the images together using a feathering 
        # technique, that decreases the weight of the pixels at the edges, making the images blend together
        # more smoothly
        small_mask = cv2.resize(mask, (mask.shape[1] // 8, mask.shape[0] // 8), interpolation=cv2.INTER_LINEAR)
        small_blur = cv2.blur(small_mask, (21, 21))
        feather = cv2.resize(small_blur, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        feather = np.clip(feather, 1e-3, 1.0)

        #For each color channel we add the warped image to the panorama, weighted 
        #by the feathering mask
        panorama=apply_feather(feather, warped,panorama,C)
        # for c in range(C):
        #     panorama[..., c] += warped[..., c] * feather
        weight += feather

    # Avoid division by zero in normalization
    weight[weight == 0] = 1

    # Normalize on the weight map to avoid overexposure
    panorama /= weight[..., None]
    #Clip values to valid range for display
    panorama = np.clip(panorama, 0, 255).astype(np.uint8)

    # Crop to non-empty region, since panorama is sized so that even if no images match 
    #it's big enough to fit them all
    gray_panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero((gray_panorama > 0).astype(np.uint8))
    x, y, w, h = cv2.boundingRect(coords)
    panorama_cropped = panorama[y:y+h, x:x+w]
    return panorama_cropped


# 1. Load distortion matrix from .mat file
mat = scipy.io.loadmat(project_root / 'dataset_medi' / 'TARATURA' / 'medium_dataset_taratura.mat')

# mat = scipy.io.loadmat("C:/Users/franc/Desktop/Scuola/Measurement/advanced_measurments_project/dataset_medi/TARATURA/medium_dataset_taratura.mat")
camera_matrix = mat['K']
dist_coeffs = mat['dist']
# 2. Load all images
image_files = sorted(
    glob.glob(filepath),
    key=alphanum_key)
frames = [cv2.imread(f, cv2.IMREAD_COLOR) for f in image_files]
frames = [f for f in frames if f is not None and f.shape == frames[0].shape]


#ORB feature matcher setup, to estimate subject movement by matching similar features
orb = cv2.ORB_create(200)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
orb.detectAndCompute(frames[0], None)  # Initialize ORB
start=time.time()
res=main(frames,orb,bf)

#UNCOMMENT IF PICTURE IS DARK
value = 100
res = cv2.add(res, np.ones(res.shape, dtype='uint8') * value)
cv2.imwrite(project_root / 'Reconstructed' / 'Green_Buco_in_meno.png', res)
#cv2.imwrite("C:/Users/franc/Desktop/Scuola/Measurement/advanced_measurments_project/Reconstructed/Green_Buco_in_meno.png", res)
print("Execution time is:",time.time()-start)

# # Show the result
plt.figure(figsize=(15, 8))
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.axis('off')
#plt.title("Reconstructed Subject (Feather Blend)")
plt.show()