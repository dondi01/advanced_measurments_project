import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

def get_centroid_and_angle(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        (cx, cy), (w, h), angle = rect
        # Adjust angle so that it always represents the "long" side as vertical
        if w < h:
            angle = angle + 90
        return (int(cx), int(cy)), angle
    return None, 0

# 1. Load all images for background computation
image_files = sorted(glob.glob("C:/Users/franc/Desktop/Scuola/Measurement/advanced_measurments_project/dataset_piccoli/Scorre_verde/Buco_in_piu/*.png"))
frames_all = [cv2.imread(f, cv2.IMREAD_COLOR) for f in image_files]
frames_all = [f for f in frames_all if f is not None and f.shape == frames_all[0].shape]
H, W, C = frames_all[0].shape

# 2. Compute background from all images
stacked_frames = np.stack(frames_all, axis=3)
background = np.median(stacked_frames, axis=3).astype(np.uint8)

# 3. Select only images 3, 4, 5 for subject processing (indices 2, 3, 4)
selected_indices = [3, 4, 5]
frames = [frames_all[i] for i in selected_indices]

# 4. Extract masks for selected frames
masks = []
for frame in frames:
    diff = cv2.absdiff(frame, background)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_diff, 6, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    masks.append(mask)

# 5. Get centroids and angles
centroids = []
angles = []
for mask in masks:
    centroid, angle = get_centroid_and_angle(mask)
    centroids.append(centroid)
    angles.append(angle)

valid_centroids = [pt for pt in centroids if pt is not None]
valid_angles = [a for pt, a in zip(centroids, angles) if pt is not None]
mean_cx = int(np.mean([pt[0] for pt in valid_centroids]))
mean_cy = int(np.mean([pt[1] for pt in valid_centroids]))
mean_angle = np.mean(valid_angles)

# 6. Align the frames based on subject position and orientation
aligned_subjects = []
for frame, mask, centroid, angle in zip(frames, masks, centroids, angles):
    if centroid is None:
        continue
    dx = mean_cx - centroid[0]
    dy = mean_cy - centroid[1]
    d_angle = mean_angle - angle

    # Rotate around the centroid
    M_rot = cv2.getRotationMatrix2D((centroid[0], centroid[1]), d_angle, 1.0)
    rotated_frame = cv2.warpAffine(frame, M_rot, (W, H))
    rotated_mask = cv2.warpAffine(mask, M_rot, (W, H))

    # Translate to mean position
    M_trans = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted_frame = cv2.warpAffine(rotated_frame, M_trans, (W, H))
    shifted_mask = cv2.warpAffine(rotated_mask, M_trans, (W, H))

    subject = cv2.bitwise_and(shifted_frame, shifted_frame, mask=shifted_mask)
    aligned_subjects.append(subject)

# 7. Blend aligned subjects to reconstruct clean view (use max for black background)
if aligned_subjects:
    aligned_stack = np.stack(aligned_subjects, axis=3)
    reconstructed_subject = np.max(aligned_stack, axis=3).astype(np.uint8)

else:
    reconstructed_subject = np.zeros((H, W, C), dtype=np.uint8)

# 8. Plot contours for each selected frame
for i, (frame, mask) in enumerate(zip(frames, masks)):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = frame.copy()
    cv2.drawContours(contour_img, cnts, -1, (0,255,0), 2)
    plt.figure()
    plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Contours on Frame {i+3}")  # +3 to match original numbering
    plt.axis('off')
    plt.show()

# 9. Show result
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(reconstructed_subject, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Reconstructed Subject (Color)")
plt.show()