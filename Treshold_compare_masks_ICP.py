import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.spatial.distance import cdist

def get_main_object_contour(contours, image_shape, area_thresh=0.9):
    """Get the contour of the main body by finding the largest one,
    excluding ones that are too big (more than 90% of the pic) to avoid the background"""
    img_area = image_shape[0] * image_shape[1]
    filtered = [c for c in contours if cv2.contourArea(c) < area_thresh * img_area]
    if not filtered:
        return None
    return max(filtered, key=cv2.contourArea)

def preprocess(image_path):
    """Takes the image path, reads the image, lightens it, converts it to grayscale,
    blurs it, and thresholds it to create a binary mask."""
    image = cv2.imread(image_path)
    lightened = cv2.convertScaleAbs(image, alpha=1, beta=100)
    gray = cv2.cvtColor(lightened, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21,21), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return image, contours, thresh

def get_contour_center(contour):
    """Calculate the centroid of a contour"""
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return (0, 0)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx, cy)

def normalize_contour(contour, center=None):
    """Normalize contour points around their centroid"""
    if center is None:
        center = get_contour_center(contour)
    
    points = contour.reshape(-1, 2).astype(np.float32)
    normalized = points - np.array(center)
    return normalized

def icp_alignment(contour1, contour2, max_iterations=50, tolerance=0.1):
    """
    Align contour1 to contour2 using Iterative Closest Point algorithm.
    Returns: transformation parameters (translation, rotation, scale)
    """
    # Normalize both contours to their centroids
    center1 = get_contour_center(contour1)
    center2 = get_contour_center(contour2)
    
    points1 = normalize_contour(contour1, center1)
    points2 = normalize_contour(contour2, center2)
    
    # Subsample points if too many (for efficiency)
    if len(points1) > 100:
        indices = np.linspace(0, len(points1)-1, 100, dtype=int)
        points1 = points1[indices]
    if len(points2) > 100:
        indices = np.linspace(0, len(points2)-1, 100, dtype=int)
        points2 = points2[indices]
    
    # Initialize transformation parameters
    total_translation = np.array([0.0, 0.0])
    total_rotation = 0.0
    scale_factor = 1.0
    
    current_points = points1.copy()
    
    for iteration in range(max_iterations):
        # Find closest points
        distances = cdist(current_points, points2)
        closest_indices = np.argmin(distances, axis=1)
        closest_points = points2[closest_indices]
        
        # Calculate transformation using least squares
        # Translation
        translation = np.mean(closest_points - current_points, axis=0)
        
        # Apply translation
        current_points += translation
        total_translation += translation
        
        # Calculate rotation using cross-correlation
        cross_corr = np.sum(current_points * closest_points, axis=0)
        auto_corr1 = np.sum(current_points * current_points, axis=0)
        auto_corr2 = np.sum(closest_points * closest_points, axis=0)
        
        # Simple rotation estimation (for small angles)
        if auto_corr1[0] != 0 and auto_corr1[1] != 0:
            rotation = np.arctan2(cross_corr[1], cross_corr[0]) - np.arctan2(auto_corr1[1], auto_corr1[0])
            
            # Apply rotation
            cos_r, sin_r = np.cos(rotation), np.sin(rotation)
            rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            current_points = current_points @ rotation_matrix.T
            total_rotation += rotation
        
        # Calculate scale
        norm1 = np.linalg.norm(current_points, axis=1)
        norm2 = np.linalg.norm(closest_points, axis=1)
        valid_norms = (norm1 > 0) & (norm2 > 0)
        
        if np.sum(valid_norms) > 0:
            scale = np.mean(norm2[valid_norms] / norm1[valid_norms])
            current_points *= scale
            scale_factor *= scale
        
        # Check convergence
        mean_error = np.mean(np.linalg.norm(current_points - closest_points, axis=1))
        if mean_error < tolerance:
            print(f"ICP converged after {iteration + 1} iterations")
            break
    
    # Calculate final transformation from center1 to center2
    final_translation = np.array(center2) - np.array(center1) + total_translation
    
    return {
        'translation': final_translation,
        'rotation': total_rotation,
        'scale': scale_factor,
        'center1': center1,
        'center2': center2
    }

def apply_transformation(image, transform_params):
    """Apply the transformation found by ICP to an image"""
    translation = transform_params['translation']
    rotation = transform_params['rotation']
    scale = transform_params['scale']
    center = transform_params['center1']
    
    h, w = image.shape[:2]
    
    # Create transformation matrix
    # First translate to origin, then scale, rotate, then translate back
    cos_r, sin_r = np.cos(rotation), np.sin(rotation)
    
    # Combined transformation matrix
    M = np.array([
        [scale * cos_r, -scale * sin_r, translation[0] - center[0] * scale * cos_r + center[1] * scale * sin_r + center[0]],
        [scale * sin_r, scale * cos_r, translation[1] - center[0] * scale * sin_r - center[1] * scale * cos_r + center[1]]
    ], dtype=np.float32)
    
    # Apply transformation
    transformed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    return transformed, M

def center_crop(img, target_shape):
    """Center crop image to target shape"""
    h, w = img.shape[:2]
    th, tw = target_shape
    y1 = max((h - th) // 2, 0)
    x1 = max((w - tw) // 2, 0)
    y2 = y1 + th
    x2 = x1 + tw
    return img[y1:y2, x1:x2]

def center_pad(img, target_shape):
    """Center pad image to target shape"""
    pad_vert = (target_shape[0] - img.shape[0]) // 2
    pad_horz = (target_shape[1] - img.shape[1]) // 2
    img = cv2.copyMakeBorder(
        img,
        pad_vert, target_shape[0] - img.shape[0] - pad_vert,
        pad_horz, target_shape[1] - img.shape[1] - pad_horz,
        cv2.BORDER_CONSTANT, value=0
    )
    return img

def align_images_icp(base_thresh, test_thresh, base_contours, test_contours):
    """Align test image to base image using ICP on main object contours"""
    
    # Get main contours
    base_main_contour = get_main_object_contour(base_contours, base_thresh.shape)
    test_main_contour = get_main_object_contour(test_contours, test_thresh.shape)
    
    if base_main_contour is None or test_main_contour is None:
        print("Could not find main contours for alignment")
        return test_thresh, None, None, None
    
    print(f"Base contour points: {len(base_main_contour)}")
    print(f"Test contour points: {len(test_main_contour)}")
    
    # Perform ICP alignment
    transform_params = icp_alignment(test_main_contour, base_main_contour)
    
    print(f"ICP Results:")
    print(f"  Translation: {transform_params['translation']}")
    print(f"  Rotation: {np.degrees(transform_params['rotation']):.2f} degrees")
    print(f"  Scale: {transform_params['scale']:.3f}")
    
    # Apply transformation to test image
    aligned_test, transform_matrix = apply_transformation(test_thresh, transform_params)
    
    return aligned_test, transform_params, base_main_contour, test_main_contour

# Main execution
start = time.time()

# Paths to your images
base_path = "C:/Users/franc/Desktop/Scuola/Measurement/advanced_measurments_project/Reconstructed/parmareggio_no.png"
test_path = "C:/Users/franc/Desktop/Scuola/Measurement/advanced_measurments_project/Reconstructed/parmareggio_ok.png"

# Preprocess both images
base_img, base_contours, base_thresh = preprocess(base_path)
test_img, test_contours, test_thresh = preprocess(test_path)

# Align test image to base image using ICP
aligned_test_thresh, transform_params, base_main_contour, test_main_contour = align_images_icp(
    base_thresh, test_thresh, base_contours, test_contours
)

# Ensure the masks have the same size
if aligned_test_thresh.shape != base_thresh.shape:
    min_h = min(base_thresh.shape[0], aligned_test_thresh.shape[0])
    min_w = min(base_thresh.shape[1], aligned_test_thresh.shape[1])
    aligned_base_thresh = center_crop(base_thresh, (min_h, min_w))
    aligned_test_thresh = center_crop(aligned_test_thresh, (min_h, min_w))
else:
    aligned_base_thresh = base_thresh

# Compute absolute difference between the two masks
diff_mask = cv2.absdiff(aligned_base_thresh, aligned_test_thresh)

# Clean up small noise
kernel = np.ones((3, 3), np.uint8)
diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)
kernel = np.ones((5, 5), np.uint8)
diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)

# Remove small contours
contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_area = 200
clean_mask = np.zeros_like(diff_mask)
for cnt in contours:
    if cv2.contourArea(cnt) > min_area:
        cv2.drawContours(clean_mask, [cnt], -1, 255, -1)
diff_mask = clean_mask

print(f"Execution time: {time.time() - start:.2f} seconds")

# Visualizations
# Plot the difference mask
plt.figure(figsize=(7, 6))
plt.imshow(diff_mask, cmap='gray')
plt.title('Difference Mask (White = Difference)')
plt.axis('off')
plt.show()

# Align the color test image
if transform_params is not None:
    aligned_test_img, _ = apply_transformation(test_img, transform_params)
    
    # Ensure same size as difference mask
    if aligned_test_img.shape[:2] != diff_mask.shape[:2]:
        aligned_test_img = cv2.resize(aligned_test_img, (diff_mask.shape[1], diff_mask.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Highlight differences on test image
    highlight = aligned_test_img.copy()
    highlight[diff_mask > 0] = [0, 0, 255]
    
    plt.figure(figsize=(7, 6))
    plt.imshow(cv2.cvtColor(highlight, cv2.COLOR_BGR2RGB))
    plt.title('Differences Highlighted on Test Image (ICP Aligned)')
    plt.axis('off')
    plt.show()

# Create overlay visualization
overlay = np.zeros((aligned_base_thresh.shape[0], aligned_base_thresh.shape[1], 3), dtype=np.uint8)
overlay[:, :, 1] = aligned_base_thresh  # Green channel
overlay[:, :, 2] = aligned_test_thresh  # Red channel

plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.imshow(overlay)
plt.title('ICP Aligned Overlay\n(Green=Base, Red=Test, Yellow=Overlap)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(diff_mask, cmap='gray')
plt.title('Difference Mask\n(White=Different)')
plt.axis('off')

plt.tight_layout()
plt.show()

# Plot the aligned masks side by side
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(aligned_base_thresh, cmap='gray')
plt.title('Base Mask')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(aligned_test_thresh, cmap='gray')
plt.title('ICP Aligned Test Mask')
plt.axis('off')

plt.show()

# Optional: Visualize the contours used for ICP
if base_main_contour is not None and test_main_contour is not None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Base contour
    base_contour_img = cv2.cvtColor(base_thresh, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(base_contour_img, [base_main_contour], -1, (0, 255, 0), 2)
    ax1.imshow(cv2.cvtColor(base_contour_img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Base Main Contour')
    ax1.axis('off')
    
    # Test contour
    test_contour_img = cv2.cvtColor(test_thresh, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(test_contour_img, [test_main_contour], -1, (0, 255, 0), 2)
    ax2.imshow(cv2.cvtColor(test_contour_img, cv2.COLOR_BGR2RGB))
    ax2.set_title('Test Main Contour (Before Alignment)')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()