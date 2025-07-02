import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
import paths
project_root = Path(__file__).resolve().parent


def rescale_and_resize_mask(aligned_mask, mask_rect, target_rect, target_shape,pad_value=255):
    # Rescale and resize a mask so that its main object's rectangle matches the target rectangle's size,
    # then crop or pad to the target shape. Used for robust geometric normalization.
    if mask_rect is not None and target_rect is not None:
        # Find dimensions of the rectangles (sorted so width <= height)
        target_w, target_h = sorted(target_rect[1])
        mask_w, mask_h = sorted(mask_rect[1])
        # Compute scaling ratios
        ratio_w = target_w / mask_w if mask_w != 0 else 1
        ratio_h = target_h / mask_h if mask_h != 0 else 1
        # Compute new dimensions for the mask
        new_w = int(aligned_mask.shape[1] * ratio_w)
        new_h = int(aligned_mask.shape[0] * ratio_h)
        # Resize to new dimensions
        resized_mask = cv2.resize(aligned_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        # Crop or pad to match the target shape
        if resized_mask.shape[0] > target_shape[0] or resized_mask.shape[1] > target_shape[1]:
            resized_mask = center_crop(resized_mask, target_shape)
        elif resized_mask.shape[0] < target_shape[0] or resized_mask.shape[1] < target_shape[1]:
            resized_mask = center_pad(resized_mask, target_shape, pad_value)
        return resized_mask
    else:
        # If rectangles are not found, return the original mask
        return aligned_mask

#To get the orientation angle and bounding rectangle of a contour, so to 
#understand how to align the test mask to the base one
def get_orientation_angle_and_rectangle(contour):
    
    #Find the minimum area rectangle that bounds the contour, its 
    #orientation angle and center
    rect = cv2.minAreaRect(contour)
    (center_x, center_y), (width, height), angle = rect

    #The way cv2.minAreaRect works is that the angle 
    #is the angle of the rectangle's longer side, so we bring to a standard
    #format, independant of wether the rectangle is wider or taller.
    if width < height:
        main_axis_angle = angle
    else:
        main_axis_angle = angle + 90

    # Normalize angle to [-90, 90)
    main_axis_angle = main_axis_angle % 180
    if main_axis_angle >= 90:
        main_axis_angle -= 180

    # The center of the rectangle
    center = (int(center_x), int(center_y))
    return main_axis_angle, center, rect

def get_main_object_contour(contours, image_shape, area_thresh=0.9):
    
    #This returns the contour with the largest area that is below the threshold,
    #it's to avoid cases where the outer rectangle, which would be the whole image,
    #is considered the main object, and therefore have the largest area.
    img_area = image_shape[0] * image_shape[1]
    filtered = [c for c in contours if cv2.contourArea(c) < area_thresh * img_area]
    if not filtered:
        return None
    return max(filtered, key=cv2.contourArea)

def preprocess(image):
    # Extract contours and thresholded mask from the image.
    lightened = cv2.convertScaleAbs(image, alpha=1, beta=100)
    # If image is grayscale, convert to 3-channel for consistency
    if len(lightened.shape) == 2 or (len(lightened.shape) == 3 and lightened.shape[2] == 1):
        lightened = cv2.cvtColor(lightened, cv2.COLOR_GRAY2BGR)
    # Compute std for each channel and pick the one with the highest std
    stds = [np.std(lightened[:, :, i]) for i in range(3)]
    best_channel = np.argmax(stds)
    channel_img = lightened[:, :, best_channel]
    # Blurring and thresholding to get a binary mask
    blurred = cv2.GaussianBlur(channel_img, (31, 31), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    #Remove small noise
    kernel = np.ones((9, 9), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # Find its contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return image, contours, thresh

def align_image_to_angle(img, contours, target_angle, angle_and_center=None):
    # Align the image to a specific target angle and center the rectangle in the center of the image
    
    # Compute the main contour from the provided contours, maximal area 
    #excluding outer rectangle, which would be the whole image.
    main_contour = get_main_object_contour(contours, img.shape)
   
    #If the rectangle center and angle are provided, use them,
    #otherwise compute them from the main contour
    if angle_and_center is not None:
        angle, rect_center, rect = angle_and_center
    else:
        if main_contour is None or len(main_contour) < 5:
            return img, None, None
        angle, rect_center, rect = get_orientation_angle_and_rectangle(main_contour)
    
    # Rotate the image to align with the target angle
    #Compute the angle it need to be rotted to align with the target angle
    rotation_angle = angle - target_angle
    
    #Compute the center of the image
    (h, w) = img.shape[:2]
    img_center = (w // 2, h // 2)
    
    # Compute the rotation matrix
    M_rot = cv2.getRotationMatrix2D(rect_center, rotation_angle, 1.0)
   
    # Compute the translation to center the rectangle
    dx = img_center[0] - int(rect_center[0])
    dy = img_center[1] - int(rect_center[1])
    
    # Add  translation into the matrix
    M_rot[0, 2] += dx
    M_rot[1, 2] += dy

    # Apply the combined transformation
    aligned_img = cv2.warpAffine(img, M_rot, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    return aligned_img, rect, main_contour

def center_crop(img, target_shape):
    #Crop the image to the target shape, keeping the content centered.
    h, w = img.shape[:2]
    th, tw = target_shape
    y1 = max((h - th) // 2, 0)
    x1 = max((w - tw) // 2, 0)
    y2 = y1 + th
    x2 = x1 + tw
    return img[y1:y2, x1:x2]

def center_pad(img, target_shape, pad_value=0):
    #Pad the image to the target shape, keeping the content centered.
    h, w = img.shape[:2]
    th, tw = target_shape
    pad_vert = (th - h) // 2
    pad_horz = (tw - w) // 2
    # Only uses the precomputed pad_vert and pad_horz on one side because 
    # if the number is odd rounding will make it uneven
    padded = cv2.copyMakeBorder(
        img,
        pad_vert, th - h - pad_vert,
        pad_horz, tw - w - pad_horz,
        cv2.BORDER_CONSTANT, value=pad_value
    )
    return padded

def compare_and_plot_masks(base_img, test_img, show_plots=False):
    # Compare and plot masks from base and test images (images as arrays). 
    # The test mask is always adapted to the base mask, 
    # never the other way around. This ensures that the base mask
    # is always the reference for alignment and scaling.

    #To compute the execution time later
    start = time.time()

    #The base one is already imported as aligned
    aligned_base_thresh=base_img
    
    # Preprocess both images, the base just to get the contours
    _, base_contours, _ = preprocess(base_img)
    test_img_proc, test_contours, test_thresh = preprocess(test_img)

    # Compute the tilting angles of both masks
    base_angle, _, _ = get_orientation_angle_and_rectangle(get_main_object_contour(base_contours, aligned_base_thresh.shape))
    test_angle, test_center, test_rect = get_orientation_angle_and_rectangle(get_main_object_contour(test_contours, test_thresh.shape))

    # Align the test mask to the base mask's orientation and center, using precomputed angle and center
    target_angle = base_angle  # Always align test to base
    aligned_test_thresh, test_rect, _ = align_image_to_angle(test_thresh, test_contours, target_angle, (test_angle, test_center, test_rect))

    # Use main object rectangles for scaling (adapt test to base)
    base_rect = cv2.minAreaRect(get_main_object_contour(base_contours, aligned_base_thresh.shape))
    # Rescale and resize the test mask to match the base mask's rectangle and shape
    target_shape = aligned_base_thresh.shape[:2]
    aligned_test_thresh = rescale_and_resize_mask(aligned_test_thresh, test_rect, base_rect, target_shape)
    
    
    
    # Ensure the masks have the same size, or absdiff will fail
    #It is needed sometimes, probably due to rounding
    if aligned_base_thresh.shape != aligned_test_thresh.shape:
        target_shape = aligned_base_thresh.shape[:2]
        aligned_test_thresh = center_crop(aligned_test_thresh, target_shape)
        # If test mask is still smaller, pad it
        pad_vert = max(target_shape[0] - aligned_test_thresh.shape[0], 0)
        pad_horz = max(target_shape[1] - aligned_test_thresh.shape[1], 0)
        if pad_vert > 0 or pad_horz > 0:
            aligned_test_thresh = cv2.copyMakeBorder(
                aligned_test_thresh,
                pad_vert // 2, pad_vert - pad_vert // 2,
                pad_horz // 2, pad_horz - pad_horz // 2,
                cv2.BORDER_CONSTANT, value=255
            )
        # If test mask is too big, crop again
        if aligned_test_thresh.shape != target_shape:
            aligned_test_thresh = center_crop(aligned_test_thresh, target_shape)


    # Compute absolute difference between the two masks
    diff_mask = cv2.absdiff(aligned_base_thresh, aligned_test_thresh)

    #clean up small noise
    kernel = np.ones((11,11), np.uint8)
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)

    ## Print execution time for the comparison
    #print(f"Execution time of compare masks: {time.time() - start:.2f} seconds")

    if show_plots:
        # Plot the aligned base mask, aligned test mask, and difference mask
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(aligned_base_thresh, cmap='gray')
        plt.title("Aligned Base Mask")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(aligned_test_thresh, cmap='gray')
        plt.title("Aligned Test Mask (Adapted)")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(diff_mask, cmap='gray')
        plt.title("Difference Mask")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    # Overlay the two masks to visualize differences
    overlay = np.zeros((aligned_base_thresh.shape[0], aligned_base_thresh.shape[1], 3), dtype=np.uint8)

    # Red for base mask, green for test mask, yellow for overlap
    overlay[(aligned_base_thresh == 0) & (aligned_test_thresh > 0)] = [255, 0, 0]  # Red: Base mask only
    overlay[(aligned_test_thresh == 0) & (aligned_base_thresh > 0)] = [0, 255, 0]  # Green: Test mask only
    overlay[(aligned_base_thresh == aligned_test_thresh)] = [255, 255, 0]  # Yellow: Overlap

    if show_plots:
        # Plot the overlay
        plt.figure(figsize=(7, 7))
        plt.imshow(overlay)
        plt.title("Overlay of Base and Test Masks")
        plt.axis('off')
        plt.show()

    return aligned_test_thresh, aligned_base_thresh, diff_mask, overlay


if __name__ == "__main__":
    scorre_path, base_shape_path, base_print_path, recomposed_path= paths.define_files("parmareggio_ok", project_root)  # Paths to the base and test images


    base = cv2.imread(base_shape_path,cv2.IMREAD_GRAYSCALE)
    test = cv2.imread(recomposed_path)
    compare_and_plot_masks(base, test, show_plots=True)