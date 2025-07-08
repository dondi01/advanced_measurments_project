import cv2
import numpy as np
import matplotlib.pyplot as plt


def crop_to_main_object(img, margin=0, area_thresh=0.99):
    """
    Finds the largest contour (excluding the one that covers the whole image),
    fits a minAreaRect, and crops the image to that rectangle with a margin.
    Args:
        img: Input image (BGR or grayscale).
        margin: Margin (in pixels) to add around the detected rectangle.
        area_thresh: Fraction of image area above which a contour is considered 'background'.
    Returns:
        Cropped image containing the main object with margin.
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    # Threshold to get binary mask
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_area = img.shape[0] * img.shape[1]
    # Filter out contours that are too large (likely the border)
    filtered = [c for c in contours if cv2.contourArea(c) < area_thresh * img_area]
    if not filtered:
        return img  # fallback: return original
    main_contour = max(filtered, key=cv2.contourArea)
    # Get min area rectangle
    rect = cv2.minAreaRect(main_contour)
    box = cv2.boxPoints(rect)
    box = box.astype(np.int32)
    # Get bounding rect with margin
    x, y, w, h = cv2.boundingRect(box)
    x = max(x - margin, 0)
    y = max(y - margin, 0)
    w = min(w + 2 * margin, img.shape[1] - x)
    h = min(h + 2 * margin, img.shape[0] - y)
    cropped = img[y:y+h, x:x+w]
    return cropped



def find_bounding_rectangle(img):
    contours=preprocess(img)[0]
    main_contour=get_main_object_contour(contours, img.shape)
    return cv2.minAreaRect(main_contour)

def center_crop(img, target_shape):
    #Crop the image to the target shape, keeping the content centered.
    h, w = img.shape[:2]
    th, tw = target_shape[:2]
    y1 = max((h - th) // 2, 0)
    x1 = max((w - tw) // 2, 0)
    y2 = y1 + th
    x2 = x1 + tw
    return img[y1:y2, x1:x2]

def center_pad(img, target_shape, pad_value=0):
    #Pad the image to the target shape, keeping the content centered.
    
    h, w = img.shape[:2]
    th, tw = target_shape[:2]
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

def match_size(base,test,pad_value=0):
    target_shape = base.shape
    aligned_test_thresh = center_crop(test, target_shape)
    # If test mask is still smaller, pad it
    pad_vert = max(target_shape[0] - aligned_test_thresh.shape[0], 0)
    pad_horz = max(target_shape[1] - aligned_test_thresh.shape[1], 0)
    if pad_vert > 0 or pad_horz > 0:
        aligned_test_thresh = center_pad(aligned_test_thresh, target_shape, pad_value)
    # If test mask is too big, crop again
    if aligned_test_thresh.shape != target_shape:
        aligned_test_thresh = center_crop(aligned_test_thresh, target_shape)
    return aligned_test_thresh


def rescale_and_resize_mask(aligned_mask, mask_rect=None, target_rect=None, target_img=None,pad_value=255):
    # Rescale and resize a mask so that its main object's rectangle matches the target rectangle's size,
    # then crop or pad to the target shape. Used for robust geometric normalization.
    target_shape = target_img.shape[:2] if target_img is not None else aligned_mask.shape[:2]
    mask_rect= find_bounding_rectangle(aligned_mask) if mask_rect is None else mask_rect
    target_rect = find_bounding_rectangle(target_img) if target_rect is None else target_rect
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
    return contours, thresh

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

def get_main_object_contour(contours, image_shape, area_thresh=0.99):
    
    #This returns the contour with the largest area that is below the threshold,
    #it's to avoid cases where the outer rectangle, which would be the whole image,
    #is considered the main object, and therefore have the largest area.
    img_area = image_shape[0] * image_shape[1]
    filtered = [c for c in contours if cv2.contourArea(c) < area_thresh * img_area]
    if not filtered:
        return None
    return max(filtered, key=cv2.contourArea)


def align_image_to_angle(img, target_angle,contours=None, angle_and_center=None):
    # Align the image to a specific target angle and center the rectangle in the center of the image
    
    # Compute the main contour from the provided contours, maximal area 
    #excluding outer rectangle, which would be the whole image.
    if contours is None:
        contours, _ = preprocess(img)
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


def align_image_to_least_rotation(img, contours=None):
    if contours==None:
        gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contours= cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    main_contour = get_main_object_contour(contours, img.shape)
    if main_contour is None or len(main_contour) < 5:
        return img, None, None

    # Get the orientation angle and bounding rectangle
    angle, rect_center, rect = get_orientation_angle_and_rectangle(main_contour)

    # Define possible axes to align to
    possible_axes = [0, 90, -90]

    # Find the axis that requires the least rotation
    best_axis = min(possible_axes, key=lambda axis: abs(angle - axis))
    rotation_angle = angle - best_axis

    # Compute the rotation matrix
    M_rot = cv2.getRotationMatrix2D(rect_center, rotation_angle, 1.0)
    (h, w) = img.shape[:2]
    rotated_img = cv2.warpAffine(img, M_rot, (w, h), flags=cv2.INTER_NEAREST)

    # Translate the rectangle to the center of the image
    img_center = (w // 2, h // 2)
    dx = img_center[0] - int(rect_center[0])
    dy = img_center[1] - int(rect_center[1])
    M_trans = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned_img = cv2.warpAffine(rotated_img, M_trans, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

    return aligned_img