import cv2
import numpy as np
import matplotlib.pyplot as plt
def crop_to_main_object(img, margin=10, area_thresh=0.99):
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
        print("No valid contours found, returning original image.")
        return img  # fallback: return original
    main_contour = max(filtered, key=cv2.contourArea)
    # Get min area rectangle
    rect = cv2.minAreaRect(main_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # Get bounding rect with margin
    x, y, w, h = cv2.boundingRect(box)
    x = max(x - margin, 0)
    y = max(y - margin, 0)
    w = min(w + 2 * margin, img.shape[1] - x)
    h = min(h + 2 * margin, img.shape[0] - y)
    cropped = img[y:y+h, x:x+w]
    #cropped=cv2.drawContours(cropped, main_contour, -1, (0, 255, 0), 2)  # Draw rectangle for visualization
    return cropped

if __name__ == "__main__":
    # Example usage
    res=crop_to_main_object(cv2.imread("C:/Users/franc/Desktop/Scuola/Measurement/advanced_measurments_project/Reconstructed/Nappies_ok.png"))
    plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    plt.axis('off') 
    plt.title("Cropped Main Object")
    plt.axis('off')
    plt.show()