import os
import cv2
import numpy as np

# Set your root directory here
root_dir = r"C:/Users/franc/Desktop/Scuola/Measurement/Carton Defect Dataset"
output_root = root_dir + '_cut'
os.makedirs(output_root, exist_ok=True)

def process_image(img, mask_debug_path=None, thresh_debug_path=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    # Adaptive thresholding for uneven lighting
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 51, 6)
    if thresh_debug_path is not None:
        cv2.imwrite(thresh_debug_path, thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback: Canny edge detection + contour filling
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
    carton_contour = max(contours, key=cv2.contourArea)
    #print(f"Largest contour area: {cv2.contourArea(carton_contour)}")
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [carton_contour], -1, 255, thickness=-1)
    if mask_debug_path is not None:
        cv2.imwrite(mask_debug_path, mask)
    mask_3ch = cv2.merge([mask, mask, mask])
    uniform_color = np.full_like(img, 200)
    result = np.where(mask_3ch == 255, img, uniform_color)
    return result

for dirpath, dirnames, filenames in os.walk(root_dir):
    rel_dir = os.path.relpath(dirpath, root_dir)
    out_dir = os.path.join(output_root, rel_dir).replace('\\', '/')
    os.makedirs(out_dir, exist_ok=True)
    mask_debug_saved = False
    for fname in filenames:
        if fname.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):
            fpath = os.path.join(dirpath, fname).replace('\\', '/')
            img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Could not read {fpath}, skipping.")
                continue
            # If image is grayscale (2D or 3D with 1 channel), convert to BGR
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # Resize to fit within 3216x1248, maintaining aspect ratio
            target_w, target_h = 3216, 1248
            h, w = img.shape[:2]
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask_debug_path = os.path.join(out_dir, 'mask_debug.png').replace('\\', '/') if not mask_debug_saved else None
            thresh_debug_path = os.path.join(out_dir, 'thresh_debug.png').replace('\\', '/') if not mask_debug_saved else None
            result = process_image(img, mask_debug_path=mask_debug_path, thresh_debug_path=thresh_debug_path)
            result=cv2.GaussianBlur(result, (21, 21), 0)  
            if not mask_debug_saved and mask_debug_path is not None:
                mask_debug_saved = True
            if result is None:
                print(f"No carton found in {fpath}, skipping.")
                continue
            out_path = os.path.join(out_dir, fname).replace('\\', '/')
            cv2.imwrite(out_path, result)
            print(f"Saved to {out_path}")

print("Processing complete.")