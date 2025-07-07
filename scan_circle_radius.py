import os
import cv2
import numpy as np

# Set your root directory here
root_dir = r"C:/Users/franc/Desktop/Scuola/Measurement/Carton Defect Dataset"

def detect_circle_radius(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    blurred = cv2.medianBlur(gray, 11)
    h, w = gray.shape
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=h//4,
                               param1=50, param2=30, minRadius=h//6, maxRadius=h//2)
    center = (w // 2, h // 2)
    best_circle = None
    min_dist = float('inf')
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            dist = np.hypot(c[0] - center[0], c[1] - center[1])
            if dist < min_dist or (dist == min_dist and (best_circle is None or c[2] > best_circle[2])):
                min_dist = dist
                best_circle = c
    if best_circle is not None and min_dist < min(h, w) * 0.2:
        return best_circle[2]
    else:
        return min(h, w) // 2 - 10

def scan_all_images_for_radius(root_dir):
    radii = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                fpath = os.path.join(dirpath, fname)
                img = cv2.imread(fpath)
                if img is None:
                    continue
                r = detect_circle_radius(img)
                radii.append(r)
                print(f"Image: {fpath}, Detected radius: {r:.2f}")
    if radii:
        print(f"Average radius: {np.mean(radii):.2f}")
        print(f"Median radius: {np.median(radii):.2f}")
        print(f"Min radius: {np.min(radii):.2f}")
        print(f"Max radius: {np.max(radii):.2f}")
    else:
        print("No images found or no circles detected.")

if __name__ == "__main__":
    scan_all_images_for_radius(root_dir)
