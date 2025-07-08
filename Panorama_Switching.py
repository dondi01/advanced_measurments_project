import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import scipy.io
import re
import time
from pathlib import Path
import paths
import functions_th as th

class reassembler:
    "This class is used to reassemble a set of pictures into a panorama"
    def __init__(self):         
        self.orb= cv2.ORB_create(200)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.project_root = Path(__file__).resolve().parent
        self.clahe = cv2.createCLAHE(clipLimit=1.75, tileGridSize=(20, 20))
        self.save_path = None
        dummy = np.zeros((32, 32), dtype=np.uint8)
        self.orb.detectAndCompute(dummy, None)

    def alphanum_key(self,s):
        def tryint(s):
            try:
                return int(s)
            except:
                return s
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    def apply_feather(self,feather, warped, panorama, C) -> np.ndarray:
        for c in range(C):
            panorama[..., c] += warped[..., c] * feather
        return panorama

    def crop_to_main_object(self,img, margin=0, area_thresh=0.99):
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

    def estimateTransformFromFeatures(self, kp_prev, des_prev, kp_curr, des_curr):
        matches = self.bf.match(des_prev, des_curr)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:10]
        src_pts = np.float32([kp_prev[m.queryIdx].pt for m in good_matches]).reshape(-1,2)
        dst_pts = np.float32([kp_curr[m.trainIdx].pt for m in good_matches]).reshape(-1,2)
        M, inliers = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC)
        M33 = np.eye(3, dtype=np.float32)
        M33[:2, :3] = M
        return M33
    
    def findBorders(self, transforms, frames):
        all_corners = []
        for i, img in enumerate(frames):
            H_i, W_i = img.shape[:2]
            corners_i = np.array([
                [0, 0, 1],
                [W_i, 0, 1],
                [W_i, H_i, 1],
                [0, H_i, 1]
            ], dtype=np.float32)
            warped = (transforms[i] @ corners_i.T).T
            all_corners.append(warped[:, :2])
        all_corners = np.vstack(all_corners)
        x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)
        panorama_width = x_max - x_min
        panorama_height = y_max - y_min
        if panorama_width > 10000 or panorama_height > 10000:
            zero_panorama = True
            threshold_coefficient = 0.75
            while zero_panorama:
                print(f"Panorama size is too large {panorama_width}x{panorama_height}, filtering outliers.")
                median = np.median(all_corners, axis=0)
                distances = np.linalg.norm(all_corners - median, axis=1)
                threshold = threshold_coefficient * np.median(np.abs(distances - np.median(distances)))
                filtered_corners = all_corners[distances < threshold]
                if filtered_corners.size == 0:
                    x_min, y_min = 0, 0
                    x_max, y_max = 0, 0
                else:
                    x_min, y_min = np.floor(filtered_corners.min(axis=0)).astype(int)
                    x_max, y_max = np.ceil(filtered_corners.max(axis=0)).astype(int)
                panorama_width = x_max - x_min
                panorama_height = y_max - y_min
                if panorama_width > 1500 and panorama_height > 1500:
                    zero_panorama = False
                elif threshold_coefficient > 20.0:
                    print("Threshold coefficient is too high, stopping filtering.")
                    zero_panorama = False
                else:
                    threshold_coefficient = threshold_coefficient + 0.25
                
        return panorama_width, panorama_height,x_min, y_min

    def finish_panorama(self, panorama, weight):
        weight[weight == 0] = 1
        panorama /= weight[..., None]
        panorama = np.clip(panorama, 0, 255).astype(np.uint8)
        gray_panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero((gray_panorama > 0).astype(np.uint8))
        x, y, w, h = cv2.boundingRect(coords)
        panorama_cropped = panorama[y:y+h, x:x+w]
        res = cv2.normalize(panorama_cropped, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # --- Mask outside main carton contour ---
        res=self.set_background_to_uniform_color(res, color=(200, 200, 200))
        res=self.crop_to_main_object(res,margin=50, area_thresh=0.99)
        res=th.align_image_to_least_rotation(res)
        return res

    def set_background_to_uniform_color(self, img, color=(200, 200, 200)):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Find the largest contour (main carton)
            main_contour = th.get_main_object_contour(contours, gray.shape)
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [main_contour], -1, color, thickness=cv2.FILLED)
            img[mask == 0] = 0
        return img
    
    # Main panorama pipeline, takes frames, orb, bf, camera_matrix, dist_coeffs as input
    def run_panorama_pipeline(self,frames,show_plots=True, save_path=None):
        frames = frames[::2]
        frames = [cv2.undistort(f, self.camera_matrix, self.dist_coeffs) for f in frames]
        frames = [self.crop_to_main_object(f, margin=100, area_thresh=0.99) for f in frames]
        H, W, C = frames[0].shape
       
        transforms = [np.eye(3, dtype=np.float32)]
        gray_prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        kp_prev, des_prev = self.orb.detectAndCompute(gray_prev, None)
        
        for i in range(1, len(frames)):
            img_curr = frames[i]
            gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)
            kp_curr, des_curr = self.orb.detectAndCompute(gray_curr, None)
            if des_prev is None or des_curr is None or len(kp_prev) < 4 or len(kp_curr) < 4:
                print(f"Skipping frame {i} due to insufficient features.")
                transforms.append(transforms[-1].copy())
                kp_prev, des_prev = kp_curr, des_curr
                continue
            M33=self.estimateTransformFromFeatures(kp_prev, des_prev, kp_curr, des_curr) 
            cumulative = transforms[-1] @ M33
            transforms.append(cumulative)
            kp_prev, des_prev = kp_curr, des_curr
        
        panorama_width,panorama_height,x_min,y_min = self.findBorders(transforms, frames)

        panorama = np.zeros((panorama_height, panorama_width, C), dtype=np.float32)
        weight = np.zeros((panorama_height, panorama_width), dtype=np.float32)

        for i, img in enumerate(frames):
            offset_M = transforms[i].copy()
            offset_M[0,2] -= x_min
            offset_M[1,2] -= y_min
            img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            img_ycrcb[:, :, 0] = self.clahe.apply(img_ycrcb[:, :, 0])
            img = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)
            warped = cv2.warpAffine(img, offset_M[:2], (panorama_width, panorama_height))
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = cv2.warpAffine((gray_img > 20).astype(np.float32), offset_M[:2], (panorama_width, panorama_height))
            small_mask = cv2.resize(mask, (mask.shape[1] // 8, mask.shape[0] // 8), interpolation=cv2.INTER_NEAREST)
            small_blur = cv2.GaussianBlur(small_mask, (21, 21), 0)
            feather = cv2.resize(small_blur, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            feather=feather**4
            feather = np.clip(feather, 1e-3, 1.0)
            panorama = self.apply_feather(feather, warped, panorama, C)
            weight += feather
        
        res=self.finish_panorama(panorama, weight)

        # --- End mask logic ---
        if save_path is not None:
            cv2.imwrite(save_path, res)
        if show_plots:
            plt.figure(figsize=(15, 8))
            plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        
        return res

# If run as a script, preserve original behavior
if __name__ == "__main__":
    p_s=reassembler()
    filepath, base_shape, _, recomposed_path = paths.define_files("parmareggio_ok",p_s.project_root)
    mat = scipy.io.loadmat(p_s.project_root / 'dataset_medi' / 'TARATURA' / 'medium_dataset_taratura.mat')
    camera_matrix = mat['K']
    dist_coeffs = mat['dist']
    image_files = sorted(
        glob.glob(filepath),
        key=p_s.alphanum_key)
    frames = [cv2.imread(f, cv2.IMREAD_COLOR) for f in image_files]
    frames = [f for f in frames if f is not None and f.shape == frames[0].shape]
    p_s.camera_matrix = camera_matrix
    p_s.dist_coeffs = dist_coeffs
    start = time.time()
    res = p_s.run_panorama_pipeline(frames,save_path=recomposed_path,show_plots=False)
    print("Execution time is:", time.time() - start)