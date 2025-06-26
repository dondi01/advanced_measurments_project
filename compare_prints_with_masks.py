import cv2
import numpy as np
import extract_print_features as epf

def compare_prints_with_masks(base_mask, test_img, test_mask, show_plots=True):
    """
    Compare a precomputed base print mask with a test carton print extracted from a test image and its mask.
    Arguments:
        base_mask: np.ndarray, binary mask of the base carton print (precomputed)
        test_img: np.ndarray, image of the test carton
        test_mask: np.ndarray, binary mask of the test carton
        show_plots: bool, whether to show diagnostic plots
    Returns:
        diff_fuzzy: np.ndarray, fuzzy difference mask between base and test prints
    """
    # Extract the print from the test image using the test mask
    test_print = epf.extract_print(test_mask, test_img, show_plots=False)
    # Ensure test_print is the same size as base_mask (never modify base_mask)
    if base_mask.shape != test_print.shape:
        target_shape = base_mask.shape[:2]
        # Crop or pad test_print to match base_mask
        def center_crop(img, target_shape):
            h, w = img.shape[:2]
            th, tw = target_shape
            y1 = max((h - th) // 2, 0)
            x1 = max((w - tw) // 2, 0)
            y2 = y1 + th
            x2 = x1 + tw
            return img[y1:y2, x1:x2]
        test_print = center_crop(test_print, target_shape)
        pad_vert = max(target_shape[0] - test_print.shape[0], 0)
        pad_horz = max(target_shape[1] - test_print.shape[1], 0)
        if pad_vert > 0 or pad_horz > 0:
            test_print = cv2.copyMakeBorder(
                test_print,
                pad_vert // 2, pad_vert - pad_vert // 2,
                pad_horz // 2, pad_horz - pad_horz // 2,
                cv2.BORDER_CONSTANT, value=0
            )
        # If test_print is still too big, crop again
        if test_print.shape != target_shape:
            test_print = center_crop(test_print, target_shape)
    # Dilate both masks to allow for tolerance in matching
    kernel = np.ones((15, 15), np.uint8)
    dil_base = cv2.dilate(base_mask, kernel, iterations=1)
    dil_test = cv2.dilate(test_print, kernel, iterations=1)
    # Missed: base print not matched by test print
    missed = (base_mask > 0) & (dil_test == 0)
    # Extra: test print not matched by base print
    extra = (test_print > 0) & (dil_base == 0)
    # Matched: print in base or test, and found in the other's band
    matched = ((base_mask > 0) & (dil_test > 0)) | ((test_print > 0) & (dil_base > 0))
    if show_plots:
        overlay = np.zeros((base_mask.shape[0], base_mask.shape[1], 3), dtype=np.uint8)
        overlay[missed] = [255, 0, 0]
        overlay[extra] = [0, 255, 0]
        overlay[matched] = [255, 255, 0]
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 7))
        plt.imshow(overlay)
        plt.title("Fuzzy Edge Match Overlay (Base vs Test Print)")
        plt.axis('off')
        plt.show()
    # Fuzzy diff mask: missed or extra
    diff_fuzzy = np.zeros_like(base_mask)
    diff_fuzzy[missed | extra] = 1
    diff_fuzzy = cv2.GaussianBlur(diff_fuzzy.astype(np.float32), (5, 5), 0)
    if show_plots:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 7))
        plt.imshow(diff_fuzzy, cmap='grey')
        plt.title("Not Matched Pixel Density (Neighborhood Count)")
        plt.axis('off')
        plt.show()
    return diff_fuzzy
