import cv2
import numpy as np
import extract_print_features as epf
import Treshold_compare_masks as tcm


#Extract the print from the test image using the test mask,
#and then compares it with the oen precomputed from the base mask.
def compare_prints_with_masks(base_mask, test_img, test_mask, show_plots=True):

    # Extract the print from the test image using the test mask
    test_print = epf.extract_print(test_mask, test_img, show_plots=False)
    
    # Ensure test_print is the same size as base_mask 
    if base_mask.shape != test_print.shape:
        # Adding [:2] to get only the height and width,
        # should not be necessary as only color images have
        # 3 channels but just in case
        target_shape = base_mask.shape[:2]
        # Crop if too big, pad if too small, using the same logic as in Treshold_compare_masks
        if test_print.shape[0] > target_shape[0] or test_print.shape[1] > target_shape[1]:
            # If test ended up bigger (had to zoom in), crop it
            test_print = tcm.center_crop(test_print, target_shape)
        elif test_print.shape[0] < target_shape[0] or test_print.shape[1] < target_shape[1]:
            # If it ended up smaller (had to zoom out), pad it with black stripes
            test_print = tcm.center_pad(test_print, target_shape, pad_value=0)
        
        # If test_print is still not the right size (due to rounding), crop again.
        # This is a safeguard, usually not needed but can happen with some images,
        # but it is good to have it just in case.
        if test_print.shape != target_shape:
            test_print = tcm.center_crop(test_print, target_shape)
            
    # Dilate both masks to allow for tolerance in matching
    kernel = np.ones((15, 15), np.uint8)
    dil_base = cv2.dilate(base_mask, kernel, iterations=1)
    dil_test = cv2.dilate(test_print, kernel, iterations=1)

    #By comparing the original ones to the dilated ones,
    #we can find the differences between the two prints, with some tolerance.
    #It's called fuzzy edge matching.

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
    #The difference mask is formed by pixels that are either missed or extra
    #Uses | because it's two arrays of the same shape, or does not work here
    diff_fuzzy[missed | extra] = 1
    #Then blurred to reduce noise
    diff_fuzzy = cv2.GaussianBlur(diff_fuzzy.astype(np.float32), (5, 5), 0)
    if show_plots:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 7))
        plt.imshow(diff_fuzzy, cmap='grey')
        plt.title("Not Matched Pixel Density (Neighborhood Count)")
        plt.axis('off')
        plt.show()
    return diff_fuzzy
