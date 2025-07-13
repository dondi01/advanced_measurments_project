import cv2
import numpy as np
import extract_print_features as epf
import Treshold_compare_masks as tcm
import paths
import matplotlib.pyplot as plt
from pathlib import Path
import functions_th as th

#Extract the print from the test image using the test mask,
#and then compares it with the oen precomputed from the base mask.
def compare_prints_with_masks(base_mask, test_img, test_mask, show_plots=True):
    if test_mask.shape != base_mask.shape:    
        test_mask=th.match_size(base_mask, test_mask, pad_value=0)
    # Extract the print from the test image using the test mask
    test_print = epf.extract_print(test_mask, test_img, show_plots=False)
    import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 10))
    # plt.imshow(test_print, cmap='gray')
    # plt.title("Test Print Extracted")
    # plt.axis('off')
    # plt.show()

    # Dilate both masks to allow for tolerance in matching
    #kernel = np.ones((21, 21), np.uint8)
    kernel= np.ones((31, 31), np.uint8)  # Smaller kernel for less aggressive dilation
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
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 7))
        plt.imshow(overlay)
        plt.title("Fuzzy Edge Match Overlay (Base vs Test Print)")
        plt.axis('off')
        # Add legend for colors
        red_patch = mpatches.Patch(color='red', label='Missed')
        green_patch = mpatches.Patch(color='green', label='Extra')
        yellow_patch = mpatches.Patch(color='yellow', label='Matched')
        plt.legend(handles=[red_patch, green_patch, yellow_patch], loc='lower right')
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

if __name__ == "__main__":
    # Define paths to the base and test images
    project_root = Path(__file__).parent
    scorre_path, base_shape_path, base_print_path, recomposed_path = paths.define_files("green_buco_in_piu", project_root)
    
    # Load the base mask and test image
    base_mask = cv2.imread(base_print_path, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.imread(recomposed_path)
    base_shape= cv2.imread(base_shape_path, cv2.IMREAD_GRAYSCALE)
    test_img= th.rescale_and_resize_mask(aligned_mask=test_img, target_img=base_shape, pad_value=0)
    # Load the test mask
    _,test_mask=th.preprocess(test_img)

    # Compare prints with masks
    diff_fuzzy = compare_prints_with_masks(base_mask, test_img, test_mask, show_plots=True)

    # --- Plot scratches overlay on initial image ---
    from skimage.morphology import skeletonize
    from skimage.measure import label, regionprops
    scratches = skeletonize(diff_fuzzy > 0)
    labeled = label(scratches)
    min_length = 50
    filtered_skeleton = np.zeros_like(scratches, dtype=np.uint8)
    for region in regionprops(labeled):
        if region.area >= min_length:
            for coord in region.coords:
                filtered_skeleton[coord[0], coord[1]] = 1
    lines_to_plot = [(coord[1], coord[0]) for region in regionprops(label(filtered_skeleton)) if region.area >= min_length for coord in region.coords]
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    if lines_to_plot:
        xs, ys = zip(*lines_to_plot)
        plt.scatter(xs, ys, color='yellow', s=1, label='Scratch')
    plt.title("Scratches Overlay on Initial Image (Skeletonized)")
    plt.axis('off')
    plt.tight_layout(pad=5)
    plt.legend()
    plt.show()