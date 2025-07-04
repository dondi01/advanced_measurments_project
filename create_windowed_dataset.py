import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
project_root = Path(__file__).resolve().parent

faulty_source_path = str( project_root / 'ml_datasets' / 'carton_baseline' / 'faulty')
healthy_source_path = str( project_root / 'ml_datasets' / 'carton_baseline' / 'healthy')
faulty_output_path = str( project_root / 'ml_datasets' / 'carton_windowed' / 'faulty')
healthy_output_path = str( project_root / 'ml_datasets' / 'carton_windowed' / 'healthy')

def center_crop(img, dim):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img

def scale_image(img, factor=1):
	"""Returns resize image by scale factor.
	This helps to retain resolution ratio while resizing.
	Args:
	img: image to be scaled
	factor: scale factor to resize
	"""
	return cv2.resize(img,(int(img.shape[1]*factor), int(img.shape[0]*factor)))


def sliding_window(image, window_size, stride):
    H, W = image.shape[:2]
    window_h, window_w = window_size
    for y in range(0, H - window_h + 1, stride):
        for x in range(0, W - window_w + 1, stride):
            window = image[y:y+window_h, x:x+window_w]
            yield (x, y, window)

def pad_image_for_sliding_window(image, window_size, stride):
    H, W = image.shape[:2]
    window_h, window_w = window_size

    # Calculate needed padding
    pad_h = (-(H - window_h) % stride) if H > window_h else window_h - H
    pad_w = (-(W - window_w) % stride) if W > window_w else window_w - W

    # Pad bottom and right sides
    padded_image = cv2.copyMakeBorder(
        image,
        top=0, bottom=pad_h,
        left=0, right=pad_w,
        borderType=cv2.BORDER_CONSTANT,
        value = 0
    )
    return padded_image


def generate_windows(image_index, image, window_size, stride, output_path):
    for i, (x, y, window) in enumerate(sliding_window(image, window_size, stride)):
        window_h, window_w = window_size
        cropped_image = image[x:x + window_w, y:y + window_h] # Slicing to crop the image
        final_output_path = str(Path(output_path)/ str(f'{image_index}_window_{i}_at_{x}_{y}.png'))
        cv2.imwrite(final_output_path, cropped_image)

def preprocess_image(image, window_size, stride): #used for training with "carton dataset"
    dim = (1550, 1550)
    image = center_crop(image, dim) # Removes vignette from dataset and useless pans
    image = scale_image(image, 0.5)  # Scale down to half size for less computational requirements
    image = pad_image_for_sliding_window(image, window_size, stride)  # Pad the image for sliding window
    return image


window_size = (512, 512)  # Size of the sliding window
stride = 250  # Step size for the sliding window
# Show all windows on the image
def generate_windowed_dataset(input_path, window_size, stride, output_path):
    i = 0
    for image in Path(input_path).iterdir():
        image = cv2.imread(str(image))
        image = preprocess_image(image, window_size, stride)
        generate_windows(i, image, window_size, stride, output_path)
        i = i + 1




generate_windowed_dataset(faulty_source_path, window_size, stride, faulty_output_path)
generate_windowed_dataset(healthy_source_path, window_size, stride, healthy_output_path)


#DEBUG OPTIONS

#test_img = cv2.imread(str(project_root / 'ml_datasets' / 'carton_baseline'/ 'faulty' / '2022-06-22_21_36_28_051.bmp'))
#test_img = preprocess_image(test_img, window_size, stride)
#generate_windows(0, test_img, window_size, stride, faulty_output_path)