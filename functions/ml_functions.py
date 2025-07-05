import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf
import re
import os
from pathlib import Path


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



# Show all windows on the image
def generate_windowed_dataset(input_path, window_size, stride, output_path):
    i = 0
    for image in Path(input_path).iterdir():
        image = cv2.imread(str(image))
        image = preprocess_image(image, window_size, stride)
        generate_windows(i, image, window_size, stride, output_path)
        i = i + 1



def parse_filename_py(file_path):
    #filename = file_path.split('\\')[-1] # path has alreaby been converted to string
    filename = os.path.basename(file_path)
    pattern = r'\d+_window_(\d+)_at_(\d+)_(\d+)\.png'
    match = re.match(pattern, filename)
    if match:
        return [int(match.group(1)), int(match.group(2)), int(match.group(3))]
    else:
        print(f"WARNING: Filename {filename} does not match the expected pattern.")
        return [0, 0, 0]

def parse_filename(file_path):
    return tf.py_function(parse_filename_py, [file_path], [tf.int32, tf.int32, tf.int32])



def process_path_py(file_path, image_size):
    width, height = tuple(image_size[:2].numpy()) # we need to do this because image_size is a tensor
    file_path = file_path.numpy().decode('utf-8')
    img = cv2.imread(file_path)
    img = cv2.resize(img, (width, height))  # Resize image to the specified size
    str_label = file_path.split(os.path.sep)[-2]
    if str_label == 'healthy':
        label = 0
    else:
        label = 1
    i, x, y = parse_filename_py(file_path)
    return img, label, i, x, y

def process_path(file_path, image_size):
    img, label, i, x, y = tf.py_function(
        process_path_py,
        [file_path, image_size],
        [tf.uint8, tf.int32, tf.int32, tf.int32, tf.int32]
    )
    #img.set_shape(image_size)  # Set the shape of the image if known
    img.set_shape((image_size[0], image_size[1], 3))  # (height, width, channels)
    label.set_shape(())  # Scalar string tensor
    i.set_shape(())
    x.set_shape(())
    y.set_shape(())
    metadata = (i, x, y)
    return img, label, metadata

def get_training_validation_datasets(input_path, batch_size, image_size):
    list_ds = tf.data.Dataset.list_files(str(Path(input_path) / '**' / '*.png'), shuffle=True)
    num_files = len(list(list_ds.as_numpy_iterator()))
    train_size = int(num_files * 0.8)
    validation_size = num_files - train_size

    extracted_ds = list_ds.map(lambda file_path: process_path(file_path, image_size))
    
    training_dataset = extracted_ds.take(train_size).batch(batch_size)
    validation_dataset = extracted_ds.skip(train_size).take(validation_size).batch(batch_size)

    #debug printing
    for batch in validation_dataset.take(1):
        print("Batch content:")
        print(batch)
    return training_dataset, validation_dataset



def get_original_dimensions(unprocessed_images_path): # returns original dimensions (prior to windowing) for mask reassembly
    original_dimensions = []
    for image_path in unprocessed_images_path.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            img = cv2.imread(str(image_path))
            if img is not None:
                original_dimensions.append(img.shape[:2])
    return original_dimensions