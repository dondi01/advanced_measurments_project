import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import re
import os
from pathlib import Path
import datetime
import shutil

image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']

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

def center_crop_folder(input_folder, output_folder, crop_dim):
    """
    Applies center_crop to all images in input_folder and saves them to output_folder.
    Args:
        input_folder (str or Path): Path to the folder with input images.
        output_folder (str or Path): Path to the folder to save cropped images.
        crop_dim (tuple): (width, height) for cropping.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for file_path in input_folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            img = cv2.imread(str(file_path))
            if img is None:
                print(f"WARNING: Could not read {file_path}")
                continue
            cropped = center_crop(img, crop_dim)
            out_path = output_folder / file_path.name
            cv2.imwrite(str(out_path), cropped)
            print(f"Cropped and saved: {out_path}")

def scale_image(img, factor=1):
	"""Returns resize image by scale factor.
	This helps to retain resolution ratio while resizing.
	Args:
	img: image to be scaled
	factor: scale factor to resize
	"""
	return cv2.resize(img,(int(img.shape[1]*factor), int(img.shape[0]*factor)))


# def sliding_window(image, window_size, stride):
#     H, W = image.shape[:2]
#     window_h, window_w = window_size

#     # Calculate the last possible start positions
#     y_starts = list(range(0, H - window_h + 1, stride))
#     x_starts = list(range(0, W - window_w + 1, stride))

#     # Ensure the last window covers the bottom/right edge
#     if y_starts[-1] != H - window_h:
#         y_starts.append(H - window_h)
#     if x_starts[-1] != W - window_w:
#         x_starts.append(W - window_w)

#     for y in y_starts:
#         for x in x_starts:
#             window = image[y:y+window_h, x:x+window_w]
#             yield (x, y, window)


def sliding_window(image, window_size, stride):
    H, W = image.shape[:2]
    window_h, window_w = window_size
    x, y = 0, 0
    def iterate_coord(w_l, max_w, stride):
        x_list = []
        x = 0
        while x < max_w - w_l + 1:
            if x + stride + w_l > max_w:
                x = max_w - w_l
                x_list.append(x)
                return x_list
            else:
                x_list.append(x)
                x += stride
            
    y_list = iterate_coord(window_h, H, stride)
    x_list = iterate_coord(window_w, W, stride)
    for y in y_list:
        for x in x_list:
            window = image[y:y+window_h, x:x+window_w]
            print(f"INFO: shape of window: {window.shape}")
            yield (x, y, window)
    # while y < H  - window_h + 1:
    #     if y + stride + window_h > H:
    #         y = H - window_h
    #     else:
    #         y += stride
    #     while x < W - window_w + 1:
    #         if x + stride + window_w > W:
    #             x = W - window_w
    #         else:
    #             x += stride


def pad_image_for_sliding_window(image, window_size, stride, border_type, padding_position='bottom_right'):
    H, W = image.shape[:2]
    window_h, window_w = window_size

    # Calculate needed padding
    pad_h = (-(H - window_h) % stride) if H > window_h else window_h - H
    pad_w = (-(W - window_w) % stride) if W > window_w else window_w - W

    # Pad bottom and right sides

    if padding_position == 'bottom_right':
        # Calculate padding for bottom and right sides
        pad_h_top = 0
        pad_w_left = 0
        pad_h_bottom = pad_h
        pad_w_right = pad_w
    elif padding_position == 'center':
        pad_h_top = int(pad_h / 2) #centering padding
        pad_w_left = int(pad_w / 2)
        pad_h_bottom = pad_h - pad_h_top
        pad_w_right = pad_w - pad_w_left

    if border_type == 'constant':
        padded_image = cv2.copyMakeBorder(
        image,
        top=pad_h_top, bottom=pad_h_bottom,
        left=pad_w_left, right=pad_w_right,
        borderType=cv2.BORDER_CONSTANT,
        value = 0
    )
    elif border_type == 'replicate':
            padded_image = cv2.copyMakeBorder(
        image,
        top=pad_h_top, bottom=pad_h_bottom,
        left=pad_w_left, right=pad_w_right,
        borderType=cv2.BORDER_REPLICATE,
    )
    elif border_type == 'reflect':
        padded_image = cv2.copyMakeBorder(
        image,
        top=pad_h_top, bottom=pad_h_bottom,
        left=pad_w_left, right=pad_w_right,
        borderType=cv2.BORDER_REFLECT,
    )

    else:
        print('WARNING: bordertype is wrong')


    return padded_image


def generate_windows(image_index, image, window_size, stride, output_path):
    for i, (x, y, window) in enumerate(sliding_window(image, window_size, stride)):
        window_h, window_w = window_size
        final_output_path = str(Path(output_path)/ str(f'window_{image_index}_at_{x}_{y}.png'))
        cv2.imwrite(final_output_path, window)

def preprocess_image(image, window_size, stride, border_type, scale_img=0.5, cropping_dim = (1550, 1550)): #used for training with "carton dataset"
    #image = center_crop(image, cropping_dim) # Removes vignette from dataset and useless pans
    image = scale_image(image, scale_img)  # Scale down to half size for less computational requirements
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert BGR to RGB for consistency
    image = pad_image_for_sliding_window(image, window_size, stride, border_type)  # Pad the image for sliding window
    return image


# Show all windows on the image
def generate_windowed_dataset(input_path, window_size, stride, border_type, output_path, scale_factor=0.5):
    for image in Path(input_path).iterdir():
        image_index = parse_input_filename(image)  # Extract the index from the filename
        image = cv2.imread(str(image))
        image = preprocess_image(image, window_size, stride, border_type, scale_factor)
        generate_windows(image_index, image, window_size, stride, output_path)



def rename_files_in_dataset(directory, classification, i): # useful for linking pre and post windowed datasets
    j = 0
    for file_path in Path(directory).iterdir(): # temp name to reset names
        temp_name = f"{j}{file_path.suffix.lower()}"
        new_path = file_path.parent / temp_name
        file_path.rename(new_path)
        j += 1

    if classification not in ['healthy', 'faulty', 'testing']:
        raise ValueError("Classification must be either 'healthy', 'faulty' or 'testing'.")
    elif classification == 'healthy':
        classification = 'h'
    elif classification == 'faulty':
        classification = 'f'
    else:
        classification = 't' #used for testing or deployment
    for file_path in Path(directory).iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            new_name = f"{classification}_c{i}{file_path.suffix.lower()}"  # Create new name with class and index
            new_path = file_path.parent / new_name
            file_path.rename(new_path)
            print(f"Renamed {file_path} to {new_name}")
            i += 1
    return i  # Return the next index so next call for the next class we have an unanbiguos index


def parse_windowed_filename_py(file_path):
    #filename = file_path.split('\\')[-1] # path has alreaby been converted to string
    filename = os.path.basename(file_path)
    pattern = r'window_(\d+)_at_(\d+)_(\d+)\.png'
    match = re.match(pattern, filename)
    if match:
        return [int(match.group(1)), int(match.group(2)), int(match.group(3))]
    else:
        print(f"WARNING: Filename {filename} does not match the expected pattern.")
        return [0, 0, 0]


def parse_input_filename(file_path): #extracts univocal code to identify the image
    filename = os.path.basename(file_path)
    pattern = r'^[htf]_c(\d+)\.[a-z]+$'
    match = re.match(pattern, filename)
    if match:
        return int(match.group(1))
    else:
        print(f"WARNING: Filename {filename} does not match the expected pattern.")
        return 0
    
#def parse_filename(file_path):
#    return tf.py_function(parse_filename_py, [file_path], [tf.int32, tf.int32, tf.int32])

def retrieve_classification_from_path(file_path):
    filename = os.path.basename(file_path)
    pattern = r'^(h|f)_c\d+\.[a-z]+$'  # Simplified pattern to capture only 'h' or 'f'
    match = re.match(pattern, filename)
    if match:
        # Return 'h' or 'f' based on the first character
        return match.group(1)
    else:
        print(f"WARNING: Filename {filename} does not match the expected pattern.")
        return None

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
    i, x, y = parse_windowed_filename_py(file_path)
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
    #for batch in validation_dataset.take(1):
    #    print("Batch content:")
    #    print(batch)
    return training_dataset, validation_dataset



def get_testing_dataset(input_path, batch_size, image_size):
    list_ds = tf.data.Dataset.list_files(str(Path(input_path) / '*.png'), shuffle=False)
    extracted_ds = list_ds.map(lambda file_path: process_path(file_path, image_size))
    batched_ds = extracted_ds.batch(batch_size)
    # Debug: print shapes of first few images
    for img, label, meta in batched_ds.take(5):
        print("DEBUG: Image shape in dataset:", img.shape)
    return batched_ds


def find_image_path_and_shape(parent_folder_path, i):
    # Iterate through the subdirectories in the parent folder
    for root, dirs, files in os.walk(parent_folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            # Check if the file matches the index
            if parse_input_filename(file_path) == i:
                # Open the image and get its shape
                image = cv2.imread(file_path)
                if image is not None:
                    height, width = image.shape[:2]
                    return file_path, width, height


def empty_directory(directory):
    """Deletes all files in the specified directory."""
    for file_path in Path(directory).iterdir():
        if file_path.is_file():
            file_path.unlink()  # Delete the file
    print(f"All files in {directory} have been deleted.")

def overlay_and_save(original_image_path, mask, output_path):


    # Load the original image
    original_image = cv2.imread(str(original_image_path))
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib

    # Overlay the mask on the original image
    overlay = np.zeros_like(original_image)
    overlay[mask == 255] = [255, 0, 0]  # Mark faulty areas in red
    result = cv2.addWeighted(original_image, 0.8, overlay, 0.2, 0)

    # Save the result
    plt.imsave(output_path, result)


def transform_coordinates(x, y, original_width, original_height, INPUT_SIZE, scale_factor): #takes into account cropping and scaling for reconstruction of the position of the detected faulty area in the original image
    x_t = int((original_width / 2) - (INPUT_SIZE[0] / 2) + (x / scale_factor))
    y_t = int((original_height / 2) - (INPUT_SIZE[1] / 2) + (y / scale_factor))
    return x_t, y_t



def prepare_data(image_index, window_size, stride, border_type, faulty_source_path, healthy_source_path, faulty_output_path, healthy_output_path, blacklist_path, skip_generation): #creates the dataset with the selected window_size and stride
    # rename files in the output directories to ensure consistency (a continous index is employed for non ambiguous naming)
    image_index = rename_files_in_dataset(faulty_source_path, 'faulty', image_index)
    image_index = rename_files_in_dataset(healthy_source_path, 'healthy', image_index)


    # Generate windowed datasets for both faulty and healthy images

    if skip_generation == 'generate_all':
        print("INFO: starting generation of windows")
        generate_windowed_dataset(faulty_source_path, window_size, stride, border_type, faulty_output_path)
        generate_windowed_dataset(healthy_source_path, window_size, stride, border_type, healthy_output_path)
        print("INFO: finished generation of windows")
        print("INFO: starting removal of false positive windows")
        remove_blacklisted_images(blacklist_path, faulty_output_path)

    elif skip_generation == 'only_remove_blacklist':
        print("INFO: skipping generation of dataset, only removing blacklisted images")
        #remove known healthy windows from the faulty directory
        print("INFO: starting removal of false positive windows")
        remove_blacklisted_images(blacklist_path, faulty_output_path)
    elif skip_generation == 'skip_all':
        print("INFO: skipping generation of dataset and not removing blacklisted images")
    else:
        print("WARNING: skip_generation parameter is not recognized, performing generation of dataset and removing blacklisted images")
        print("INFO: starting generation of windows")
        generate_windowed_dataset(faulty_source_path, window_size, stride, border_type, faulty_output_path)
        generate_windowed_dataset(healthy_source_path, window_size, stride, border_type, healthy_output_path)
        print("INFO: finished generation of windows")
        print("INFO: starting removal of false positive windows")
        remove_blacklisted_images(blacklist_path, faulty_output_path)    
    return image_index


def remove_blacklisted_images(blacklist_path, faulty_output_path):
    """Deletes all files in the specified directory that match the blacklist."""
    for blacklist_item in Path(blacklist_path).iterdir():
        was_found = False
        for file_path in Path(faulty_output_path).iterdir():
            if file_path.is_file() and os.path.basename(file_path) == os.path.basename(blacklist_item): #here we verify that the name of the file in the blacklist matches the one of the output path
                file_path.unlink()  # Delete the file
                was_found = True
                break #early stop to not waist iterations
        if not was_found:
            print(f"WARNING: blacklisted image {os.path.basename(blacklist_item)} was not found")
    print(f"INFO: All files matching blacklist {blacklist_path} in {faulty_output_path} have been deleted.")
    

def log_training_session(project_root, history, train_model_flag, num_of_epochs, INPUT_SIZE, stride, border_type, seed, learning_rate, batch_size, dense_layer_dropout, data_directory, unwindowed_data_directory, windowed_performance_metrics, whole_image_performance_metrics):
    if train_model_flag:
        # Define the plot folder path
        plot_folder = Path(project_root) / 'plots'
        #plot_folder.mkdir(exist_ok=True)  # Create the folder if it doesn't exist

        # Define the log file path
        log_file = plot_folder / 'training_log.txt'

        # Get current date and time
        now = datetime.datetime.now()
        date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")

        # Extract final training and validation metrics
        final_train_accuracy = history.history['binary_accuracy'][-1]
        final_val_accuracy = history.history['val_binary_accuracy'][-1]
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]

        # Define settings parameters
        settings = {
            "num_of_epochs": num_of_epochs,
            "INPUT_SIZE": INPUT_SIZE,
            "stride": stride,
            "border_type": border_type,
            "seed": seed,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "dense_layer_dropout": dense_layer_dropout,
            "data_directory": data_directory,
            "unwindowed_data_directory": unwindowed_data_directory
        }

        # Log the training session details
        with open(log_file, 'a') as f:
            f.write(f"Training Session Log - {date_time_str}\n")
            f.write("Settings:\n")
            for key, value in settings.items():
                f.write(f"{key}: {value}\n")
            f.write("\nResults:\n")
            f.write(f"Final Training Accuracy: {final_train_accuracy}\n")
            f.write(f"Final Validation Accuracy: {final_val_accuracy}\n")
            f.write(f"Final Training Loss: {final_train_loss}\n")
            f.write(f"Final Validation Loss: {final_val_loss}\n")
            f.write("\n" + "-"*50 + "\n\n")
            for key, value in windowed_performance_metrics.items(): #logging and printing performance metrics
                f.write(f"windowed_{key}: {value}\n")
                print(f"windowed_{key}: {value}")
            for key, value in whole_image_performance_metrics.items():
                f.write(f"whole_image_{key}: {value}\n")
                print(f"whole_image_{key}: {value}")


def compute_ml_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    return metrics

def process_input_data(image_index, window_size, batch_size, stride, input_path, working_folder_path): #this function is similar to the ones used to prepare the dataset but it is meant as the input into the algorithm
    classification = 'testing'
    scale_factor=1
    image_index = 0
    border_type = 'reflect'  # Default border type, can be changed if needed
    rename_files_in_dataset(input_path, classification, image_index)
    generate_windowed_dataset(input_path, window_size, stride, border_type, working_folder_path, scale_factor)  # No scaling for testing
    test_windows = get_testing_dataset(working_folder_path, batch_size, window_size)
    return test_windows

def classify_dataset(dataset, model, scale_factor, window_size, unwindowed_data_directory, output_path):

    # Get true labels and predictions for the validation set
    y_pred = []
    log_metadata = []
    masks = {}
    paths = {}
    
    #dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    for images, labels, tensor_metadata in dataset:

        preds = model.predict(images)
        preds = (preds > 0.5).astype(int).flatten()  # Binary threshold
        #preparing data for confusion matrix
        y_pred.extend(preds)

        # Collect metadata for each prediction
        tensor_i, tensor_x, tensor_y = tensor_metadata
        list_i = tensor_i.numpy().tolist()
        list_x = tensor_x.numpy().tolist()
        list_y = tensor_y.numpy().tolist()

        log_metadata.extend(list(zip(list_i, list_x, list_y)))

    for pred, (i, x, y) in zip(y_pred, log_metadata):
        # Reconstruct the original image dimensions
        original_path, original_width, original_height = find_image_path_and_shape(unwindowed_data_directory, i)
        if i not in masks:
            masks[i] = np.zeros((original_height, original_width), dtype=np.uint8)
            paths[i] = original_path
        if pred == 1:
            # Mark the area as faulty in the mask
            #x, y = transform_coordinates(x, y, original_width, original_height, window_size, scale_factor)
            masks[i][y:y+int(window_size[0] / scale_factor), x:x+int(window_size[1] / scale_factor)] = 255

    #construct overlays for each image

    #first we empty the output directory
    empty_directory(output_path)
    # Iterate over the dictionary items to ensure correct alignment
    for i in masks:
        mask = masks[i]
        original_image_path = paths[i]
        # Save the mask as an image
        overlay_path = str(Path(output_path) / f'overlay_{i}.png')
        overlay_and_save(original_image_path, mask, overlay_path)
        print(f"Mask for image {i} saved to {overlay_path}")
    print("INFO: Overlays saved to output directory.")


def copy_folder_contents(src_folder, dst_folder):
    """
    Copies all files from src_folder to dst_folder.
    Creates dst_folder if it does not exist.
    """
    src = Path(src_folder)
    dst = Path(dst_folder)
    dst.mkdir(parents=True, exist_ok=True)
    for file_path in src.iterdir():
        if file_path.is_file():
            shutil.copy2(file_path, dst / file_path.name)
    print(f"Copied all files from {src} to {dst}")