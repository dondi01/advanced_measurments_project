import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, f1_score
import numpy as np
import time
from pathlib import Path
import tensorflow as tf
from keras.applications import ResNet50
from keras.utils import plot_model
import shutil
import paths
import sys
sys.path.append('./functions')
import ml_functions as mlfn
project_root = Path(__file__).resolve().parent

image_index = 0
window_size = (512, 512)
batch_size = 32
stride = 250
scale_factor = 0.5

input_path = str(project_root / 'ml_datasets' / 'deployment' / 'input_folder')
working_folder_path = str(project_root / 'ml_datasets' / 'deployment' / 'working_folder')
output_path = str(project_root / 'ml_datasets' / 'deployment' / 'output_folder')

# Clean up the working folder and output folder
mlfn.empty_directory(working_folder_path)
mlfn.empty_directory(output_path)

# Generate windowed dataset
test_dataset = mlfn.process_input_data(image_index, window_size, batch_size, stride, input_path, working_folder_path)

# Load the model
model = tf.keras.models.load_model(str(project_root / 'cnn_models' /'model06071930.keras'))

# Classify the dataset
mlfn.classify_dataset(test_dataset, model, scale_factor, window_size, input_path, output_path)