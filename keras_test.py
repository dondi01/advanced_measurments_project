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

# SETTIINGS

train_model_flag = True
skip_generation = 'skip_all' # 'generate_all', 'only_remove_blacklist', 'skip_all'
num_of_epochs = 25
INPUT_SIZE = (512, 512, 3)
stride = 250
border_type = 'reflect'
seed = 42
batch_size = 32
dense_layer_dropout = 0.2
data_directory = str(project_root / 'ml_datasets' /'carton_windowed')
unwindowed_data_directory = str(project_root / 'ml_datasets' /'carton_baseline')
output_directory = str(project_root / 'ml_datasets' / 'output') #for overlays
faulty_source_path = str( project_root / 'ml_datasets' / 'carton_baseline' / 'faulty')
healthy_source_path = str( project_root / 'ml_datasets' / 'carton_baseline' / 'healthy')
faulty_output_path = str( project_root / 'ml_datasets' / 'carton_windowed' / 'faulty')
healthy_output_path = str( project_root / 'ml_datasets' / 'carton_windowed' / 'healthy')
blacklist_path = str(project_root / 'ml_datasets' / 'blacklists' / 'carton_windowed_s250_w512_h512')



tf.keras.utils.set_random_seed(seed)

def use_model(input_path, output_path, window_size, stride, border_type):
    mlfn.generate_windowed_dataset(input_path, window_size, stride, border_type, output_path)


def train_model(INPUT_SIZE, training_dataset, validation_dataset):
    """Function to train the model, can be used in the future for training"""
    base_model = ResNet50(
                    include_top = False,
                    weights='imagenet',
                    input_shape=INPUT_SIZE,
                    name='resnet50_base'
                    )  # Load pretrained on ImageNet
    base_model.trainable = False  # Freeze the base model
    #base_model.summary()
    #plot_model(model, to_file='resnet50.png', show_shapes=True, show_layer_names=True)


    ## MODEL BUILDING

    #definition a data augmentation layer only active during training
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", input_shape=INPUT_SIZE),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ])

    inputs = tf.keras.Input(shape=INPUT_SIZE, name='input_layer') #there is a possibility to do sparse and batch shape and size, check if useful
    x = data_augmentation(inputs) # only actctive during training
    x = base_model(x, training=False)  # Pass the input through the base model
    x = tf.keras.layers.GlobalAveragePooling2D()(x)  # Add a global average pooling layer
    x = tf.keras.layers.Dropout(dense_layer_dropout)(x)  # Add a dropout layer for regularization
    x = tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')(x)  # Add a dense layer for binary classification
    model = tf.keras.Model(inputs=inputs, outputs=x, name='resnet50_cardboard')
    #model.summary()


    # definition of optimizer, loss and metrics
    model.compile(optimizer= tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.BinaryCrossentropy(from_logits=False),
                metrics = [tf.keras.metrics.BinaryAccuracy()])

    ### TRAINING
    history = model.fit(training_dataset, epochs=num_of_epochs, validation_data=validation_dataset)
    return model, history



print(shutil.which("dot"))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


image_index = 0

mlfn.prepare_data(image_index, INPUT_SIZE[:2], stride, border_type, faulty_source_path, healthy_source_path, faulty_output_path, healthy_output_path, blacklist_path, skip_generation)
training_dataset, validation_dataset = mlfn.get_training_validation_datasets(data_directory, batch_size, INPUT_SIZE)

if train_model_flag:
    print("INFO: starting training")
    model, history = train_model(INPUT_SIZE, training_dataset, validation_dataset)
    print("INFO: saving model")
    model.save(str(project_root / 'cnn_models' /'test.keras'))


    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig(str(project_root / 'plots' /'training_plot.png'))
    plt.close()
    #plt.show()
else:
    model = tf.keras.models.load_model(str(project_root / 'cnn_models' /'test.keras'))
    print("INFO: Model loaded from file.")
    





# Get true labels and predictions for the validation set
y_true = []
y_pred = []
#original_metadata = mlfn.get_original_image_metadata(unwindowed_data_directory)
log_metadata = []

validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

for images, labels, tensor_metadata in validation_dataset:

    preds = model.predict(images)
    preds = (preds > 0.5).astype(int).flatten()  # Binary threshold
    #preparing data for confusion matrix
    y_pred.extend(preds)
    y_true.extend(labels.numpy().astype(int).flatten())

    # Collect metadata for each prediction
    tensor_i, tensor_x, tensor_y = tensor_metadata
    list_i = tensor_i.numpy().tolist()
    list_x = tensor_x.numpy().tolist()
    list_y = tensor_y.numpy().tolist()

    log_metadata.extend(list(zip(list_i, list_x, list_y)))

masks = {}
paths = {}
scale_factor = 0.5
for pred, (i, x, y) in zip(y_pred, log_metadata):
    # Reconstruct the original image dimensions
    original_path, original_width, original_height = mlfn.find_image_path_and_shape(unwindowed_data_directory, i)
    if i not in masks:
        masks[i] = np.zeros((original_height, original_width), dtype=np.uint8)
        paths[i] = original_path
    if pred == 1:
        # Mark the area as faulty in the mask
        x, y = mlfn.transform_coordinates(x, y, original_width, original_height, INPUT_SIZE, scale_factor)
        masks[i][y:y+int(INPUT_SIZE[0] / scale_factor), x:x+int(INPUT_SIZE[1] / scale_factor)] = 255


#construct overlays for each image

#first we empty the output directory
mlfn.empty_directory(output_directory)
# Iterate over the dictionary items to ensure correct alignment
initial_dataset_confusion_matrix = {}
for i in masks:
    mask = masks[i]
    original_image_path = paths[i]
    # Save the mask as an image
    output_path = str(Path(output_directory) / f'overlay_{i}.png')
    mlfn.overlay_and_save(original_image_path, mask, output_path)
    print(f"Mask for image {i} saved to {output_path}")

    #gather data if the initial image was faulty or not (before windowing, which might create images with no faults, even if classified as faulty)
    if np.all(mask == 0):
        y_pred_i = 0
    else:
        y_pred_i = 1
    
    true_classification = mlfn.retrieve_classification_from_path(original_image_path)

    if true_classification == 'h':
        y_i = 0
    else:
        y_i = 1

    initial_dataset_confusion_matrix[i] = {
        'y_i' : y_i,
        'y_pred_i' : y_pred_i
    }

# Compute confusion matrix (windowed)
plt.figure(figsize=(8, 8))
cm = confusion_matrix(y_true, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.savefig(str(project_root / 'plots' /'confusion_matrix_windowed.png'))
plt.close()


y_i_list = [v['y_i'] for v in initial_dataset_confusion_matrix.values()]
y_pred_i_list = [v['y_pred_i'] for v in initial_dataset_confusion_matrix.values()]

# Compute confusion matrix
plt.figure(figsize=(8, 8))
cm = confusion_matrix(y_i_list, y_pred_i_list, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.savefig(str(project_root / 'plots' / 'confusion_matrix_original_data.png'))
plt.close()

perf_metrics_windowed = mlfn.compute_ml_metrics(y_true, y_pred)
perf_metrics_whole_image = mlfn.compute_ml_metrics(y_i_list, y_pred_i_list)
#train_model_flag = True # override for debug
mlfn.log_training_session(project_root, history, train_model_flag, num_of_epochs, INPUT_SIZE, stride, border_type, seed, batch_size, dense_layer_dropout, data_directory, unwindowed_data_directory, perf_metrics_windowed, perf_metrics_whole_image)