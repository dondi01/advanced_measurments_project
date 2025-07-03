import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
import tensorflow as tf
from keras.applications import ResNet50
from keras.utils import plot_model
import shutil
import paths
project_root = Path(__file__).resolve().parent
print(shutil.which("dot"))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

INPUT_SIZE = (512, 512, 3)
seed = 42
tf.keras.utils.set_random_seed(seed)
batch_size = 32
data_directory = str(project_root / 'ml_datasets' )


# loading datasetsn for training and validation
training_dataset = tf.keras.utils.image_dataset_from_directory(
    data_directory,
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=INPUT_SIZE[:2],
    batch_size = batch_size
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    data_directory,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=INPUT_SIZE[:2],
    batch_size = batch_size
)

base_model = ResNet50(
                include_top = False,
                weights='imagenet',
                input_shape=INPUT_SIZE,
                name='giovanni_resnet50'
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
x = tf.keras.layers.Dropout(0.2)(x)  # Add a dropout layer for regularization
x = tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')(x)  # Add a dense layer for binary classification
model = tf.keras.Model(inputs=inputs, outputs=x, name='giovanni_resnet')
#model.summary()


# definition of optimizer, loss and metrics
model.compile(optimizer= tf.keras.optimizers.Adam(),
              loss = tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics = [tf.keras.metrics.BinaryAccuracy()])

### TRAINING
history = model.fit(training_dataset, epochs=2, validation_data=validation_dataset)

model.save("test.keras")

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
plt.show()



