import tensorflow as tf
import paths
from pathlib import Path  
import matplotlib.pyplot as plt 
import cv2 
project_root = Path(__file__).resolve().parent



if __name__ == "__main__":
    tensor=tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=90,
        width_shift_range=0,
        height_shift_range=0,
        brightness_range=None,
        shear_range=0,
        zoom_range=1.5,
        channel_shift_range=0.0,
        fill_mode='nearest',
        cval=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0,
        interpolation_order=1,
        dtype=None
    )
    _,_,_,recomposed=paths.define_files("green_scratched", project_root)
    recomposed=cv2.imread(recomposed, cv2.IMREAD_COLOR)
    recomposed_=tensor.random_transform(recomposed, seed=42)

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(recomposed)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title('Transformed Image')
    plt.imshow(recomposed_)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

