import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent

filepath = str(project_root / 'Reconstructed' / '*.png')

test_brown = cv2.imread(str(project_root / 'dataset_piccoli' / 'dataset_luce_parallela' / 'green31.png'))

def iterative_transformer(image, n, kernel_dilate=(3, 3), kernel_blur=(5, 5)):
    for i in range(n):
        image = cv2.dilate(image, kernel_dilate, iterations=1)
        image = cv2.blur(image, kernel_blur)
        _, image = cv2.threshold(test_subtracted, 0, 255, cv2.THRESH_BINARY)

        #image = cv2.GaussianBlur(image, kernel_blur, 0)
    return image

test_brown = cv2.cvtColor(test_brown, cv2.COLOR_BGR2GRAY)
test_brown_blurred = cv2.medianBlur(test_brown, 11)
test_subtracted = cv2.subtract(test_brown, test_brown_blurred)
test_subtracted = cv2.threshold(test_subtracted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
test_subtracted = iterative_transformer(test_subtracted, 100, (100,100), (51,51))
test_subtracted = cv2.Canny(test_subtracted, 100, 200)
#test_subtracted = cv2.cvtColor(test_subtracted, cv2.COLOR_YCrCb2GRAY)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(test_brown, cv2.COLOR_BGR2RGB))
plt.title("Original (YCrCb)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(test_subtracted, cv2.COLOR_BGR2RGB))
plt.title("Blurred (YCrCb)")
plt.axis('off')

plt.show()