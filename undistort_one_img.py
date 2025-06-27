import cv2
import scipy.io
from pathlib import Path
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent
mat = scipy.io.loadmat(project_root / 'dataset_medi' / 'TARATURA' / 'medium_dataset_taratura.mat')
camera_matrix = mat['K']
dist_coeffs = mat['dist']
img= cv2.imread(str(project_root / 'dataset_piccoli' / 'dezoommata_green.png'), cv2.IMREAD_COLOR)
img_und = cv2.undistort(img, camera_matrix, dist_coeffs) 

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("distorted Image")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_und, cv2.COLOR_BGR2RGB))
plt.title("Undistorted Image")
plt.axis('off')
plt.show()