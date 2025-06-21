import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("C:/Users/franc/Desktop/Scuola/Measurement/advanced_measurments_project/Reconstructed/green_buco_in_piu.png")
lightened = cv2.convertScaleAbs(image, alpha=1, beta=100)
gray = cv2.cvtColor(lightened, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# thresh = cv2.adaptiveThreshold(
#     blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#     cv2.THRESH_BINARY_INV, 11, 2
# )
thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
h, w = thresh.shape
for cnt in contours:
    x, y, cw, ch = cv2.boundingRect(cnt)
    if cw >= w - 2 or ch >= h - 2:
        continue 
# Draw all contours
all_contours_img = lightened.copy()
cv2.drawContours(all_contours_img, contours, -1, (255, 0, 0), 2)

# Draw only rectangular contours (less strict)
rectangular_img = lightened.copy()
for cnt in contours:
    epsilon = 0.03 * cv2.arcLength(cnt, True)  # More tolerant
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    if len(approx) == 4 and cv2.contourArea(cnt) > 50:  # Lower area threshold, no convexity check
        cv2.drawContours(rectangular_img, [approx], -1, (0, 255, 0), 2)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(all_contours_img, cv2.COLOR_BGR2RGB))
plt.title('All Contours')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(rectangular_img, cv2.COLOR_BGR2RGB))
plt.title('Rectangular Contours')
plt.axis('off')

plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
plt.title('Thresholded Image')
plt.axis('off')

plt.show()