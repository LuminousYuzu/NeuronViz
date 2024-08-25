import cv2
import numpy as np
import matplotlib.pyplot as plt

# Absolute path to the image
image_path = '/Users/kyleliu/Desktop/Neuron Project/6.jpg'

# Load the image
image = cv2.imread(image_path)

# Check if image is loaded successfully
if image is None:
    raise FileNotFoundError(f"Image file not found at {image_path}")

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Manual Thresholding
manual_threshold_value = 50
_, binary_manual = cv2.threshold(gray_image, manual_threshold_value, 255, cv2.THRESH_BINARY)

# Otsu's Thresholding
_, binary_otsu = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Define the contour area threshold
contour_area_threshold = 0  # Adjust this value based on your needs

# Function to draw contours on the image
def draw_contours(binary_img, image, area_threshold):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [c for c in contours if cv2.contourArea(c) > area_threshold]
    contoured_image = image.copy()
    cv2.drawContours(contoured_image, filtered_contours, -1, (0, 255, 0), 2)
    return contoured_image, len(filtered_contours)

# Draw contours on the images and get the count of contours
contoured_manual, count_manual = draw_contours(binary_manual, image, contour_area_threshold)
contoured_otsu, count_otsu = draw_contours(binary_otsu, image, contour_area_threshold)

# Convert BGR images to RGB for matplotlib
def convert_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

image_rgb = convert_to_rgb(image)
binary_manual_rgb = cv2.cvtColor(binary_manual, cv2.COLOR_GRAY2RGB)
binary_otsu_rgb = cv2.cvtColor(binary_otsu, cv2.COLOR_GRAY2RGB)

contoured_manual_rgb = convert_to_rgb(contoured_manual)
contoured_otsu_rgb = convert_to_rgb(contoured_otsu)

# Display the images
plt.figure(figsize=(15, 10))

# Original image
plt.subplot(2, 3, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

# Manual Threshold image
plt.subplot(2, 3, 2)
plt.imshow(binary_manual_rgb, cmap='gray')
plt.title('Manual Threshold Image')
plt.axis('off')

# Contoured image from Manual Thresholding
plt.subplot(2, 3, 3)
plt.imshow(contoured_manual_rgb)
plt.title(f'Contoured Manual Threshold\nNumber of contours: {count_manual}')
plt.axis('off')

# Otsu's Threshold image
plt.subplot(2, 3, 5)
plt.imshow(binary_otsu_rgb, cmap='gray')
plt.title('Otsu Threshold Image')
plt.axis('off')

# Contoured image from Otsu's Thresholding
plt.subplot(2, 3, 6)
plt.imshow(contoured_otsu_rgb)
plt.title(f'Contoured Otsu Threshold\nNumber of contours: {count_otsu}')
plt.axis('off')

plt.tight_layout()
plt.show()

