import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_centroid(contour):
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = 0, 0
    return (cx, cy)

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Absolute path to the image
image_path = '/Users/kyleliu/Desktop/Neuron Project/4.jpg'

# Load the image
image = cv2.imread(image_path)

# Check if image is loaded successfully
if image is None:
    raise FileNotFoundError(f"Image file not found at {image_path}")

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
threshold_value = 50  # You can adjust this value based on your image
_, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Count the number of contours
number_of_neurons = len(contours)

# Draw contours on the original image
contoured_image = image.copy()

# Find centroids of the contours
centroids = [find_centroid(c) for c in contours]

# Threshold for marking close contours (in pixels)
distance_threshold = 50  # Adjust this value based on your needs

# List to keep track of contours to be drawn
contours_to_draw = []

# Mark contours that are within the threshold distance
for i, centroid1 in enumerate(centroids):
    for j, centroid2 in enumerate(centroids):
        if i != j and distance(centroid1, centroid2) < distance_threshold:
            # Add contours if they are within the distance threshold
            if not any(np.array_equal(contours[i], existing) for existing in contours_to_draw):
                contours_to_draw.append(contours[i])
            if not any(np.array_equal(contours[j], existing) for existing in contours_to_draw):
                contours_to_draw.append(contours[j])

# Draw only the selected contours
cv2.drawContours(contoured_image, contours_to_draw, -1, (0, 255, 0), 2)

# Convert BGR images to RGB for matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
contoured_image_rgb = cv2.cvtColor(contoured_image, cv2.COLOR_BGR2RGB)
binary_image_rgb = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)  # Convert to RGB for consistency

# Display the original, binary, and contoured images side by side
plt.figure(figsize=(15, 5))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

# Binary image
plt.subplot(1, 3, 2)
plt.imshow(binary_image_rgb, cmap='gray')
plt.title('Binary Threshold Image')
plt.axis('off')

# Contoured image
plt.subplot(1, 3, 3)
plt.imshow(contoured_image_rgb)
plt.title(f'Contoured Image\nNumber of neurons: {number_of_neurons}')
plt.axis('off')

plt.tight_layout()
plt.show()
