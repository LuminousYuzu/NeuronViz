import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    """Load an image from the specified path."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file not found at {image_path}")
    return image

def convert_to_grayscale(image):
    """Convert an image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_thresholds(gray_image):
    """Apply manual and Otsu's thresholding to the grayscale image."""
    # Manual Thresholding
    manual_threshold_value = 50
    _, binary_manual = cv2.threshold(gray_image, manual_threshold_value, 255, cv2.THRESH_BINARY)
    
    # Otsu's Thresholding
    otsu_threshold_value, binary_otsu = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary_manual, binary_otsu, otsu_threshold_value

def draw_contours(binary_img, image, area_threshold):
    """Draw contours on the image based on the binary image and area threshold."""
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [c for c in contours if cv2.contourArea(c) > area_threshold]
    contoured_image = image.copy()
    cv2.drawContours(contoured_image, filtered_contours, -1, (0, 255, 0), 2)
    return contoured_image, len(filtered_contours)

def overlay_text(img, text, position, font_scale=1, color=(255, 255, 255), thickness=2):
    """Overlay text on an image."""
    img_copy = img.copy()
    cv2.putText(img_copy, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    return img_copy

def compute_histogram(gray_image, bins=31):
    """Compute the histogram of pixel intensities."""
    hist = cv2.calcHist([gray_image], [0], None, [bins], [0, bins])
    return hist.flatten()

def display_results(image, binary_manual, contoured_manual, count_manual,
                    binary_otsu, contoured_otsu, count_otsu, otsu_threshold_value, hist):
    """Display the results including the original image, binary images, contoured images, and histogram."""
    def convert_to_rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert images to RGB for matplotlib
    image_rgb = convert_to_rgb(image)
    binary_manual_rgb = cv2.cvtColor(binary_manual, cv2.COLOR_GRAY2RGB)
    binary_otsu_rgb = cv2.cvtColor(binary_otsu, cv2.COLOR_GRAY2RGB)
    contoured_manual_rgb = convert_to_rgb(contoured_manual)
    contoured_otsu_rgb = convert_to_rgb(contoured_otsu)
    
    # Overlay the Otsu threshold value on the final contoured image
    text_position = (10, 30)  # Adjust position as needed
    contoured_otsu_with_text = overlay_text(contoured_otsu_rgb, f'Otsu Threshold: {otsu_threshold_value:.2f}', text_position)
    
    # Display the images and histogram
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # First row
    axs[0, 0].imshow(image_rgb)
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(binary_manual_rgb, cmap='gray')
    axs[0, 1].set_title('Manual Threshold Image')
    axs[0, 1].axis('off')
    
    axs[0, 2].imshow(contoured_manual_rgb)
    axs[0, 2].set_title(f'Contoured Manual Threshold\nNumber of contours: {count_manual}')
    axs[0, 2].axis('off')
    
    # Second row
    axs[1, 0].bar(range(0, 31), hist, width=1, color='blue')
    axs[1, 0].set_title('Pixel Intensity Distribution')
    axs[1, 0].set_xlabel('Pixel Intensity')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].set_xlim(0, 30)
    
    axs[1, 1].imshow(binary_otsu_rgb, cmap='gray')
    axs[1, 1].set_title(f'Otsu Threshold Image\nThreshold: {otsu_threshold_value:.2f}')
    axs[1, 1].axis('off')
    
    axs[1, 2].imshow(contoured_otsu_with_text)
    axs[1, 2].set_title(f'Contoured Otsu Threshold\nNumber of contours: {count_otsu}')
    axs[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    image_path = '/Users/kyleliu/Desktop/Neuron Project/6.jpg'
    
    # Load and process the image
    image = load_image(image_path)
    gray_image = convert_to_grayscale(image)
    binary_manual, binary_otsu, otsu_threshold_value = apply_thresholds(gray_image)
    
    # Define the contour area threshold
    contour_area_threshold = 0  # Adjust this value as needed
    
    # Draw contours
    contoured_manual, count_manual = draw_contours(binary_manual, image, contour_area_threshold)
    contoured_otsu, count_otsu = draw_contours(binary_otsu, image, contour_area_threshold)
    
    # Compute histogram
    hist = compute_histogram(gray_image, bins=31)
    
    # Display results
    display_results(image, binary_manual, contoured_manual, count_manual,
                    binary_otsu, contoured_otsu, count_otsu, otsu_threshold_value, hist)

if __name__ == "__main__":
    main()







