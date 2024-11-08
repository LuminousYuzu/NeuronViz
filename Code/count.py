import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tkinter import Tk, filedialog, Scale, Button, Label, HORIZONTAL, StringVar, Entry
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import PhotoImage
from PIL import Image, ImageTk

def load_image(image_path):
    """Load an image from the specified path."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file not found at {image_path}")
    return image

def convert_to_grayscale(image):
    """Convert an image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_thresholds(gray_image, manual_threshold_value):
    """Apply manual and Otsu's thresholding to the grayscale image."""
    # Manual Thresholding
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

def process_images_in_folder(folder_path, manual_threshold_value):
    """Process each image in the folder and return results."""
    results = []
    otsu_thresholds = []

    # Create a subfolder for storing contoured images
    contoured_images_folder = os.path.join(folder_path, 'contoured_images')
    os.makedirs(contoured_images_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            image = load_image(image_path)
            gray_image = convert_to_grayscale(image)
            binary_manual, binary_otsu, otsu_threshold_value = apply_thresholds(gray_image, manual_threshold_value)
            
            # Define the contour area threshold
            contour_area_threshold = 0  # Adjust this value as needed
            
            # Draw contours
            contoured_manual, count_manual = draw_contours(binary_manual, image, contour_area_threshold)
            contoured_otsu, count_otsu = draw_contours(binary_otsu, image, contour_area_threshold)
            
            # Save contoured images
            contoured_manual_path = os.path.join(contoured_images_folder, f'contoured_manual_{filename}')
            contoured_otsu_path = os.path.join(contoured_images_folder, f'contoured_otsu_{filename}')
            cv2.imwrite(contoured_manual_path, contoured_manual)
            cv2.imwrite(contoured_otsu_path, contoured_otsu)
            
            # Store results
            results.append([image_path, count_manual, count_otsu])
            otsu_thresholds.append(otsu_threshold_value)
    
    return results, otsu_thresholds

def update_results(folder_path, manual_threshold_value, output_csv_path, canvas, ax):
    """Update the results and plot based on the manual threshold value."""
    results, otsu_thresholds = process_images_in_folder(folder_path, manual_threshold_value)
    
    # Save results to CSV
    df = pd.DataFrame(results, columns=['Image Path', 'Manual Count', 'Otsu Count'])
    df.to_csv(output_csv_path, index=False)
    
    # Update plot
    ax.clear()
    ax.hist(otsu_thresholds, bins=20, color='blue', alpha=0.7)
    ax.set_title('Distribution of Otsu\'s Thresholds')
    ax.set_xlabel('Otsu Threshold Value')
    ax.set_ylabel('Frequency')
    canvas.draw()

def on_folder_drop(event, manual_threshold_slider, manual_threshold_var, canvas, ax):
    """Handle folder drop event."""
    global folder_path, output_csv_path  # Declare as global to modify them
    folder_path = event.data.strip('{}')  # Remove curly braces if present
    if os.path.isdir(folder_path):
        output_csv_path = os.path.join(folder_path, 'results.csv')
        update_results(folder_path, manual_threshold_slider.get(), output_csv_path, canvas, ax)
    else:
        print("Invalid folder path")

def main():
    # Create a TkinterDnD root window
    root = TkinterDnD.Tk()
    root.title("Image Analysis Dashboard")
    
    # Declare folder_path and output_csv_path as global
    global folder_path, output_csv_path
    folder_path = None  # Initialize with None
    output_csv_path = None  # Initialize with None

    # Create a frame for the controls
    control_frame = ttk.Frame(root, padding="10")
    control_frame.pack(side='top', fill='x')
    
    # Commenting out the icon loading part
    # script_dir = os.path.dirname(__file__)  # Directory of the script
    # image_path = os.path.join(script_dir, 'NeuronViz', 'icon.webp')  # Construct the relative path
    # image = Image.open(image_path)
    # folder_icon = ImageTk.PhotoImage(image)

    # Create a label without the icon
    drag_label = Label(root, text="Drag your folder to start", compound='left', font=("Arial", 14))
    drag_label.pack(pady=20)
    
    # Create a matplotlib figure for the histogram
    fig, ax = plt.subplots(figsize=(5, 4))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
    
    # Create a variable to hold the manual threshold value
    manual_threshold_var = StringVar(value='50')
    
    # Create a slider for manual threshold value
    manual_threshold_slider = ttk.Scale(control_frame, from_=0, to=255, orient=HORIZONTAL,
                                        command=lambda val: manual_threshold_var.set(f"{int(float(val))}"))
    manual_threshold_slider.set(50)  # Default value
    manual_threshold_slider.pack(side='left', padx=5, pady=5)
    
    # Create a label to display the slider value
    slider_value_label = ttk.Label(control_frame, textvariable=manual_threshold_var)
    slider_value_label.pack(side='left', padx=5, pady=5)
    
    # Create an entry box for manual threshold value
    manual_threshold_entry = Entry(control_frame, textvariable=manual_threshold_var, width=5)
    manual_threshold_entry.pack(side='left', padx=5, pady=5)
    
    # Create a button to update results
    update_button = ttk.Button(control_frame, text="Update Results", command=lambda: update_results(
        folder_path, int(manual_threshold_var.get()), output_csv_path, canvas, ax) if folder_path and output_csv_path else print("No folder selected"))
    update_button.pack(side='left', padx=5, pady=5)
    
    # Bind the drop event to the root window
    root.drop_target_register(DND_FILES)
    root.dnd_bind('<<Drop>>', lambda event: on_folder_drop(event, manual_threshold_slider, manual_threshold_var, canvas, ax))
    
    # Run the Tkinter main loop
    root.mainloop()

if __name__ == "__main__":
    main()








