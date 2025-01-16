import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.measure import profile_line 

def pixel_slice_selection(image_path):
    """
    Open an image, display it, allow pixel slice selection, and return the array of selected pixels.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        np.ndarray: Array of selected pixels.
    """
    # Open the image and convert it to a NumPy array
    image = Image.open(image_path)
    image_array = np.array(image)

    # Show the image using matplotlib
    fig, ax = plt.subplots()
    ax.imshow(image_array,cmap='gray')
    ax.set_title("Select a rectangular region by dragging")

    # Create a callback to capture the selected region
    coords = {"start": None, "end": None}

    def on_click(event):
        if event.inaxes == ax:
            coords["start"] = (int(event.ydata), int(event.xdata))
            print(f"Selection started at: {coords['start']}")

    def on_release(event):
        if event.inaxes == ax:
            coords["end"] = (int(event.ydata), int(event.xdata))
            print(f"Selection ended at: {coords['end']}")
            plt.close(fig)

    # Connect the mouse events
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("button_release_event", on_release)

    plt.show()

    # Extract the pixel slice
    profile = profile_line(image_array,coords["start"],coords["end"])
    return profile


# Example usage
image_path = "results/library/usaf_taped/magnitude.png"
profile = pixel_slice_selection(image_path)

plt.plot(profile)
plt.ylabel('intensity')
plt.xlabel('distance along line')
plt.show()
