import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import os
from PIL import Image
import sys
import numpy as np
import time
from RPiLedMatrix import RPiLedMatrix
from picamera2 import Picamera2
import importlib 
import gpiod

# Custom functions
import fpm_functions as fpm 
importlib.reload(fpm) # Reload

# Data gathering script for FPM. Gather entire camera FOV, so we can reconstruct full frame later
# To take a cropped dataset use main.py

##########################################################################################################
# Key setup variables

data_folder = 'data/recent' # For saving data images
img_size = 300 # Small image size (for preview)

grid_size = 15 # Entire LED array is 16x16 but due to misalignment we will only use 15x15
num_images = grid_size**2
fpm_exposure = int(600e3)  # In microseconds for FPM image capture, 500-700ms
led_color = 'white' # Illumination color, white is best (green possible with longer exposure)

# Use different exposure time in darkfield
df_exposure_on = False 
df_start = 20 # Image number of first darkfield, 20 for 60mm LED2SAMPLE
df_exposure = int(1000e3) 

# Misc
brightfield_preview = True # Preview bright or darkfield
preview_exposure = int(60e3) if brightfield_preview else int(500e3) # In microseconds for preview
x_coords,y_coords = fpm.LED_spiral(grid_size,x_offset=1,y_offset=0) # LED sequence (ensure first LED is aligned with optical axis)
crop_start_x = int(1456/2 - img_size/2) # For preview
crop_start_y = int(1088/2 - img_size/2)

## Miscelaneous 

# Turn on fan
def fan(flag):
    # Set the chip and line (pin number on the chip)
    chip_name = "gpiochip4"
    line_offset = 45  # This corresponds to GPIO pin 45
    # Open the GPIO chip
    chip = gpiod.Chip(chip_name)
    line = chip.get_line(line_offset)
    # Request the line as an output with a value of 0 (low)
    line.request(consumer="my_gpio_control", type=gpiod.LINE_REQ_DIR_OUT)
    # Set the GPIO pin
    line.set_value(0 if flag==True else 1)  # 0 for on, 1 for off
    # Release the line when done
    line.release()
fan(True)

# Clean up camera and LED array (in case they were left running)
def cleanup():
    led_matrix = RPiLedMatrix()
    led_matrix.off()
    camera = Picamera2()
    camera.stop()
    camera.close()
cleanup()

#######################################################################################################################
# Image preview and alignment process

# Set up figure - this will be the main UI for the entire process
plt.ion() # Allow live plotting
fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Figure size (10x5)
fig.suptitle('Fourier Ptychography - Data Gathering')

# Axis 0 will be full FOV, axis 1 will be cropped region
axes[0].set_aspect(1456 / 1088)  # Aspect ratio for the full frame
axes[1].set_aspect('equal')  # Aspect ratio for the cropped frame is square
axes[0].set_title("Full Camera FOV")
axes[1].set_title("Cropped region")

# Initialise LED array
led_matrix = RPiLedMatrix()
led_matrix.set_rotation(135) # Ensure 0,0 is bottom left pixel and as shown on microscope

# Initialize camera
camera = Picamera2()
# print(camera.camera_controls)
still_config = camera.create_still_configuration(
    # Need to use whole region then crop (otherwise we lose resolution)
    main={'size': (1456,1088)}, controls={"AnalogueGain": 1, 'ExposureTime': preview_exposure, 'AeEnable': False})
camera.configure(still_config)
camera.start()

# Preview alignment using matplotlib
if brightfield_preview:
    led_matrix.show_circle(radius=2, color=led_color, brightness=1)  # Turn on brightfield LEDs
else:
    led_matrix.show_circle(radius=2, color='black', outside_color=led_color) # Darkfield preview
abort_script = False
continue_script = False

# Handles arrow key crop adjustments
def on_key(event):
    global crop_start_x, crop_start_y
    match event.key:
        case 'up': 
            crop_start_y -= 5
        case 'down': 
            crop_start_y += 5
        case 'left': 
            crop_start_x -= 5
        case 'right':
            crop_start_x += 5
    
# Function to abort script
def abort_callback(event):
    global abort_script
    abort_script = True
    plt.close(fig) 
    
# Function to continue (take data)
def continue_callback(event):
    global continue_script
    continue_script = True
    button_continue.ax.set_visible(False)

# Connect the events to their handlers
fig.canvas.mpl_connect('key_press_event', on_key)

# Placeholder arrays for initialization (makes plotting faster)
placeholder_frame = np.zeros((1088, 1456), dtype=np.uint8)  # Full frame size
placeholder_cropped = np.zeros((img_size, img_size), dtype=np.uint8)  # Cropped size

# Initialize the plots
full_frame_plot = axes[0].imshow(placeholder_frame, vmin=0, vmax=255)  # Full frame plot
cropped_frame_plot = axes[1].imshow(placeholder_cropped, vmin=0, vmax=255)  # Cropped frame plot

# Add buttons below the figure
button_ax_continue = fig.add_axes([0.55, 0.02, 0.15, 0.05]) # [left, bottom, width, height]
button_continue = Button(button_ax_continue, 'Continue')
button_continue.on_clicked(continue_callback)

button_ax_abort = fig.add_axes([0.3, 0.02, 0.15, 0.05]) 
button_abort = Button(button_ax_abort, 'Abort')
button_abort.on_clicked(abort_callback)

# Main preview loop
while not (abort_script or continue_script):
    # print(camera.capture_metadata())
    # Capture frames
    frame = camera.capture_array()  # Entire region
    cropped_frame = frame[crop_start_y:crop_start_y+img_size, crop_start_x:crop_start_x+img_size]  # Cropped region

    # Update plot data without clearing
    full_frame_plot.set_data(frame)
    cropped_frame_plot.set_data(cropped_frame)
    
    # Add rectangle to show crop region
    for patch in list(axes[0].patches): # Clear all patches before adding a new one
        patch.remove()  # Remove each patch from the axis
    rectangle = patches.Rectangle((crop_start_x, crop_start_y),img_size, img_size, linewidth=2, edgecolor='red', facecolor='none')
    axes[0].add_patch(rectangle)  # Add rectangle to full frame view

    # Rescale axes to fit data
    axes[0].autoscale()
    axes[1].autoscale()

    plt.pause(0.1)  # Short pause for smoother updates
    time.sleep(0.1) # Helps stabalize camera 

# Disconnect the events to their handlers now we are done with preview
fig.canvas.mpl_disconnect('key_press_event')

if abort_script:
    print('Script aborted...')
elif continue_script:
    
    #########################################################################################################
    # Start taking images now that sample is aligned

    # Take a brightfield image (or darkfield if chosen)
    brightfield = camera.capture_array() # uint8 RGB numpy array
    brightfield_pil = Image.fromarray(brightfield).convert('L') # Grayscale pillow image
    brightfield_pil.save(os.path.join(data_folder,'brightfield.png'), format='PNG') # Save as png

    # Define the data grid (single large grayscale image for visualization)
    downsampled_size = img_size // 15  # Each image should be this small
    data_grid = np.zeros((15 * downsampled_size, 15 * downsampled_size), dtype=np.uint8)

    # Update main figure to indicate FPM process has begun
    # Axis 0 will be now be brightfield image, axis 1 will be data grid which will fill in as we take images
    axes[0].cla()
    axes[1].cla()
    axes[0].set_aspect('equal') 
    axes[1].set_aspect('equal')
    axes[0].set_title("Brightfield")
    axes[1].set_title("Data Grid")

    axes[0].imshow(brightfield,cmap='gray')
    data_grid_display = axes[1].imshow(data_grid, cmap='gray', vmin=0, vmax=255)

    # Refresh
    plt.draw()   
    plt.pause(0.1)  

    # Take FPM images
    camera.set_controls({"ExposureTime": fpm_exposure})

    for i in range(num_images):
        if abort_script:
            print('Script aborted...')
            break
        
        led_matrix.show_pixel(x_coords[i], y_coords[i], brightness=1, color=led_color)
        if i == 0:
            time.sleep(0.5)  # Only need on first iteration 
        
        # Different darkfield exposure time
        if i == df_start and df_exposure_on:
            camera.set_controls({'ExposureTime': df_exposure})
            
        image = camera.capture_array() # uint8 RGB numpy array
        image_pil = Image.fromarray(image).convert('L') # Grayscale pillow image
        img_path = os.path.join(data_folder, f'image_{i}.png') # Create path name
        image_pil.save(img_path, format='PNG') # Save as png 
        
        # Downsample image for the data grid
        downsampled = np.array(image_pil.resize((downsampled_size, downsampled_size)))

        # Determine position in the 15x15 grid
        row = y_coords[i] * downsampled_size  # Ensure correct indexing
        col = (x_coords[i]-1) * downsampled_size

        # Insert into the data grid
        if 0 <= row < data_grid.shape[0] and 0 <= col < data_grid.shape[1]:
            data_grid[row:row + downsampled_size, col:col + downsampled_size] = downsampled

        # Update UI using set_data
        data_grid_display.set_data(data_grid)  
        plt.pause(0.01) 

        # Status message
        progress = int((i+1)/num_images * 100)
        sys.stdout.write(f'\r Image Gathering Progress: {progress}%... ') # Write to same line
        sys.stdout.flush()

    if not abort_script:
        print('\n Image Gathering Done!')
        plt.imsave('data/data_grids/recent.png',data_grid,cmap='gray')

# Turn off LED matrix and camera             
led_matrix.off()
camera.stop()
camera.close()