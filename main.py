import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

# This script will always do image gathering, and then can optionally do reconstruction as well
# For full frame use gather_data_NEW.py then reconstruct_NEW.py - recommended

##########################################################################################################
# Key setup variables

# Folders
data_folder = 'data/recent' # For saving data images
results_folder = 'results/recent' # For saving results

reconstruction = False # Do reconstruction after gathering images

# Imaging parameters
grid_size = 15 # Entire LED array is 16x16 but due to misalignment we will only use 15x15
img_size = 300 # 100-300 is sensible for square images (any bigger and reconstruction will be slow)
brightfield_preview = True # Preview bright or darkfield
preview_exposure = int(60e3) if brightfield_preview else int(500e3) # In microseconds for preview
fpm_exposure = int(600e3)  # In microseconds for FPM image capture, 300-600ms
led_color = 'green' # Illumination color
WLENGTH = 550e-9 # Central wavelength of LED light, m, 550nm for white, 630nm for red, 460nm for blue
x_coords,y_coords = fpm.LED_spiral(grid_size,x_offset=1,y_offset=0) # LED sequence (ensure first LED is aligned with optical axis)

# Set parameters for reconstruction algorithm
options = {
    'max_iter': 8, # Number of iterations
    'alpha': 1, # Regularisation parameter for object update
    'beta': 1, # Regularisation parameter for pupil update
    'plot_mode': 1, # 0, plot only at end; 1, plot every iteration
    'LED_correction': 0, # 0, off; 1, accurate; 2, fast. Update wavevectors during reconstruction 
    'update_method': 2, # 1, PIE; 2, ePIE; 3, rPIE. Update method, ePIE reccomended
    'momentum': False, # Use momentum on alpha and beta (tuned for ePIE only)
    'intensity_correction': True, # Adjust image intensity to account for LED variation
}

# Optical system parameters
LED2SAMPLE = 50 # Distance from LED array to the sample, (shorter distance leads to greater overlap of adjacent spectra)
LED_P = 3.3 # LED pitch, mm
NA = 0.1 # Objective numerical aperture
PIX_SIZE = 1150e-9 # Pixel size on object plane, m. Directly measured for V3
x_initial = y_initial = 0 # Offset of first LED to optical axis (can be tuned slightly if reconstruction has crease like artefacts)

# Image gathering parameters
num_images = grid_size**2 # Total number of FPM images
crop_start_x = int(1456/2 - img_size/2) # These crop values ensure images are in center of camera FOV
crop_start_y = int(1088/2 - img_size/2)

# Tuning of LED2SAMPLE, as well as x_initial and y_initial can make a big difference (other parameters not so much)
# Also ensure the pixel size is correctly measured (can use quality_testing with USAF target, or info in README.txt)

## Miscelaneous 

# Turn on fan
def fan_on():
    # Set the chip and line (pin number on the chip)
    chip_name = "gpiochip4"
    line_offset = 45  # This corresponds to GPIO pin 45
    # Open the GPIO chip
    chip = gpiod.Chip(chip_name)
    line = chip.get_line(line_offset)
    # Request the line as an output with a value of 0 (low)
    line.request(consumer="my_gpio_control", type=gpiod.LINE_REQ_DIR_OUT)
    # Set the GPIO pin
    line.set_value(0)  # 0 for on, 1 for on
    # Release the line when done
    line.release()

# Clean up camera and LED array (in case they were left running)
def cleanup():
    led_matrix = RPiLedMatrix()
    led_matrix.off()
    camera = Picamera2()
    camera.stop()
    camera.close()

#######################################################################################################################
# Image preview and alignment process

# Set up figure - this will be the main UI for the entire process
plt.ion() # Allow live plotting
fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Figure size (10x5)

# Axis 0 will be full FOV, axis 1 will be cropped region
axes[0].set_aspect(1456 / 1088)  # Aspect ratio for the full frame
axes[1].set_aspect('equal')  # Aspect ratio for the cropped frame is square
axes[0].set_title("Full Camera FOV")
axes[1].set_title("Cropped Camera Region")

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

# Trying to stabalize images for preview...
camera.set_controls({ 
    "AwbEnable": False,  
    "NoiseReductionMode": 1, # Might be interesting to play with 
})

# Preview alignment using matplotlib
print("Align your sample mechanically or move region with key arrows. Press ENTER when ready.")
if brightfield_preview:
    led_matrix.show_circle(radius=2, color='white', brightness=1)  # Turn on brightfield LEDs
else:
    led_matrix.show_circle(radius=2, color='black', outside_color='white') # Darkfield preview
quit_preview = False
plot_closed = False

# Handles quit by 'enter' and arrow key crop adjustments
def on_key(event):
    global quit_preview, crop_start_x, crop_start_y
    match event.key:
        case 'enter': 
            quit_preview = True
        case 'up': 
            crop_start_y -= 1
        case 'down': 
            crop_start_y += 1
        case 'left': 
            crop_start_x -= 1
        case 'right':
            crop_start_x += 1
    
# Handles closing figure (by clicking the x)
def on_close(event):
    global plot_closed
    plot_closed = True  # Set flag when the plot window is closed

# Connect the events to their handlers
fig.canvas.mpl_connect('key_press_event', on_key)
fig.canvas.mpl_connect('close_event', on_close)

# Placeholder arrays for initialization (makes plotting faster)
placeholder_frame = np.zeros((1088, 1456), dtype=np.uint8)  # Full frame size
placeholder_cropped = np.zeros((img_size, img_size), dtype=np.uint8)  # Cropped size

# Initialize the plots
full_frame_plot = axes[0].imshow(placeholder_frame, vmin=0, vmax=255)  # Full frame plot
cropped_frame_plot = axes[1].imshow(placeholder_cropped, vmin=0, vmax=255)  # Cropped frame plot

# Main preview loop
while not (quit_preview or plot_closed):
    # print(camera.capture_metadata())
    # Capture frames
    frame = camera.capture_array()  # Entire region (useful for aligning sample)
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
fig.canvas.mpl_disconnect('close_event')

#########################################################################################################
# Start taking images now that sample is aligned

# Take a brightfield image (or darkfield if chosen)
brightfield = camera.capture_array()[crop_start_y:crop_start_y+img_size,crop_start_x:crop_start_x+img_size] # img_size x img_size RGB image
brightfield_pil = Image.fromarray(brightfield).convert('L') # Grayscale pillow image
brightfield_pil.save(os.path.join(data_folder,'brightfield.png'), format='PNG') # Save as png
brightfield = np.array(brightfield_pil) # Keep as array

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
images = np.zeros((img_size,img_size,num_images))  # np array to store grayscale arrays
camera.set_controls({"ExposureTime": fpm_exposure})

for i in range(num_images):
    led_matrix.show_pixel(x_coords[i], y_coords[i], brightness=1, color=led_color)
    if i == 0:
        time.sleep(0.5)  # Only need on first iteration 
    image = camera.capture_array()[crop_start_y:crop_start_y+img_size,crop_start_x:crop_start_x+img_size] # img_size x img_size RGB image
    image_pil = Image.fromarray(image).convert('L') # Grayscale pillow image
    img_path = os.path.join(data_folder, f'image_{i}.png') # Create path name
    image_pil.save(img_path, format='PNG') # Save as png 
    
    image = np.array(image_pil) # Convert to array for reconstruction
    images[:,:,i] = image # Insert into images array
    
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
    sys.stdout.write(f'\r Image Gathering Progress: {progress}%') # Write to same line
    sys.stdout.flush()

print('\n Image Gathering Done!')
plt.imsave('data/data_grids/recent.png',data_grid,cmap='gray')

# Turn off LED matrix and camera             
led_matrix.off()
camera.stop()
camera.close()

####################################################################################################################
# Reconstruction

if reconstruction:
    axes[0].imshow(data_grid,cmap='gray') # Move data grid to left axis as we will use right axis to show reconstruction
    axes[0].set_title('Data Grid')
    
    # Derived variables
    F_CUTOFF = 2*NA/WLENGTH # Highest spatial frequency we can resolve in the optical system due to diffraction, lp/m
    F_SAMPLING = 1/PIX_SIZE # Sampling frequency (based on sensor pixel size and magnification), lp/m
    # Nyquist sampling criterion: sampling_ratio >2 -> oversampling (good), sampling_ratio <2 -> undersampling (aliasing may occur)
    SAMPLING_RATIO = F_SAMPLING / F_CUTOFF # Ensure above 2
    # print(f'Sampling ratio: {SAMPLING_RATIO}')
    sampling_size = 1/(img_size*PIX_SIZE) # Distance between discrete points in the Fourier domain (used to scale wavevectors for indexing)
    x_abs = (x_coords - x_coords[0]) * LED_P + x_initial # x distances of LEDs from first LED and optical axis, mm
    y_abs = (y_coords - y_coords[0]) * LED_P + y_initial # y distances

    # Size of reconstructed image (for given parameters upsampling is between 2 and 5 depending on grid_size)
    # Can do seperately x and y if image is not square
    # upsampling_ratio = fpm.calculate_upsampling_ratio(img_size, grid_size, LED2SAMPLE, LED_P, NA, WLENGTH, sampling_size)
    upsampling_ratio = 5 # Or can use a set value
    obj_size = upsampling_ratio * img_size
    print(f'Upsampling ratio: {upsampling_ratio}; Reconstructed Pixel Size: {int(1e9*PIX_SIZE/(upsampling_ratio))}nm')

    # LED wavevectors - scaled for use in Fourier domain. To get true wavevectors multiply by sampling size
    kx,ky = fpm.calculate_wavevectors(x_abs, y_abs, LED2SAMPLE, WLENGTH, sampling_size)

    # Initial pupil function (binary mask)
    # x,y is our normalised frequency domain for the images, cutoff frequency = 1 (both x and y)
    x,y = np.meshgrid(np.linspace(-SAMPLING_RATIO,SAMPLING_RATIO,img_size), np.linspace(-SAMPLING_RATIO,SAMPLING_RATIO,img_size))
    theta,r = np.arctan2(y,x), np.sqrt(x**2 + y**2) # Polar coordinates
    pupil_radius = (1/SAMPLING_RATIO) * (img_size/2) # In pixels
    pupil_binary = r<1 # Binary mask for frequencies below cutoff frequency (higher frequencies cannot be resolved due to diffraction)

    # Initial object estimate (using first image)
    img = np.sqrt(images[:,:,0]) # Amplitude of central image
    F_img = fpm.FT(img) # Fourier transformed image (with shift)
    F_img = F_img * pupil_binary # Apply pupil function
    pad_width = int((obj_size - img_size) / 2) # Padding to make correct size
    obj = np.pad(F_img,pad_width,'constant',constant_values=0) # Initial object in frequency domain

    # Do reconstruction (main code)
    rec_obj,rec_pupil,kx_updated,ky_updated = fpm.reconstruct(images, kx, ky, obj, pupil_binary, options, fig, axes, pupil=None)

    # Keep plot open
    plt.ioff()
    plt.show() 

    ###########################################################################################
    # Save results

    # Recovered object
    obj_mag = np.abs(rec_obj) # Magnitude
    obj_arg = np.angle(rec_obj) # Phase

    # Recovered pupil
    pupil_mag = np.abs(rec_pupil)
    pupil_arg = np.angle(rec_pupil)

    obj_mag = obj_mag / np.max(np.abs(obj_mag))
    obj_mag = (obj_mag * 255).astype(np.uint8)
    obj_mag = Image.fromarray(obj_mag)
    obj_mag.save('results/recent/magnitude.png')

    obj_arg = plt.cm.hot(obj_arg)  # Use colormap
    obj_arg = (obj_arg * 255).astype(np.uint8)
    obj_arg = Image.fromarray(obj_arg)
    obj_arg.save('results/recent/phase.png',format='PNG')

    # Brightfield (or darkfield)
    bf = Image.open(os.path.join(data_folder,'brightfield.png'))
    bf.save('results/recent/brightfield.png')