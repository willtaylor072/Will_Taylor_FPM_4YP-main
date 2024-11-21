import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from PIL import Image
import sys
import numpy as np
import time
from RPiLedMatrix import RPiLedMatrix
from picamera2 import Picamera2
import gpiod

# Custom functions
import fpm_functions as fpm 

##########################################################################################################
# Key setup variables

# Folders
data_folder = 'data/recent' # For saving data images (diagnostics only)
results_folder = 'results/recent' # For saving results

# Imaging parameters
grid_size = 5 # 1->16, recommend 4-8 for stability, time and performance balacnce
img_size = 300 # 100-300 is sensible for square images (any bigger and reconstruction will be slow)
preview_exposure = 30000 # In microseconds for preview
brightfield_exposure = 50000  # In microseconds for brightfield
fpm_exposure = 300000  # In microseconds for FPM

# Set parameters for reconstruction algorithm
options = {
    'max_iter': 5, # Number of iterations
    'alpha': 6, # Regularisation parameter, <10, DOES make a difference, 5 seems good for most cases
    'beta': 1, # Regularisation parameter, >0, not important
    'plot_mode': 0, # 0, off; 1, plot every image; 2, plot every iteration
    'quality_threshold': 0, # Will only use images with dynamic range greater than this (set to 0 to use all images)
    'moderator_on': False, # Will reduce impact on object update from low information images (helps stability)
    'LED_correction': False, # Adjust kx and ky values to their optimal positions
}

# Optical system parameters
LED2SAMPLE = 54 # Distance from LED array to the sample, 54mm (larger distance leads to closer overlapping Fourier circles, optimal 40-60%)
LED_P = 3.3 # LED pitch, mm
N_GLASS = 1.52 # Glass refractive index
NA = 0.1 # Objective numerical apature
PIX_SIZE = 1.09e-6 # Pixel size on object plane, m, 1.09um for 3D printed microscope (directly measured with USAF slide)
WLENGTH = 550e-9 # Central wavelength of LED light, m

###########################################################################################################
# Setup and misc

# Turn on RPI fan
# Set the chip and line (pin number on the chip)
chip_name = "gpiochip4"
line_offset = 45  # This corresponds to GPIO pin 45
# Open the GPIO chip
chip = gpiod.Chip(chip_name)
line = chip.get_line(line_offset)
# Request the line as an output with a value of 0 (low)
line.request(consumer="my_gpio_control", type=gpiod.LINE_REQ_DIR_OUT)
# Set the GPIO pin to low (0) or high (1)
line.set_value(0)  # 0 for LOW, 1 for HIGH
# Release the line when done
line.release()

# Clean up camera and LED array (in case they were left running)
led_matrix = RPiLedMatrix()
led_matrix.off()
camera = Picamera2()
camera.stop()
camera.close()

#######################################################################################################################
# Image preview and alignment process

# Set up image gathering stage
num_images = grid_size**2 # Total number of FPM images
crop_start_x = int(1456/2 - img_size/2) # These crop values ensure images are in center of camera FOV
crop_start_y = int(1088/2 - img_size/2)

# Initialise LED array
led_matrix = RPiLedMatrix()
led_matrix.set_rotation(135) # Ensure 0,0 is bottom left pixel

# Initialize camera
camera = Picamera2()
still_config = camera.create_still_configuration(
    main={'size': (1456,1088)}, controls={"AnalogueGain": 1, 'ExposureTime': preview_exposure} # Need to use whole region then crop (otherwise we lose resolution)
)
camera.configure(still_config)
camera.start()

# Preview alignment using matplotlib
print("Align your sample mechanically or move region with key arrows. Press ENTER when ready.")
led_matrix.show_circle(radius=2, color='white', brightness=1)  # Turn on brightfield LEDs
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

# Set up interactive mode
plt.ion()
fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Figure size (10x5)

# Adjust layout and axis sizes
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.3)
axes[0].set_aspect(1456 / 1088)  # Aspect ratio for the full frame
axes[1].set_aspect('equal')  # Aspect ratio for the cropped frame is square

# Connect the events to their handlers
fig.canvas.mpl_connect('key_press_event', on_key)
fig.canvas.mpl_connect('close_event', on_close)

# Placeholder arrays for initialization (makes plotting faster)
placeholder_frame = np.zeros((1088, 1456), dtype=np.uint8)  # Full frame size
placeholder_cropped = np.zeros((img_size, img_size), dtype=np.uint8)  # Cropped size

# Initialize the plots
full_frame_plot = axes[0].imshow(placeholder_frame, vmin=0, vmax=255)  # Full frame plot
cropped_frame_plot = axes[1].imshow(placeholder_cropped, vmin=0, vmax=255)  # Cropped frame plot

# Set plot titles
axes[0].set_title("Full Camera FOV")
axes[1].set_title("Cropped Camera Region")

# Main preview loop
while not (quit_preview or plot_closed):
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

    plt.pause(0.05)  # Short pause for smoother updates

plt.ioff() # Turn off interactive mode but leave figure open while waiting

#########################################################################################################
# End preview, start taking images

# Take a brightfield image (already have brightfield LEDs on)
camera.set_controls({"ExposureTime": brightfield_exposure})
brightfield = camera.capture_array()[crop_start_y:crop_start_y+img_size,crop_start_x:crop_start_x+img_size] # img_size x img_size RGB image
brightfield_pil = Image.fromarray(brightfield).convert('L') # Grayscale pillow image
brightfield_pil.save(os.path.join(data_folder,'brightfield.png'), format='PNG') # Save as png
brightfield = np.array(brightfield_pil) # Keep as array

# Take FPM images
images = np.zeros((img_size,img_size,num_images))  # np array to store grayscale arrays
x_coords,y_coords = fpm.LED_spiral(grid_size, x_offset=0, y_offset=1) # LED sequence (offset is important to align central LED to optical axis)
camera.set_controls({"ExposureTime": fpm_exposure})

for i in range(num_images):
    led_matrix.show_pixel(x_coords[i], y_coords[i], brightness=1)
    time.sleep(0.2)  # Short pause for LED
    image = camera.capture_array()[crop_start_y:crop_start_y+img_size,crop_start_x:crop_start_x+img_size] # img_size x img_size RGB image
    image_pil = Image.fromarray(image).convert('L') # Grayscale pillow image
    img_path = os.path.join(data_folder, f'image_{i}.png') # Create path name
    image_pil.save(img_path, format='PNG') # Save as png 
    
    image = np.array(image_pil) # Convert to array for reconstruction
    images[:,:,i] = image # Insert into images array
    
    # Status message
    progress = int((i+1)/num_images * 100)
    sys.stdout.write(f'\r Image Gathering Progress: {progress}%') # Write to same line
    sys.stdout.flush()

print('\n Image Gathering Done!')

# Turn off LED matrix and camera             
led_matrix.off()
camera.stop()
camera.close()

####################################################################################################################
# Reconstruction

# Derived variables
F_CUTOFF = 2*NA/WLENGTH # Highest spatial frequency we can resolve in the optical system due to diffraction, lp/m
F_SAMPLING = 1/PIX_SIZE # Sampling frequency (based on sensor pixel size and magnification), lp/m
x_abs = (x_coords - x_coords[0])*LED_P # x distances of LEDs from center LED
y_abs = (y_coords - y_coords[0])*LED_P # y distances of LEDs from center LED

# Size of object image (for given parameters upsampling is between 2 and 5 depending on grid_size)
# Can do seperately x and y if image is not square
obj_size = fpm.calculate_object_size(img_size, grid_size, LED2SAMPLE, LED_P, NA, WLENGTH, PIX_SIZE)
print(f'Upsampling ratio: {obj_size/img_size}; Reconstructed Pixel Size: {int(1e9*PIX_SIZE/(obj_size/img_size))}nm')

# Initial pupil function (binary mask)
# Nyquist sampling criterion: sampling_ratio >2 -> oversampling, sampling_ratio <2 -> undersampling (aliasing may occur)
sampling_ratio = F_SAMPLING / F_CUTOFF 
# x,y is our normalised frequency domain for the images, cutoff frequency = 1 (both x and y)
x,y = np.meshgrid(np.linspace(-sampling_ratio,sampling_ratio,img_size), np.linspace(-sampling_ratio,sampling_ratio,img_size))
theta,r = np.arctan2(y,x), np.sqrt(x**2 + y**2) # Polar coordinates
pupil_radius = (1/sampling_ratio) * (img_size/2) # For plotting and diagnostics
pupil = r<1 # Binary mask for frequencies below cutoff frequency (higher frequencies cannot be resolved due to diffraction)

# Initial object estimate (using first image)
img = np.sqrt(images[:,:,0]) # Amplitude of central image
F_img = fpm.FT(img) # Fourier transformed image (with shift)
F_img = F_img * pupil # Apply pupil function
pad_width = int((obj_size - img_size) / 2) # Padding to make correct size
obj = np.pad(F_img,pad_width,'constant',constant_values=0) # Initial object in frequency domain

# Reconstruction with calculated kx and ky (quickstart)
kx,ky = fpm.calculate_fourier_positions(x_abs, y_abs, LED2SAMPLE, WLENGTH, PIX_SIZE, img_size)
rec_obj,rec_pupil,kx_updated,ky_updated = fpm.reconstruct(images, kx, ky, obj, pupil, options)
# np.save(os.path.join(data_path,'kx_updated'),kx_updated)
# np.save(os.path.join(data_path,'ky_updated'),ky_updated)

# # Reconstruction with loaded / optimal kx and ky (optimal values depend on crop size)
# kx = np.load(os.path.join(data_path,'kx_updated.npy'))
# ky = np.load(os.path.join(data_path,'ky_updated.npy'))
# rec_obj,rec_pupil,_,_ = reconstruct(images, kx, ky, obj, pupil, options)

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(15,3))

contrast = 3 # Contrast for magnitude plot

obj_mag = np.abs(rec_obj)**contrast # Magnitude 
obj_mag = obj_mag / np.max(np.abs(obj_mag)) # Normalise
obj_arg = np.angle(rec_obj) # Phase
obj_arg = obj_arg / np.max(np.abs(obj_arg)) # Normalise

axes[0].imshow(obj_mag,cmap='gray')
axes[0].set_title('Object magnitude')
axes[1].imshow(obj_arg, cmap='gray')
axes[1].set_title('Object phase')
axes[2].imshow(brightfield, cmap='gray')
axes[2].set_title('Brightfield image')

plt.show()

# Save results
obj_mag = (obj_mag * 255).astype(np.uint8)
obj_mag = Image.fromarray(obj_mag)
obj_mag.save('results/recent/magnitude.png')

obj_arg = plt.cm.gray(obj_arg)  # Necessary for saving image
obj_arg = (obj_arg * 255).astype(np.uint8)
obj_arg = Image.fromarray(obj_arg)
obj_arg.save('results/recent/phase.png',format='PNG')

brightfield_pil.save(os.path.join(results_folder,'brightfield.png'), format='PNG') # Also save a copy to the results
