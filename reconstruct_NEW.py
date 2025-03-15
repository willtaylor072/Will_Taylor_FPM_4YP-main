import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import os
from PIL import Image
import sys
import numpy as np
import importlib 

# Custom functions
import fpm_functions as fpm 
importlib.reload(fpm) # Reload

# Reconstruction for FPM datasets - full frame only (use reconstruct_OLD.ipynb for cropped datasets)

##########################################################################################################

# Folders
data_folder = 'data/library/full_frame' # Get data from here
results_folder = 'results/recent' # Save results here

# Setup
grid_size = 15
img_size = 200 # Individual reconstruction size, 100-300

# If True, reconstruct entire frame by stitching together multiple reconstructions
# If False, we will select a small tile to reconstruct
full_reconstruction = False 

# Set parameters for reconstruction algorithm
options = {
    'max_iter': 5, # Number of iterations
    'alpha': 1, # Regularisation parameter for object update
    'beta': 1, # Regularisation parameter for pupil update
    'plot_mode': 1, # 0, plot only at end; 1, plot every iteration
    'LED_correction': 0, # 0, off; 1, accurate; 2, fast; 3, first iteration only
    'update_method': 2, # 1, PIE; 2, ePIE; 3, rPIE. Update method, ePIE reccomended
    'momentum': False, # Use momentum on alpha and beta (tuned for ePIE only)
    'intensity_correction': True, # Adjust image intensity to account for LED variation
    'plot_magnitude': True, # Plot magnitude or phase
}

# Optical system parameters
LED2SAMPLE = 53
x_initial = 0.5
y_initial = -0.5
LED_P = 3.3
NA = 0.1
PIX_SIZE = 850e-9 # 1150 for 3x, 725 for new (measured), 850 for 4x (expected)
WLENGTH = 550e-9

# LED sequence    
x_coords, y_coords = fpm.LED_spiral(grid_size)
x_abs = (x_coords - x_coords[0]) * LED_P + x_initial # x distances of LEDs from optical axis, mm
y_abs = (y_coords - y_coords[0]) * LED_P + y_initial # y distances 

brightfield = np.array(Image.open(os.path.join(data_folder,'brightfield.png')))/255
x_lim = brightfield.shape[1]
y_lim = brightfield.shape[0]
num_images = grid_size**2

###############

# Set up preview for region selection
plt.ion() # Allow live plotting
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('Fourier Ptychography - Reconstruction')

crop_start_x = int(x_lim/2 - img_size/2)
crop_start_y = int(y_lim/2 - img_size/2)

# Axis 0 will be full brightfield, axis 1 will be cropped region to reconstruct
axes[0].set_aspect(x_lim / y_lim)  # Aspect ratio for the full frame
axes[1].set_aspect('equal')  # Aspect ratio for the cropped frame is square
axes[0].set_title('Use arrow keys to move region, w/e to resize')
abort_script = False
do_reconstruction = False
v_y=v_x=0 # For moving preview frame

# Handles arrow key crop adjustments
def on_key(event):
    global crop_start_x, crop_start_y,v_x,v_y,img_size
    acceleration = 3
    max_speed = 30
    match event.key:
        case 'up': 
            v_y = min(v_y + acceleration, max_speed)
            crop_start_y -= v_y
            crop_start_y = max(0,crop_start_y)
        case 'down': 
            v_y = min(v_y + acceleration, max_speed)
            crop_start_y += v_y
            crop_start_y = min(crop_start_y,y_lim-img_size)
        case 'left': 
            v_x = min(v_x + acceleration, max_speed)
            crop_start_x -= v_x
            crop_start_x = max(crop_start_x,0)
        case 'right':
            v_x = min(v_x + acceleration, max_speed)
            crop_start_x += v_x
            crop_start_x = min(crop_start_x,x_lim-img_size)
        case 'w':
            img_size = min(img_size+5,300)
        case 'e':
            img_size = max(img_size-5,100)

def on_release(event):
    global v_x,v_y
    v_x=v_y=0 # Stop box moving
    
# Callback to abort script
def abort_callback(event):
    global abort_script
    abort_script = True
    plt.close(fig) 
    
# Callback to do reconstruction
def reconstruct_callback(event):
    global do_reconstruction
    do_reconstruction = True
    button_reconstruct.set_active(False)

# Callback to reset process
def reset_callback(event):
    global cropped_frame_plot
    button_reset.set_active(False)
    button_reconstruct.set_active(True)
    # Show the grayscale preview 
    cropped_frame_plot = axes[1].imshow(placeholder_cropped, vmin=0, vmax=1,cmap='gray')
    axes[1].set_title('Cropped brightfield')
    
# Callback to toggle magnitude/phase plot
def toggle_mode_callback(event):
    global options
    options['plot_magnitude'] = not options['plot_magnitude']
    text = 'Plot magnitude' if options['plot_magnitude'] else 'Plot phase'
    button_toggle_mode.label.set_text(text)

# Connect the events to their handlers
fig.canvas.mpl_connect('key_press_event', on_key)
fig.canvas.mpl_connect('key_release_event', on_release)

# Placeholder arrays for initialization (makes plotting faster)
placeholder_frame = np.zeros((1088, 1456), dtype=np.uint8)  # Full frame size
placeholder_cropped = np.zeros((img_size, img_size), dtype=np.uint8)  # Cropped size

# Initialize the plots
full_frame_plot = axes[0].imshow(brightfield, cmap='gray')  # Full frame plot
cropped_frame_plot = axes[1].imshow(placeholder_cropped, vmin=0, vmax=1,cmap='gray')  # Cropped frame plot

# Add buttons below the figure
button_ax_reconstruct = fig.add_axes([0.55, 0.02, 0.15, 0.05]) # [left, bottom, width, height]
button_reconstruct = Button(button_ax_reconstruct, 'Reconstruct')
button_reconstruct.on_clicked(reconstruct_callback)

button_ax_reset = fig.add_axes([0.8, 0.02, 0.15, 0.05])
button_reset = Button(button_ax_reset, 'Reset')
button_reset.on_clicked(reset_callback)
button_reset.set_active(False)

button_ax_toggle_mode = fig.add_axes([0.8, 0.92, 0.15, 0.05]) 
text = 'Plot magnitude' if options['plot_magnitude'] else 'Plot phase'
button_toggle_mode = Button(button_ax_toggle_mode, text)
button_toggle_mode.on_clicked(toggle_mode_callback)

button_ax_abort = fig.add_axes([0.1, 0.02, 0.15, 0.05]) 
button_abort = Button(button_ax_abort, 'Abort')
button_abort.on_clicked(abort_callback)

# Main process loop
while not abort_script:
    # Cropped region of brightfield
    cropped_frame = brightfield[crop_start_y:crop_start_y+img_size, crop_start_x:crop_start_x+img_size]  # Cropped region

    # Update plot data without clearing
    cropped_frame_plot.set_data(cropped_frame)
    
    # Add rectangle to show crop region
    for patch in list(axes[0].patches): # Clear all patches before adding a new one
        patch.remove()  # Remove each patch from the axis
    rectangle = patches.Rectangle((crop_start_x, crop_start_y),img_size, img_size, linewidth=2, edgecolor='red', facecolor='none')
    axes[0].add_patch(rectangle)  # Add rectangle to full frame view

    plt.pause(0.1) # Short delay

    # If we clicked the reconstruct button
    if do_reconstruction:
        do_reconstruction = False
        # Use the crop region specified to form dataset
        images = np.zeros((img_size,img_size,num_images)) # Initialise array for storing images
        brightfield_crop = brightfield[crop_start_y:crop_start_y+img_size,crop_start_x:crop_start_x+img_size]

        # Best to read images one at a time to avoid memory issues (especially for full frame datasets)
        for i in range(num_images): # ~ 2s to load all 256 images into array
            filename = os.path.join(data_folder, f'image_{i}.png') # Construct path
            img = np.array(Image.open(filename),dtype=np.uint8) # Open image as numpy array
            img = img[crop_start_y:crop_start_y+img_size,crop_start_x:crop_start_x+img_size] # Crop
            images[:,:,i] = img
            
        # Derived variables
        F_CUTOFF = 2*NA/WLENGTH # Highest spatial frequency we can resolve in the optical system due to diffraction, lp/m
        F_SAMPLING = 1/PIX_SIZE # Sampling frequency (based on sensor pixel size and magnification), lp/m
        # Nyquist sampling criterion: sampling_ratio >2 -> oversampling (good), sampling_ratio <2 -> undersampling (aliasing may occur)
        SAMPLING_RATIO = F_SAMPLING / F_CUTOFF # Ensure above 2
        # print(f'Sampling ratio: {SAMPLING_RATIO}')
        # img_size * PIX_SIZE is the total object size in spacial domain (~300um)
        sampling_size = 1/(img_size*PIX_SIZE) # Sampling size in the Fourier domain (used to scale wavevectors for indexing)

        # Size of reconstructed object (for given parameters upsampling is between 2 and 5 depending on grid_size)
        # upsampling_ratio = fpm.calculate_upsampling_ratio(img_size, grid_size, LED2SAMPLE, LED_P, NA, WLENGTH, sampling_size)
        upsampling_ratio = 5 # Or can use a set value, recommended 5
        obj_size = upsampling_ratio * img_size

        # LED wavevectors - scaled for indexing in Fourier domain. To get true wavevectors multiply by sampling size * 2pi
        kx,ky = fpm.calculate_wavevectors(x_abs, y_abs, LED2SAMPLE, WLENGTH, sampling_size)

        # Initial pupil function (binary mask)
        # x,y is our normalised frequency domain for the images, cutoff frequency = 1 (both x and y)
        x,y = np.meshgrid(np.linspace(-SAMPLING_RATIO,SAMPLING_RATIO,img_size), np.linspace(-SAMPLING_RATIO,SAMPLING_RATIO,img_size))
        theta,r = np.arctan2(y,x), np.sqrt(x**2 + y**2) # Polar coordinates
        # pupil_radius = (1/SAMPLING_RATIO) * (img_size/2) # In pixels
        # pupil_radius = NA/WLENGTH * img_size * PIX_SIZE
        pupil_binary = r<1 # Binary mask for frequencies below cutoff frequency (higher frequencies cannot be resolved due to diffraction)

        # Initial object estimate (using central image)
        img = np.sqrt(images[:,:,0]) # Amplitude of central image
        F_img = fpm.FT(img) # Fourier transformed image
        F_img = F_img * pupil_binary # Apply pupil function
        pad_width = int((obj_size - img_size) / 2) # Padding to make correct size
        obj = np.pad(F_img,pad_width,'constant',constant_values=0) # Initial object in spacial frequency (Fourier domain)
        
        # Main function for FPM reconstruction
        rec_obj,rec_pupil,kx_updated,ky_updated = fpm.reconstruct(images, kx, ky, obj, pupil_binary, options, fig, axes)    

        # Enable option to repeat process
        button_reset.set_active(True)
        
        