import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from PIL import Image
import sys
import numpy as np
from scipy.fft import fft2, ifft2
import time
from RPiLedMatrix import RPiLedMatrix
from picamera2 import Picamera2
import gpiod

sys.stderr = open(os.devnull, 'w')

##################################################################################################################
# Functions
##################################################################################################################

# Starting at 7,7, generate coordinates to turn on LEDs in a spiral pattern, moving right up left down right up left down....
# 0,0 is bottom left LED when rotation is 135 degrees
def LED_spiral(n):
    # Input: n; gridsize to generate coordinates for
    # Returns: x_coords, y_coords; arrays for LED coordinates 
    
    # Initialize the arrays to store the x and y coordinates
    x_coords = np.zeros(n**2, dtype=int)
    y_coords = np.zeros(n**2, dtype=int)
    
    # Starting point
    x, y = 7, 7
    x_coords[0], y_coords[0] = x, y
    
    step_size = 1  # How far to move in each direction before changing direction
    index = 1  # Tracks the number of coordinates generated so far
    
    while index < n**2:
        # Move right
        for _ in range(step_size):
            if index >= n**2:
                break
            x += 1
            x_coords[index], y_coords[index] = x, y
            index += 1
        
        # Move up
        for _ in range(step_size):
            if index >= n**2:
                break
            y += 1
            x_coords[index], y_coords[index] = x, y
            index += 1
        
        step_size += 1  # Increase step size after moving right and up
        
        # Move left
        for _ in range(step_size):
            if index >= n**2:
                break
            x -= 1
            x_coords[index], y_coords[index] = x, y
            index += 1
        
        # Move down
        for _ in range(step_size):
            if index >= n**2:
                break
            y -= 1
            x_coords[index], y_coords[index] = x, y
            index += 1
        
        step_size += 1  # Increase step size after moving left and down
    
    return x_coords, y_coords

# Find a suitable size of the object
def calculate_object_size(img_size, grid_size, LED2SAMPLE, LED_P, NA, WLENGTH, PIX_SIZE):
    sampling_size = 1/(img_size*PIX_SIZE) # Sampling size in x
    r = np.sqrt(2*(grid_size*LED_P*0.5)**2) # Max radius of LED from center
    led_na = r/(np.sqrt(r**2+LED2SAMPLE**2)) # Max NA of LED
    max_freq = led_na/WLENGTH + NA/WLENGTH # Maximum spacial frequency in x
    
    return np.ceil(2*np.round(2*max_freq/sampling_size)/img_size)*img_size # obj_size will be a multiple of img_size

# Find the LED positions in Fourier domain
def calculate_fourier_positions(x, y, LED2SAMPLE, WLENGTH, PIX_SIZE, img_size):
    sampling_size = 1/(img_size*PIX_SIZE) # Sampling size in x
    kx = np.zeros(len(x))
    ky = np.zeros(len(x))
    for i in range(len(x)):
        sin_thetax = x[i]/(np.sqrt(x[i]**2 + LED2SAMPLE**2))
        kx[i] = sin_thetax/(WLENGTH*sampling_size)
        
        sin_thetay = y[i]/(np.sqrt(y[i]**2 + LED2SAMPLE**2))
        ky[i] = sin_thetay/(WLENGTH*sampling_size)
    
    return kx,ky

# Shifted fourier transform
def FT(x):
    return np.fft.fftshift(fft2(np.fft.ifftshift(x)))

# Shifted inverse fourier transform
def IFT(x):
    return np.fft.fftshift(ifft2(np.fft.ifftshift(x)))

# Update kx and ky for the current image by finding the kx and ky that minimise error between image estimate
# and current low res image. search_range should be odd
def update_LED_positions_accurate(obj,img,kx,ky,img_size,obj_center,image_number,search_range=10):
    
    # Easier to crop obj using x_start and y_start 
    x_start = int(obj_center + kx - img_size//2)
    y_start = int(obj_center - ky - img_size//2)  
    
    # Define the range we will search for a minimum
    x_offsets = range(-(search_range // 2), (search_range // 2) + 1)
    y_offsets = range(-(search_range // 2), (search_range // 2) + 1)

    min_error = float('inf') # Min error for this iteration
    # error_heatmap = np.zeros((search_range,search_range)) # For visualising algorithm
    
    # Find error between image and estimated image, where we offset the object crop region slightly to find estimated image
    for x in x_offsets:
        for y in y_offsets:       
            img_est = IFT(obj[y_start+y:y_start+y+img_size, x_start+x:x_start+x+img_size]) # Estimated image is IFT of cropped spectrum at the shifted center
            error = np.sum(np.abs(img_est - img)**2) # Error between estimated and measured image
            # error_heatmap[(search_range // 2 - y), (x + search_range // 2)] = error # Add error to heatmap (convert from cartesian to image coords)
    
            # Track the smallest error position
            if error < min_error:
                min_error = error
                optimal_x = x # Offsets for mimimum error within the sub-region
                optimal_y = y
            
    # # Plot the heatmap for a specific image number
    # if image_number == 15:  # Image number to inspect correction algorithm
    #     plt.imshow(error_heatmap, cmap='hot', extent=[x_offsets.start, x_offsets.stop - 1, y_offsets.start, y_offsets.stop - 1])
    #     plt.colorbar(label='Error')
    #     plt.xlabel('X Offset')
    #     plt.ylabel('Y Offset')
    #     plt.title(f'Error Landscape, image {image_number}')

    #     # Label the minimum error cell
    #     plt.text(optimal_x, optimal_y, 'X', color='white', fontsize=12, ha='center', va='center', fontweight='bold')
    #     plt.show()
        
    #     # Diagnostics
    #     print(f"optimal_x: {optimal_x}, optimal_y: {optimal_y}, min_error: {min_error}, x_start: {x_start}, y_start: {y_start}")

                    
    # We have optimal x_start and y_start so just rearrange for kx,ky using below relations
    # x_start = int(obj_center + kx[i] - img_size//2)
    # y_start = int(obj_center - ky[i] - img_size//2)  
    kx = x_start - obj_center + img_size//2
    ky = obj_center - img_size//2 - y_start
        
    return kx,ky 

# Plotting for visualising reconstruction (used in reconstruct function)
def plot(axes,fig, obj,x_start,y_start,img_size,obj_center,pupil,kx,ky,i,iter,plot_mode,update_size):
    # Clear previous plots
    for ax in axes:
        ax.cla()  # Clear the current axes
      
    axes[0].imshow(np.log(np.abs(obj) + 1), cmap='gray') # Show with log scale
    axes[0].set_title(f'Spectrum of object: image {i+1}, iteration {iter+1} ')
    if plot_mode == 1:
        square = patches.Rectangle((x_start, y_start), img_size, img_size, linewidth=0.5, edgecolor='red', facecolor='none')
        axes[0].add_patch(square)
        circle = patches.Circle((obj_center+kx[i],obj_center-ky[i]), radius=pupil_radius,linewidth =0.5, edgecolor='red',facecolor='none')
        axes[0].add_patch(circle)

    axes[1].imshow(np.abs(IFT(obj)), cmap='gray')
    axes[1].set_title('Currently reconstructed object')
    
    # axes[2].imshow(np.angle(pupil),cmap='gray')
    # axes[2].set_title('Current pupil phase')
    # axes[2].imshow(np.abs(pupil),cmap='gray')
    # axes[2].set_title('Current pupil magnitude')
    axes[2].plot(update_size)
    axes[2].set_title('Object update size')
    # axes[2].set_ylim(0,1)
    
    # Update the figure
    plt.draw()          # Redraw the current figure
    plt.pause(0.1)      # Pause to allow the figure to update

# Reconstruct object and pupil function using Quasi Newton algorithm
def reconstruct(images, kx, ky, obj, pupil, options):
    # Inputs: 
    # images; low res image array data, in order taken
    # kx,ky; location of LEDs in Fourier domain, in order of images taken
    # obj; initial estimate for object in frequency domain
    # pupil; initial pupil function
    # options; alpha, beta (regularisation), max_iter, plotting, quality_threshold
    
    # Returns: 
    # rec_obj; recovered object
    # rec_pupil; recovered pupil function
    # kx, ky; updated LED positions
    
    # Unpack options
    alpha = options['alpha'] # <10, DOES MAKE DIFFERENCE
    beta = options['beta'] # Not important
    max_iter = options['max_iter'] # Number of iterations to run algorithm (1 iteration uses all images)
    plot_mode = options['plot_mode'] # 0, off; 1, plot every image; 2, plot every iteration
    quality_threshold = options['quality_threshold'] # Used to crop bad images from dataset
    LED_correction = options['LED_correction'] # Correct kx,ky - LED coordinates
    moderator_on = options['moderator_on'] # Increases stability
    
    # Other parameters
    img_size = images.shape[0] # Square, same size as pupil function
    num_images = images.shape[2]
    obj_size = obj.shape[0] # Square
    obj_center = obj_size // 2 # Center of object (used for inserting spectra in correct place)
    pupil_binary = np.copy(pupil) # Original pupil function (binary mask)
    pupil = pupil.astype('complex64') # Pupil function for updating needs to be complex 
    update_size = np.zeros(num_images) # To monitor object update size (can spot instability numerically)

    # For removing low quality images
    if quality_threshold != 0:
        # Initialize empty lists to store good images and their coordinates
        good_images = []
        img_quality = []
        kx_new = []
        ky_new = []
        
        # Loop through each image and check if it meets the quality threshold
        for i in range(num_images):
            img = images[:, :, i] # Image to check
            dynamic_range = (np.max(img) - np.min(img)) * 256
            if dynamic_range >= quality_threshold:
                good_images.append(img)   # Append the good image to the list
                img_quality.append(dynamic_range) # Save the quality metric
                kx_new.append(kx[i])      # Append the corresponding kx coordinate
                ky_new.append(ky[i])      # Append the corresponding ky coordinate
                
        # Convert lists to numpy arrays and override the old variables
        images = np.array(good_images)  # Shape will be (num_good_images, img_size, img_size)
        images = np.transpose(good_images, (1, 2, 0)) # Shape will be (img_size, img_size, num_good_images) as required
        num_images = images.shape[2]
        img_quality = np.array(img_quality)
        kx = np.array(kx_new)
        ky = np.array(ky_new)
        update_size = np.zeros(num_images) # To monitor object update size (can spot instability numerically)
     
    # If plotting, create axis and figure here to save resources
    if plot_mode != 0:
        global pupil_radius 
        fig, axes = plt.subplots(1, 3, figsize=(15,4))
        plt.ion() # If interactive mode is on figures are shown on creation

    # Main loop
    for iter in range(max_iter):
        for i in range(num_images): # For each image in data set   
            x_start = int(obj_center + kx[i] - img_size//2) # For cropping object spectrum
            y_start = int(obj_center - ky[i] - img_size//2)  
            
            # Define variables for object and pupil updating 
             
            # The relevant part of object spectrum to update
            object_update = obj[y_start:y_start+img_size, x_start:x_start+img_size]
             
            # Measured image amplitude
            img = np.sqrt(images[:,:,i])
            
            # Estimated image amplitude from object (complex)
            img_est = IFT(object_update)
            
            # The update image (in Fourier domain) is composed of the magnitude of the measured image, the phase of the estimated image
            # and also the spectrum of the estimated image is subtracted
            update_image = FT(img*np.exp(1j*np.angle(img_est))) - FT(img_est)
            
            # Object update
            numerator = np.abs(pupil) * np.conj(pupil) * update_image
            denominator = np.max(np.abs(pupil)) * (np.abs(pupil)**2 + alpha)
            object_update = numerator / denominator
            if moderator_on:
                moderator = (img_quality[i]/np.max(img_quality)) # Moderate update step based on image quality (provides stability)
            else:
                moderator = 1
            obj[y_start:y_start+img_size, x_start:x_start+img_size] += moderator * object_update # Add to main spectrum
            update_size[i] = np.mean(np.abs(object_update)) # To check instability 
            
            # Pupil update
            numerator = np.abs(object_update) * np.conj(object_update) * update_image * pupil_binary
            denominator = np.max(obj) * (np.abs(object_update)**2 + beta)
            pupil_update = numerator / denominator
            pupil += pupil_update
      
            # LED position (kx,ky) correction for image we just used
            if LED_correction == 1:
                kx_new,ky_new = update_LED_positions_accurate(obj,img,kx[i],ky[i],img_size,obj_center,i)
                kx[i] = kx_new # Updated LED positions
                ky[i] = ky_new
                
                
            # Plot every image
            if plot_mode == 1:
                plot(axes,fig, obj,x_start,y_start,img_size,obj_center,pupil,kx,ky,i,iter,plot_mode,update_size)
                
        # Status message
        progress = int((iter+1)/max_iter * 100)
        sys.stdout.write(f'\r Reconstruction Progress: {progress}%') # Write to same line
        sys.stdout.flush()
        
        # Plot every iteration
        if plot_mode == 2:
            plot(axes,fig, obj,x_start,y_start,img_size,obj_center,pupil,kx,ky,i,iter,plot_mode,update_size)
    
    print("\n Reconstruction Done!") # Move to new line in terminal
    
    # To keep plot open when function is done
    if plot_mode != 0:
        plt.ioff()
        plt.show()

    return IFT(obj),pupil,kx,ky

###########################################################################################################################
# Main script. Handles image acquisition and reconstruction and UI
###########################################################################################################################

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
# Image preview and gatehering

data_folder = 'data/recent' # For saving data images (diagnostics only)
results_folder = 'results/recent' # For saving results

# Set up image gathering stage
grid_size = 4 # 1->16, recommend 4-8 for stability, time and performance balacnce
num_images = grid_size**2 # Total number of FPM images
brightfield_exposure = 50000  # In microseconds for brightfield
fpm_exposure = 400000  # In microseconds for FPM
img_size = 300 # 100-300 is sensible for square images (otherwise reconstruction will be too slow)
crop_start_x = int(1456/2 - img_size/2) # These crop values ensure images are in center of camera FOV
crop_start_y = int(1088/2 - img_size/2)
x_coords,y_coords = LED_spiral(grid_size) # Generates arrays for x and y to turn on LEDs in spiral pattern

# Initialise LED array
led_matrix = RPiLedMatrix()
led_matrix.set_rotation(135) # Ensure 0,0 is bottom left pixel

# Initialize camera
camera = Picamera2()
still_config = camera.create_still_configuration(
    main={'size': (1456,1088)}, controls={"AnalogueGain": 1, 'ExposureTime': 30000} # Need to use whole region then crop (otherwise we lose resolution)
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

# Experiment parameters
LED2SAMPLE = 54 # Distance from LED array to the sample, 54mm (larger distance leads to closer overlapping Fourier circles, optimal 40-60%)
LED_P = 3.3 # LED pitch, mm
N_GLASS = 1.52 # Glass refractive index
NA = 0.1 # Objective numerical apature
PIX_SIZE = 1.09e-6 # Pixel size on object plane, m, 1.09um for 3D printed microscope (directly measured with USAF slide)
WLENGTH = 550e-9 # Central wavelength of LED light, m

# Derived variables
F_CUTOFF = 2*NA/WLENGTH # Highest spatial frequency we can resolve in the optical system due to diffraction, lp/m
F_SAMPLING = 1/PIX_SIZE # Sampling frequency (based on sensor pixel size and magnification), lp/m
x_abs = (x_coords - x_coords[0])*LED_P # x distances of LEDs from center LED
y_abs = (y_coords - y_coords[0])*LED_P # y distances of LEDs from center LED

# Size of object image (for given parameters upsampling is between 2 and 5 depending on grid_size)
# Can do seperately x and y if image is not square
obj_size = calculate_object_size(img_size, grid_size, LED2SAMPLE, LED_P, NA, WLENGTH, PIX_SIZE)
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
F_img = FT(img) # Fourier transformed image (with shift)
F_img = F_img * pupil # Apply pupil function
pad_width = int((obj_size - img_size) / 2) # Padding to make correct size
obj = np.pad(F_img,pad_width,'constant',constant_values=0) # Initial object in frequency domain

# Set parameters for reconstruction algorithm
options = {
    'max_iter': 8, # Number of iterations
    'alpha': 6, # Regularisation parameter, <10, DOES make a difference, 5 seems good for most cases
    'beta': 1, # Regularisation parameter, >0, not important
    'plot_mode': 0, # 0, off; 1, plot every image; 2, plot every iteration
    'quality_threshold': 0, # Will only use images with dynamic range greater than this (set to 0 to use all images)
    'moderator_on': False, # Will reduce impact on object update from low information images (helps stability)
    'LED_correction': 0, # 0, off; 1, accurate; 2, fast (not working)
}

# Reconstruction with calculated kx and ky (quickstart)
kx,ky = calculate_fourier_positions(x_abs, y_abs, LED2SAMPLE, WLENGTH, PIX_SIZE, img_size)
rec_obj,rec_pupil,_,_ = reconstruct(images, kx, ky, obj, pupil, options)

# # Reconstruction with LED correction to find optimal kx and ky
# options = {
#     'max_iter': 1, # Number of iterations
#     'alpha': 5, # Regularisation parameter, <10, DOES make a difference, 5 seems good for most cases
#     'beta': 1, # Regularisation parameter, >0, not important
#     'plot_mode': 0, # 0, off; 1, plot every image; 2, plot every iteration
#     'quality_threshold': 0, # Will only use images with dynamic range greater than this (set to 0 to use all images)
#     'moderator_on': False, # Will reduce impact on object update from low information images (helps stability)
#     'LED_correction': 1, # 0, off; 1, accurate; 2, fast (not working)
# }
# kx,ky = calculate_fourier_positions(x_abs, y_abs, LED2SAMPLE, WLENGTH, PIX_SIZE, img_size) # Initial guess for kx,ky
# rec_obj,rec_pupil,kx_updated,ky_updated = reconstruct(images, kx, ky, obj, pupil, options)
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
