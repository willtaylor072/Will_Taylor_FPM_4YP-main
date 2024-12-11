# Functions for FPM process

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import clear_output, display
import os
import sys
from PIL import Image
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
    
# Generate coordinates to turn on LEDs in a spiral pattern, moving right up left down right up left down....
# 0,0 is bottom left LED when rotation is 135 degrees. Can use offsets to center the starting point with optical axis. 
# N.b. If offsets are non zero, we can't use entire 16x16 LED array - reccomended not to use offsets but just adjust wavevectors.
def LED_spiral(n, x_offset=0, y_offset=0):
    
    # Initialize the arrays to store the x and y coordinates
    x_coords = np.zeros(n**2, dtype=int)
    y_coords = np.zeros(n**2, dtype=int)
    
    # Starting point.
    x, y = 7+x_offset, 7+y_offset
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
def calculate_upsampling_ratio(img_size, grid_size, LED2SAMPLE, LED_P, NA, WLENGTH, sampling_size):
    
    r = np.sqrt(2*(grid_size*LED_P*0.5)**2) # Max radius of LED from center
    led_na = r/(np.sqrt(r**2+LED2SAMPLE**2)) # Max NA of LED
    max_freq = led_na/WLENGTH + NA/WLENGTH # Maximum spacial frequency in x
    
    return np.ceil(2*np.round(2*max_freq/sampling_size)/img_size) # Upsampling ratio

# Find the scaled LED wavevectors (for placing each low res image in Fourier domain)
def calculate_wavevectors(x, y, LED2SAMPLE, WLENGTH, sampling_size):
    
    kx = np.zeros(len(x)) # x components of wavevector for each LED illumination
    ky = np.zeros(len(x)) # y components
    for i in range(len(x)):
        # Actual wavevectors are sin_theta/wavelength, but we calculate the scaled ones 
        # so we can use them for indexing in the Fourier domain. 
        sin_thetax = x[i]/(np.sqrt(x[i]**2 + LED2SAMPLE**2))
        kx[i] = sin_thetax/(WLENGTH*sampling_size) # Scaled wavevector is true wavevector / sampling_size
        
        sin_thetay = y[i]/(np.sqrt(y[i]**2 + LED2SAMPLE**2))
        ky[i] = sin_thetay/(WLENGTH*sampling_size)
    
    return kx,ky

# Shifted fourier transform
def FT(x):
    return fftshift(fft2(x))

# Shifted inverse fourier transform
def IFT(x):
    return ifft2(ifftshift(x))

# Plotting for visualising reconstruction (regular python version, 1 axis to plot on)
def plot_py(fig,axes,obj):
    # We use axes[1] to show object (axis[0] is for brightfield reference)
    axes[1].cla()  # Clear the current axes
    axes[1].imshow(np.abs(IFT(obj)), cmap='gray')
    axes[1].set_title('Currently reconstructed object')
    
    # Update the figure
    plt.draw()   
    plt.pause(0.1)     
    
# Plotting for visualising reconstruction (notebook version)
def plot_ipynb(fig,axes,obj,x_start,y_start,img_size,obj_center,pupil,kx,ky,i,iter,plot_mode,update_size,quality):
    # Clear previous plots
    for ax in axes:
        ax.cla()  # Clear the current axes

    # Show spectrum
    axes[0].imshow(np.log(np.abs(obj) + 1), cmap='gray') # Show with log scale
    axes[0].set_title(f'Spectrum of object: image {i+1}, iteration {iter+1} ')
    if plot_mode == 2: # Plot every image
        square = patches.Rectangle((x_start, y_start), img_size, img_size, linewidth=0.5, edgecolor='red', facecolor='none')
        axes[0].add_patch(square)
        # circle = patches.Circle((obj_center+kx[i],obj_center-ky[i]), radius=pupil_radius,linewidth =0.5, edgecolor='red',facecolor='none')
        # axes[0].add_patch(circle) # Need to pass pupil radius

    # Show reconstructed image
    axes[1].imshow(np.abs(IFT(obj)), cmap='gray')
    axes[1].set_title('Currently reconstructed object')
    
    # Show pupil
    # axes[2].imshow(np.angle(pupil),cmap='gray')
    # axes[2].set_title('Current pupil phase')
    # axes[2].imshow(np.abs(pupil),cmap='gray')
    # axes[2].set_title('Current pupil magnitude')
    
    # Show update size
    axes[2].plot(update_size)
    axes[2].set_title('Object update size')
    
    # Show quality of reconstruction
    axes[3].plot(quality,'r')
    axes[3].set_title('Reconstruction sharpness')
    
    # Update the figure
    clear_output(wait=True)  # Clear the output before displaying the new figure 
    display(fig)  # Display the updated figure
    plt.pause(0.1)  # Pause to allow the figure to update if needed
    
# Update kx and ky for the current image by finding the kx and ky that minimise error between image estimate
# and current low res image. search_range should be odd
def update_LED_positions_accurate(obj,img,pupil,kx,ky,img_size,obj_center,image_number,search_range=10):
    
    # Easier to crop obj using x_start and y_start 
    x_start = int(obj_center + kx - img_size//2)
    y_start = int(obj_center - ky - img_size//2)  
    
    # Define the range we will search for a minimum
    x_offsets = range(-(search_range // 2), (search_range // 2) + 1)
    y_offsets = range(-(search_range // 2), (search_range // 2) + 1)

    min_error = float('inf') # Min error for this iteration
    
    # Find error between image and estimated image, where we offset the object crop region slightly to find estimated image
    for x in x_offsets:
        for y in y_offsets:       
            estimated_image = IFT(obj[y_start+y:y_start+y+img_size, x_start+x:x_start+x+img_size]*pupil) # Estimated image defined as usual (but with offsetted spectrum)
            error = np.mean((np.abs(estimated_image) - img)**2) # MSE between estimated and measured image
            # error_heatmap[(search_range // 2 - y), (x + search_range // 2)] = error # Add error to heatmap (convert from cartesian to image coords)
    
            # Track the smallest error position
            if error < min_error:
                min_error = error
                optimal_x = x # Offsets for mimimum error within the sub-region
                optimal_y = y
            
    # # Plot the heatmap for a specific image number
    # error_heatmap = np.zeros((search_range,search_range)) # For visualising algorithm
    # if image_number == 15:  # Image number to inspect correction algorithm
    #     plt.imshow(error_heatmap, cmap='hot', extent=[x_offsets.start, x_offsets.stop - 1, y_offsets.start, y_offsets.stop - 1])
    #     plt.colorbar(label='Error')
    #     plt.xlabel('X Offset')
    #     plt.ylabel('Y Offset')
    #     plt.title(f'Error Landscape, image {image_number}')

    #     # Label the minimum error position
    #     plt.text(optimal_x, optimal_y, 'X', color='white', fontsize=12, ha='center', va='center', fontweight='bold')
    #     plt.show()
        
    #     # Diagnostics
    #     print(f"optimal_x: {optimal_x}, optimal_y: {optimal_y}, min_error: {min_error}, x_start: {x_start}, y_start: {y_start}")

    # Apply the optimal offsets to the crop region
    x_start += optimal_x 
    y_start += optimal_y
    
    # We have optimal x_start and y_start so just rearrange for kx,ky using below relations
    # x_start = int(obj_center + kx[i] - img_size//2)
    # y_start = int(obj_center - ky[i] - img_size//2)  
    kx = x_start - obj_center + img_size//2 # Will be integers now due to discrete nature of optimisation
    ky = obj_center - img_size//2 - y_start
        
    return kx,ky 

# Modification of accurate algorithm to use a subrange to search for minimum error LED position.
# CURRENTLY DOES NOT WORK - landscape is not smooth, so we don't converge using subranges
def update_LED_positions_fast(obj,img,pupil,kx,ky,img_size,obj_center,image_number,subrange_size=5):
    
    iteration_limit = 10 # Max number of re-center and search iterations
    
    # Easier to crop obj using x_start and y_start 
    x_start = int(obj_center + kx - img_size//2)
    y_start = int(obj_center - ky - img_size//2)  
    
    # Calculate the offsets to test for. If subrange_size = 5 these offsets will be [-2,-1,0,1,2]
    x_offsets = range(-(subrange_size // 2), (subrange_size // 2) + 1)
    y_offsets = range(-(subrange_size // 2), (subrange_size // 2) + 1)
    
    for _ in range(iteration_limit):
        min_error = float('inf') # Min error for this iteration
        # error_heatmap = np.zeros((subrange_size,subrange_size)) # For visualising algorithm
        
        # Find error between image and estimated image, where we offset the object crop region slightly to find estimated image
        for x in x_offsets: 
            for y in y_offsets:       
                estimated_image = IFT(obj[y_start+y:y_start+y+img_size, x_start+x:x_start+x+img_size]*pupil) # Estimated image is IFT of cropped spectrum at the shifted center
                error = np.mean((np.abs(estimated_image) - img)**2) # MSE between estimated and measured image
                # error_heatmap[(subrange_size // 2 - y), (x + subrange_size // 2)] = error # Add error to heatmap (convert from cartesian to image coords)
       
                # We are looking for the jiggle (x,y) that minimises the error 
                if error < min_error:
                    min_error = error
                    optimal_x = x # Offsets for mimimum error within the sub-region
                    optimal_y = y
                    
        # # Plot the heatmap for a specific image number
        # if image_number == 15: # Image number to inspect correction algorithm
        #     plt.imshow(error_heatmap, cmap='hot', extent=[x_offsets.start, x_offsets.stop - 1, y_offsets.start , y_offsets.stop - 1])
        #     plt.colorbar(label='Error')
        #     plt.xlabel('X Offset')
        #     plt.ylabel('Y Offset')
        #     plt.title(f'Error Landscape, image {image_number}, iteration {_}')
            
        #     # Label the minimum error cell
        #     plt.text(optimal_x, optimal_y, 'X', color='white', fontsize=12, ha='center', va='center', fontweight='bold')
        #     plt.show()
            
        #     # Diagnostics
        #     print(f"Iteration {_}, optimal_x: {optimal_x}, optimal_y: {optimal_y}, min_error: {min_error}, x_start: {x_start}, y_start: {y_start}")
              
        if optimal_x == 0 and optimal_y == 0: # Minimum error is at center of subrange, so we know best LED position is here
            found_min = True
            break
        else: # Update search region by moving towards optimal x,y
            x_start += optimal_x
            y_start += optimal_y
        
    # We have optimal x_start and y_start so just rearrange for kx,ky using below relations
    # x_start = int(obj_center + kx[i] - img_size//2)
    # y_start = int(obj_center - ky[i] - img_size//2)  
    kx = x_start - obj_center + img_size//2
    ky = obj_center - img_size//2 - y_start
        
    return kx,ky  

# Reconstruct object and pupil function from series of low res images
def reconstruct(images, kx, ky, obj, pupil_binary, options, fig, axes, pupil=None):
    # Inputs: 
    # images; low res image array data, in order taken
    # kx,ky; location of LEDs in Fourier domain, in order of images taken
    # obj; initial estimate for object in frequency domain
    # pupil_binary; binary mask for low pass cutoff
    # options; alpha, beta (regularisation), max_iter, plotting, LED_correction
    # fig, axes; for plotting
    # pupil; known initial pupil function (optional)
    
    # Returns: 
    # IFT(obj); recovered object
    # pupil; recovered pupil function
    # kx,ky; updated LED positions (or same as input if no LED correction)
    
    # Unpack options
    alpha = options['alpha'] # Regularisation for object update
    beta = options['beta'] # Regularisation for pupil update
    max_iter = options['max_iter'] # Number of iterations to run algorithm (1 iteration uses all images)
    plot_mode = options['plot_mode'] # If using .py use 0,1. For notebook use 2,3
    update_method = options['update_method'] # 1 for QN, 2 for EPRY (alpha beta need to be changed accordingly)
    LED_correction = options['LED_correction'] # Do correction for kx,ky - LED coordinates
    
    # Other parameters
    img_size = images.shape[0] # Square, same size as pupil function
    num_images = images.shape[2]
    obj_size = obj.shape[0] # Square
    obj_center = obj_size // 2 # Center of object (used for inserting spectra in correct place) 
    update_size = np.zeros(num_images) # To monitor object update size (can spot instability numerically)
    quality = np.zeros(num_images)
    
    if pupil is None:
        pupil = np.copy(pupil_binary) # Start with binary mask if no pupil function passed
    pupil = pupil.astype('complex64') # Pupil function for updating needs to be complex  
    
    # Main loop
    for iter in range(max_iter):
        for i in range(num_images): # For each image in data set  
            # Determine object crop region
            x_start = int(obj_center + kx[i] - img_size//2) 
            y_start = int(obj_center - ky[i] - img_size//2)  
            
            # The relevant part of object spectrum to update
            object_cropped = obj[y_start:y_start+img_size, x_start:x_start+img_size] # Updates to object_cropped will directly modify main spectrum
            
            # Measured image amplitude
            img = np.sqrt(images[:,:,i])
            
            # Estimated image in Fourier domain, i.e. simulated exit wave throgh sample
            # estimated_image = np.copy(object_cropped) # Cheating but works (pseudo-ptychography)
            estimated_image = object_cropped * pupil # Correct method (actual ptychography)
            
            if update_method == 1:
                # Momentum can still be used (less obvious)
                # alpha = 0.2*(1+iter)
                
                # The update image (in Fourier domain) is composed of the magnitude of the measured image, 
                # the phase of the estimated image and also the estimated image spectrum is subtracted
                update_image = FT(img*np.exp(1j*np.angle(IFT(estimated_image)))) - estimated_image
                
                # Object update QN
                numerator = np.abs(pupil) * np.conj(pupil) * update_image
                denominator = np.max(np.abs(pupil)) * (np.abs(pupil)**2 + alpha)
                object_update = numerator / denominator
                object_cropped += object_update # Update cropped region

                # Pupil update QN
                numerator = np.abs(object_cropped) * np.conj(object_cropped) * update_image * pupil_binary
                denominator = np.max(np.abs(obj)) * (np.abs(object_cropped)**2 + beta)
                pupil_update = numerator / denominator
                pupil += pupil_update
                # pupil[pupil_binary] = np.exp(1j*np.angle(pupil[pupil_binary])) # Intensity constraint
            
            # ePIE algorithm (extened ptychographic iterative engine)   
            elif update_method == 2:
                # Momentum
                alpha = 0.4*(1+iter)
                beta = 0.4*(1+iter)
                
                # Update image formed with magnitude of measured image, normalised by estimated image
                # and again spectrum of estimated image is removed
                update_image = FT(img*IFT(estimated_image)/np.abs(IFT(estimated_image))) - estimated_image
                
                # Object update EPRY
                numerator = np.conj(pupil) * update_image
                denominator = np.max(np.abs(pupil))**2
                object_update = numerator / denominator
                object_cropped += alpha * object_update # Add to main spectrum with weight alpha
                
                # Pupil update EPRY
                numerator = np.conj(object_cropped) * update_image * pupil_binary
                denominator = np.max(np.abs(object_cropped))**2
                pupil_update = numerator / denominator
                pupil += beta * pupil_update # Update pupil with weight beta
                # pupil[pupil_binary] = np.exp(1j*np.angle(pupil[pupil_binary])) # Intensity constraint
            
            update_size[i] = np.mean(np.abs(object_update)) # To check instability
      
            # LED position (kx,ky) correction for image we just used, algorithm 1
            if LED_correction == 1:
                kx_new,ky_new = update_LED_positions_accurate(obj,img,pupil,kx[i],ky[i],img_size,obj_center,i)
                kx[i] = kx_new # Updated LED positions
                ky[i] = ky_new
            
            # Algorithm 2    
            if LED_correction == 2:
                kx_new,ky_new = update_LED_positions_fast(obj,img,pupil,kx[i],ky[i],img_size,obj_center,i)
                kx[i] = kx_new # Updated LED positions
                ky[i] = ky_new
                
            # Plot every image
            if plot_mode == 2:
                plot_ipynb(fig,axes,obj,x_start,y_start,img_size,obj_center,pupil,kx,ky,i,iter,plot_mode,update_size,quality) # Plotting for notebook
        
        # Status message
        progress = int((iter+1)/max_iter * 100)
        sys.stdout.write(f'\r Reconstruction Progress: {progress}%') # Write to same line
        sys.stdout.flush()
        
        # Plot every iteration
        if plot_mode == 1:
            plot_py(fig,axes,obj) # Plotting for main.py 
        elif plot_mode == 3:
            plot_ipynb(fig,axes,obj,x_start,y_start,img_size,obj_center,pupil,kx,ky,i,iter,plot_mode,update_size,quality) # Plotting for notebook
    
    print('\n Reconstruction Done!') # Write to new line

    return IFT(obj),pupil,kx,ky

