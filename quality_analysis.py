import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import fpm_functions as fpm

# This script is for checking the reconstruction quality of USAF target high resolution image

# Select image and setup parameters
image_path = "results/library/usaf_taped/magnitude.png"
pixel_size = 1.025 / 5 # Pixel size in reconstructed image is original pixel size (in microns) divided by upsampling ratio
threshold_intensity = 0.6 # Threshold for first black line, adjust between 0.3-0.7 until error is smallest

# Prepare image
image_pil = Image.open(image_path)
angle = 2.6 # Degrees acw
image_pil_rotated = image_pil.rotate(angle,resample=Image.BICUBIC,expand=False)
# image_pil_rotated.save('test.png') # Check rotation
image = np.array(image_pil_rotated) # Convert to numpy array

# Select the group and element for testing
feature = [7,6] # Group, element for smallest lines
# feature = [6,1] # Biggest feature
resolution = 2**(feature[0] + (feature[1]-1)/6) # lp/mm
line_width = (1/resolution) * 1000 /2 # Microns per line
print(f"Line width for group {feature[0]}, element {feature[1]}: {line_width:.2f} microns")

# Get line profile and start/end coordinate from helper function
profile,coords = fpm.pixel_slice_selection(image,feature[0],feature[1])
profile -= np.min(profile) # Scale 0-1
profile /= np.max(profile)

### Can loop this entire block to optimise threshold_intensity

# Profile tells us nothing about distances, so create 2D arrays to hold intensity in [0] and distance in [1]
theoretical_profile = np.zeros((2,len(profile))) 
measured_profile = np.zeros((2,len(profile)))
measured_profile[0] = profile # The intensity values we recorded
start_dist = 0 # Distance from user inputed line start to first black line (in microns)
entered_feature = False # Flag for threshold logic

# Loop to create theoretical profile
for i in range(len(profile)):
    distance = i*pixel_size # Distance along line is number of pixels (index) * pixel_size
    
    # Save distances in each profile for plotting
    theoretical_profile[1][i] = distance
    measured_profile[1][i] = distance

    if measured_profile[0][i] < threshold_intensity and not entered_feature: # First black line on user drawn profile
        entered_feature = True
        start_dist = distance
    
    if entered_feature:   
        # Generate theoretical intensity
        if (distance-start_dist) < line_width: # First black line
            val = 0 
        elif (distance-start_dist) < 2*line_width: # First white gap
            val = 1
        elif (distance-start_dist) < 3*line_width: # Second black line
            val = 0
        elif (distance-start_dist) < 4*line_width: # Second white gap
            val = 1
        elif (distance-start_dist) < 5*line_width: # Third black line
            val = 0
        else:
            val = None # After feature region ends
            measured_profile[0][i] = None # Remove unwanted bits from the measurement
    else:
        val = None # Before feature region begins
        measured_profile[0][i] = None
        
    theoretical_profile[0][i] = val

# Offset the distances in the arrays so first black line is at 0 microns
theoretical_profile[1] -= start_dist
measured_profile[1] -= start_dist
 
# Find the error within the features region
error = 0 # MAE
num_points = 0 # Number of points within the region
for i in range(len(profile)):
    if measured_profile[1][i] >= 0 and measured_profile[1][i] < 5*line_width: # Within region
        num_points += 1
        error += np.abs(measured_profile[0][i]-theoretical_profile[0][i]) # Sum of absolute errors
error /= num_points # To get mean

### END LOOP

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(image,cmap='gray')
y_values = [coords['start'][0], coords['end'][0]]
x_values = [coords['start'][1], coords['end'][1]]
axes[0].plot(x_values, y_values, color='red')

axes[1].plot(measured_profile[1],measured_profile[0],label='Measured profile') # Plot distance on x axis and intensity on y
axes[1].plot(theoretical_profile[1],theoretical_profile[0],label='True profile')
axes[1].set_title('Measured vs Actual line profile')
plt.vlines((0,5*line_width),-0.1,1.1,colors='black') # Indicate where features start and end
plt.xlabel('Distance from start of feature (microns)')
plt.ylabel('Intensity')
plt.annotate(f'MAE: {error:.3f}',[0,-0.1])
plt.legend(loc='upper left')

plt.show()


