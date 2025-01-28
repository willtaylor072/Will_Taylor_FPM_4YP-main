import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import fpm_functions as fpm
from skimage.measure import profile_line
from skimage.filters import threshold_otsu
import os

#### This script is for checking the reconstruction quality of USAF target high resolution image ####
# It checks for general correlation between expected features and the given features
# This script could be improved by detecting edge sharpness in feature regions

# When selecting a slice:
# * Select the central portion of a feature, across its lines
# * Errors are different for the vertical and horizontal lines

# Function to select pixel slice selection on an image with x,y snap
def pixel_slice_selection(image,group,element,snap=True):
    has_clicked = False # Flag for click logic

    # Show the image using matplotlib
    fig,ax = plt.subplots()
    ax.imshow(image,cmap='gray')
    ax.set_title(f"Select the line profile for group {group} element {element} ")

    # Create a callback to capture the selected region
    coords = {"start": None, "end": None}

    def on_click(event):
        nonlocal has_clicked # Allow us to access the flag
        if event.inaxes == ax and not has_clicked:
            coords["start"] = (int(event.ydata), int(event.xdata))
            # print(f"Selection started at: {coords['start']}")
            has_clicked = True
        elif event.inaxes == ax and has_clicked:
            y = int(event.ydata)
            x = int(event.xdata)
            # Handle the snap behaviour
            if snap:
                if abs(y-coords['start'][0]) < abs(x-coords['start'][1]): # Closer in y
                    y = coords['start'][0]
                else: # Closer in x
                    x = coords['start'][1] 
            coords['end'] = (y,x)
            # print(f"Selection ended at: {coords['end']}")
            plt.close(fig) # Close fig to initiate return

    # Connect the mouse events
    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.show() # Show image and wait for user inputs

    # Extract the pixel slice once fig has been closed
    profile = profile_line(image,coords["start"],coords["end"])
    return profile,coords

# Find the error between the user selected profile and the theoretical feature profile for a given alignment offset
def get_error(profile,pixel_size,line_width,offset):
    
    "profile: User selected line slice constaining intensity at each pixel index"
    "pixel_size: Size of pixel in microns in high resolution image"
    "line_width: Width in microns of solid black or solid white line feature on USAF target"
    "offset: Array index of first black line pixel"
    
    # Method: Once the approx offset has been found, try all offsets in a certain range around it to find optimal offset 
    # (so the measured profile is best aligned with the theoretical profile). This ensures the quality test is fair for different datasets. 

    # Create theoretical profile and distance array
    theoretical_profile = np.zeros((len(profile))) 
    distances = np.zeros(len(profile)) # Used for plotting

    # Loop to create theoretical profile
    for i in range(len(profile)):
        distance = (i-offset) * pixel_size # Distance along line, zeroed at the offset
        distances[i] = distance
        
        # Generate theoretical intensity
        if distance >= 0:
            if distance < line_width: # First black line
                val = 0 
            elif distance < 2*line_width: # First white gap
                val = 1
            elif distance < 3*line_width: # Second black line
                val = 0
            elif distance < 4*line_width: # Second white gap
                val = 1
            elif distance < 5*line_width: # Third black line
                val = 0
            else: # After the region
                val = None 
        else: # Before the region
            val = None 
            
        theoretical_profile[i] = val
    
    # Find the error within the features region
    error = 0 # MSE
    num_points = 0 # Number of points within the region
    for i in range(len(profile)):
        if distances[i] >= 0 and distances[i] < 5*line_width: # Within region
            num_points += 1
            error += (profile[i]-theoretical_profile[i])**2 # Sum of square errors
    if num_points == 0:
        return 100,0,0
    error /= num_points # To get mean
    
    return error, distances, theoretical_profile

# Find the approx index of the first black line pixel in the feature
def get_approx_offset(profile):
    
    "profile: User selected line slice constaining intensity at each pixel index"
    
    # Find suitable threshold for black with Otsu's method:
    # Determines the optimal threshold by minimizing the variance within pixel intensity classes (black vs. non-black).
    threshold = threshold_otsu(profile)
    # print(threshold)
    
    # Loop through profile elements till we find black pixel
    for i in range(len(profile)):
        if profile[i] < threshold:
            return i

# Select feature to analyse, [group, element]. [7,6] is smallest, [6,1] is biggest
feature = [6,1] 
# feature = [7,6]

# Image name is name of folder where magnitude.png is
# Pixel size in reconstructed image is original pixel size (in microns) divided by upsampling ratio
# (If pixel size is uncertain: select exact feature start and end by zooming into plot and using vertical lines,
# then uncomment the first print statement below)
# Angle is rotation degrees acw

image_name = 'v2_usaf_taped'
pixel_size = 0.205 
angle = 2.6

# image_name = 'v2_usaf'
# pixel_size = 0.271 
# angle = 1.2

# image_name = 'v1_usaf'
# pixel_size = 0.19
# angle = -1.2 

# image_name = 'v1_usaf_optimal'
# pixel_size = 0.226
# angle = -1.2

# image_name = 'usaf_matlab'
# pixel_size = 0.286
# angle = 0

# Prepare image
image_pil = Image.open(os.path.join('results/library',image_name,'magnitude.png'))
image_pil_rotated = image_pil.rotate(angle,resample=Image.BICUBIC,expand=False)
# image_pil_rotated.save(os.path.join(image_folder,'rotation_test.png')) # Check rotation
image = np.array(image_pil_rotated) # Convert to numpy array

# Find line_width
resolution = 2**(feature[0] + (feature[1]-1)/6) # lp/mm
line_width = (1/resolution) * 1000 /2 # Microns per line
print(f"Line width for group {feature[0]}, element {feature[1]}: {line_width:.2f} microns")

# Get line profile and start/end coordinate from helper function
profile,coords = pixel_slice_selection(image,feature[0],feature[1])
profile /= 255 # Range should be 0-1 (but don't normalise with image information!)
vertical = True if coords["end"][1] == coords["start"][1] else False # Flag for vertical/horizontal line

# Calculate pixel size by selecting precisely start and end of feature
print(f'Number of pixels: {len(profile)}, 5 line width: {round(5*line_width,3)}um, pixel size: {round(5*line_width/len(profile),3)}um')

### Optimise offset ###

# Find approx index for first black line
approx_offset = get_approx_offset(profile)

# Test offsets around the approx offset
search_range = round(len(profile)/5) # Number of values to search either side of offset
offsets = np.linspace(approx_offset-search_range,approx_offset+search_range,search_range*2+1)
errors = np.zeros(len(offsets)) # To store errors at each adjusted offset

for idx, offset in enumerate(offsets):
    errors[idx],_,_ = get_error(profile,pixel_size,line_width,offset)

optimal_offset = offsets[np.argmin(errors)]

# # Plot optimisation curve
# fig, ax = plt.subplots()
# ax.plot(offsets,errors)
# ax.vlines([approx_offset,optimal_offset],0,1,['black','red'])   

# Find the profiles and distances using the optimal offset
error,distances,theoretical_profile = get_error(profile,pixel_size,line_width,optimal_offset)


### Repeat twice either side of the profile to average across the line section ###
lines_per_side = 3 # Number of lines per side of central line
pixel_offset = round(line_width/pixel_size*0.9 * 5/(lines_per_side*2+1)) # Line length ~ line width * 5
profile = profile_line(image,coords["start"],coords["end"])/255

# Create vertical and horizontal offset profiles
profiles = [profile]  # Initialize with the original profile

if vertical:
    # Calculate perpendicular offset vector for vertical case
    offset_vector = np.array([0, pixel_offset])  # Only y changes for vertical offset
else:
    # Calculate perpendicular offset vector for horizontal case
    offset_vector = np.array([pixel_offset, 0])  # Only x changes for horizontal offset

# Find profile lines on either side of center
for i in range(1, 1+lines_per_side): 
    # Positive offset line
    start_pos = np.array(coords["start"]) + i * offset_vector
    end_pos = np.array(coords["end"]) + i * offset_vector
    profiles.append(profile_line(image, start_pos, end_pos)/255)

    # Negative offset line
    start_neg = np.array(coords["start"]) - i * offset_vector
    end_neg = np.array(coords["end"]) - i * offset_vector
    profiles.append(profile_line(image, start_neg, end_neg)/255)

profile_errors = []
for profile in profiles:
    error,_,_ = get_error(profile,pixel_size,line_width,optimal_offset)
    profile_errors.append(error)

mean_error = np.mean(profile_errors)

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(image,cmap='gray')
# Just draw the edge and central lines
if vertical:
    x_starts = [coords["start"][1] + lines_per_side*i*pixel_offset for i in range(-1,2)]
    axes[0].vlines(x_starts,coords["start"][0],coords["end"][0],colors=['black','red','black'])
else:
    y_starts = [coords["start"][0] + lines_per_side*i*pixel_offset for i in range(-1,2)]
    axes[0].hlines(y_starts,coords["start"][1],coords["end"][1],colors=['black','red','black'])
axes[0].set_title(f'{image_name}, group: {feature[0]}, element: {feature[1]}')

axes[1].plot(distances,profile,label='Measured profile') # Plot distance on x axis and intensity on y
axes[1].plot(distances,theoretical_profile,label='True profile')
axes[1].set_title('Measured vs Actual line profile')
axes[1].vlines((0,5*line_width),-0.1,1.1,colors='black') # Indicate where features start and end
plt.xlabel('Distance from start of feature (microns)')
plt.ylabel('Intensity')
axes[1].annotate(f'Central MSE: {error:.3f}, Average MSE: {mean_error:.3f}',[0,-0.1])
axes[1].legend(loc='upper left')

# Save to results folder
# save_name = f'{feature}_{'vertical' if vertical else 'horizontal'}.png'
# plt.savefig(os.path.join('results/library',image_name,save_name))

plt.show()


