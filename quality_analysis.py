import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import fpm_functions as fpm
from skimage.measure import profile_line
from skimage.filters import threshold_otsu
from scipy.optimize import curve_fit
from skimage.restoration import richardson_lucy
from scipy.fft import fft,ifft
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
def get_error(profile,pixel_size,line_width,offset,get_res=False):
    
    "profile: User selected line slice constaining intensity at each pixel index"
    "pixel_size: Size of pixel in microns in high resolution image"
    "line_width: Width in microns of solid black or solid white line feature on USAF target"
    "offset: Array index of first black line pixel"

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
                val = 0 
        else: # Before the region
            val = 0 
            
        theoretical_profile[i] = val
    
    # Find the error within the features region
    error = 0 # MSE
    num_points = 0 # Number of points within the region
    for i in range(len(profile)):
        if distances[i] >= 0 and distances[i] < 5*line_width: # Within region
            num_points += 1
            error += (profile[i]-theoretical_profile[i])**2 # Sum of square errors
    if num_points == 0:
        return
    error /= num_points # To get mean
    
    ## Estimate resolution
    res = 0
    psf = 0
    if get_res:
        edges = [line_width*i for i in range(6)] # Distances where transition occurs
        indices = [np.argmin(np.abs(distances - edge)) for edge in edges] # Indices where the edge is found

        # Specify region to do deconvolution
        i = indices[1] # Line edge
        dist = 2.5 # Distance either side of edge for deconvolution, 2.5um works
        r = round(dist/pixel_size) # Offset either direction
        cropped_profile_measured = profile[i-r:i+r]
        cropped_profile_true = theoretical_profile[i-r:i+r]
        
        # Find point spread function (psf)
        
        # Explicit deconvolution
        # theoretical profile * psf = measured profile, so we can deconvolve in Fourier domain
        # psf = ifft(fft(cropped_profile_measured)/(fft(cropped_profile_true)+1e-8))
        # psf = np.abs(psf) # Don't care about phase
        # psf/= np.max(psf)
        
        # Iterative method
        # Very important in this algorithm to use correct iteration number (too many will overfit noise and artificially
        # narrow the psf), 25 iterations works
        psf = richardson_lucy(np.array(cropped_profile_measured), np.array(cropped_profile_true), num_iter=25)
        
        # Crop psf
        psf = psf[int(len(psf)*0.3):int(len(psf)*0.9)] # Crop again to remove boundary artefacts
                
        def gaussian(x, a, x0, sigma):
            return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
        
        # Measure the Full Width at Half Maximum (FWHM) of the PSF
        # Fit Gaussian to the PSF
        x = np.arange(len(psf),dtype=np.float64) # x series data (just pixel indices)
        y = np.array(psf,dtype=np.float64) # y series data             
        mean = np.argmax(y)
        sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
        popt, _ = curve_fit(gaussian, x, y, p0=[1, mean, sigma],)
        sigma = abs(popt[2])
        res = 2.355 * sigma  # Convert Gaussian sigma to FWHM (resolution)
        res *= pixel_size # Convert from number of pixels to physical distance (um)
    
    return error, res, psf, distances, theoretical_profile

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
# feature = [6,1] 
# feature = [7,6]
# feature = [6,3]

# Image name is name of folder where magnitude.png is
# Pixel size in reconstructed image is original pixel size (in microns) divided by upsampling ratio
# (If pixel size is uncertain: select exact feature start and end by zooming into plot and using vertical lines,
# then uncomment the first print statement below)
# Angle is rotation degrees acw

# Resolution for 10,12,14 iterations, recommended 15 for single edge

image_name = 'v3_usaf_47' # 0.377, 0.33, 0.294
pixel_size = 0.23
angle = -1

# image_name = 'v3_usaf_58'
# pixel_size = 0.23
# angle = -1

# image_name = 'v2_usaf_taped' # 0.635, 0.50, 0.428
# pixel_size = 0.205 
# angle = 2.6

# image_name = 'v1_usaf' # 0.670, 0.605, 0.472
# pixel_size = 0.19
# angle = -1.2 

# image_name = 'v1_usaf_optimal' # 0.488, 0.423, 0.431
# pixel_size = 0.226
# angle = -1.2

# image_name = 'usaf_matlab' # 0.436, 0.384, 0.369
# pixel_size = 0.286
# angle = 0

# Select other image to measure pixel size (comment out first line of next section)
# image_pil = Image.open('data/library/usaf_v3_NEW/image_0.png')
# angle = -0.5
# pixel_size = 1.15

image_pil = Image.open(os.path.join('results/library',image_name,'magnitude.png'))
image_pil_rotated = image_pil.rotate(angle,resample=Image.BICUBIC,expand=False)
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
print(f'Number of pixels: {len(profile)}, five line width: {round(5*line_width,3)}um, pixel size: {round(5*line_width/len(profile),3)}um')

### Optimise offset ###

# Find approx index for first black line
approx_offset = get_approx_offset(profile)

# Test offsets around the approx offset
search_range = round(len(profile)/5) # Number of values to search either side of offset
offsets = np.linspace(approx_offset-search_range,approx_offset+search_range,search_range*2+1)
errors = np.zeros(len(offsets)) # To store errors at each adjusted offset

for idx, offset in enumerate(offsets):
    errors[idx],_,_,_,_ = get_error(profile,pixel_size,line_width,int(offset))

optimal_offset = offsets[np.argmin(errors)]

# # Plot optimisation curve
# fig, ax = plt.subplots()
# ax.plot(offsets,errors)
# ax.vlines([approx_offset,optimal_offset],0,1,['black','red'])   

# Get the theoretical profile and distances using the optimal offset (used for plotting)
error,res,psf,distances,theoretical_profile = get_error(profile,pixel_size,line_width,optimal_offset,get_res=True)

plt.plot(np.arange(len(psf)) * pixel_size, psf)
plt.xlabel('Length (microns)')
plt.ylabel('Intensity')
plt.title('Estimated point spread function')
plt.show()
print(res)

### Repeat either side of the profile to average across the line section ###
lines_per_side = 3 # Number of lines per side of central line
pixel_offset = round(line_width/pixel_size*0.7 * 5/(lines_per_side*2+1)) # Line length = line width * 5
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

profile_errors = [error]
profile_resolutions = [res]

for profile in profiles:
    error,res,_,_,_ = get_error(profile,pixel_size,line_width,optimal_offset,get_res=True)
    profile_errors.append(error)
    profile_resolutions.append(res)

# Mean across all segments
mean_error = np.mean(profile_errors) 
mean_res = np.mean(profile_resolutions)

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
axes[1].vlines((0,5*line_width),-0.2,1.1,colors='black') # Indicate where features start and end
plt.xlabel('Distance from start of feature (microns)')
plt.ylabel('Intensity')
axes[1].annotate(f'Average MSE: {mean_error:.3f} (arb)',[0,-0.1])
axes[1].annotate(f'Estimated resolution: {mean_res:.3f}um',[0,-0.2])
axes[1].legend(loc='upper left')

# Save
save_name = f'{feature}_{'vertical' if vertical else 'horizontal'}.png'
plt.savefig(os.path.join('results/library',image_name,save_name))

plt.show()
