Will Taylor 4th year project code for Fourier ptychographic microscopy 

KEY SCRIPTS
* gather_data_NEW.py for taking datasets (full frame)
* reconstruct_NEW.py for reconstructing the dataset (full frame)
* results_viewer.py for viewing full frame reconstructions

* main.py has combined code but only for cropped regions
* fpm_functions.py contains functions for all scripts

gpioset gpiochip4 45=0 to turn fan on on RPi

If camera or LED matrix get stuck in on state need to run the following in either notebook or .py script:
led_matrix = RPiLedMatrix()
led_matrix.off()
camera = Picamera2()
camera.stop()
camera.close()

Typical algorithm config:
options = {
    'max_iter': 5, # Number of iterations
    'alpha': 1, # Regularisation parameter for object update
    'beta': 1, # Regularisation parameter for pupil update
    'plot_mode': 1 for .py, 3 for .ipynb
    'LED_correction': 0, # 0, off; 1, accurate; 2, fast; 3, first iteration only
    'update_method': 2, # 1, PIE; 2, ePIE; 3, rPIE
    'momentum': False, # Use momentum on alpha and beta (tuned for ePIE only)
    'intensity_correction': False, # Adjust image intensity to account for LED variation
}

PIE is slowest, typically I use ePIE. If stability is required (e.g. poor data) use rPIE.
alpha = beta = 1 seems to usually work.


KEY FINDINGS AND CONSIDERATIONS

Central LED (first LED in sequence) should be close to optical axis. If it is not directly aligned with
the optical axis (i.e. vertically through aperture of objective) then we need to use x_offset and y_offset
within LED_spiral algorithm to fix. For each offset unit we reduce max grid size by 1. I.e. for 1,1 offset max grid size is 15.
For consistancy use LED rotation of 135 degrees so that LED (0,0) is in indicated position on microscope (back left).

The path from the top of the objective to the camera lens must be covered in order to avoid ambient light from washing out the signal. 
Adjusting the z stage to focus will slightly move the image (due to tilting), so do this first and then align x,y. 

Momentum may be used to speed up recovery process.

(For usaf_v3_64 the aperture under the sample was widened from 3mm to 3.4mm which made the darkfield images higher contrast (less leakge))
The hole under the sample should be taped to 1.25x1.67mm to match the camera FOV and not let any unecessary light into the system from the 
LEDs.

Squiggly artefacts in reconstructed images are very sensitive to input wavevectors -> i.e. LED to sample distance and initial LED position

Exposure time of 0.5s is good for USAF for white light

Also larger LED to sample distance might be useful for natural samples, to provide more redundant data in reconstruction
For the USAF target we used the shortest LED to sample distance to include the highest frequency spacial information and get the best resolution, but natural samples might need a larger brightfield region to reconstruct effectively

Green LED light (550nm) worked well - didn't require much adjustment to exposure (just a slight increase of say 100ms)
Objective is corrected for 550nm spherical aberrations, so may reduce noise in darkfield (not extensively tested)

Main aberrations are spherical (and usually some defocus)
Nominal upsampling ratio of 5 is usually good

Data gathering time for 15x15 with 0.3-0.5s exposue ~ 90-120s
Reconstruction time for 300x300 greyscale image with 8 iterations using ePIE and no LED correction ~ 70s (for raspberry pi 5 8Gb)
~ 3 min total

For natural samples phase image can be really useful to see 3D structure of sample (especially when attenuation is low)
Often the best LED2SAMPLE distance to use is 5mm LARGER than measured value (give more redundancy in recovery)

USEFUL GITHUB COMMANDS
git reset --hard origin/main    # Pulls the main repo regardless of any mess in the current working tree (overriding current device)

#############################################

BEST RESULTS OBTAINED
setup: 
v3 microscope with shortest LED to sample distance (3x magnification)
aperture under slide: 3mm (widening to 3.5mm reduced quality)
15x15 grid, white LED light
exposure = 0.5s
x_offset = 1 
y_offset = 0 - first LED was (8,7)

data: usaf_v3_47 

Optimised paramters:
LED2SAMPLE = 50 (not 47!)
LED_P = 3.3
NA = 0.1
PIX_SIZE = 1150e-9
WLENGTH = 550e-9
x_initial = 0.9
y_initial = -0.5

algorithm:
ePIE
alpha = beta = 1
wavevector correction off
intensity correction on
iterations = 8 (it was repeated with 15 but made no visible improvement)

result: v3_usaf_47

#############################################

Optical parameters:

System parameters for V3 microscope:
LED2SAMPLE: measure directly or see name (closest distance is about 50mm)
x_initial = 0.9
y_initial = -0.5 # These can be finely tuned in sensitivity_testing.ipynb
PIX_SIZE = 1150e-9 (measured for usaf_v3_47), 850 for 4x magnification, 725 measured value
LED sequence offset
x_offset = 1
y_offset = 0 # This is the correct alignment for the sequence - (8,7) is central LED

System parameters for V2 microscope:
LED2SAMPLE = 80 # Distance from LED array to the sample, mm (larger distance leads to closer overlapping Fourier circles)
LED_P = 3.3 # LED pitch, mm
NA = 0.1 # Objective numerical aperture
PIX_SIZE = 1025e-9 # Pixel size on object plane, m
WLENGTH = 550e-9 # Central wavelength of LED light, m
x_initial = 0 # x distance from central LED to optical axis, mm (+ve if central LED is to right of optical axis)
y_initial = 0 # y distance from central LED to optical axis, mm (+ve if central LED is below optical axis)

System parameters for V1 microscope:
LED2SAMPLE = 54 # Distance from LED array to the sample, mm (larger distance leads to closer overlapping Fourier circles)
LED_P = 3.3 # LED pitch, mm
NA = 0.1 # Objective numerical aperture
PIX_SIZE = 1090e-9 # Pixel size on object plane, m
WLENGTH = 550e-9 # Central wavelength of LED light, m
x_initial = -2.83 # x distance from first LED to optical axis, mm (+ve if first LED is to right of optical axis)
y_initial = -3.39 # y distance from first LED to optical axis, mm (+ve if first LED is above optical axis)
