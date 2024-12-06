Will Taylor 4th year project code for Fourier ptychographic microscopy 

KEY SCRIPTS
* reconstruction.ipynb is the Jupyter notebook version for the Fourier reconstruction
* gather_data.ipynb is the Jupyter notebook version for gathering the dataset of images

* main.py does whole process in one script
* fpm_functions.py contains functions for main.py (should also work in the notebooks but careful with plotting)

gpioset gpiochip4 45=0 to turn fan on on RPi

If camera or LED matrix get stuck in on state need to run the following in either notebook or .py script:
led_matrix = RPiLedMatrix()
led_matrix.off()
camera = Picamera2()
camera.stop()
camera.close()

KEY FINDINGS AND CONSIDERATIONS

Central LED (first LED in sequence) should be close to optical axis. If it is not directly aligned with
the optical axis (i.e. vertically through aperture of objective) then we need to use x_offset and y_offset 
to adjust wavevectors accordingly (by adjusting x_abs and y_abs). Ideally we use LED correction to get most accurate wavevectors but this can take lots of time.
To determine x and y offsets easiest method is to use fpm.display_data and compare central LED to center of brightfield (optical axis). 
For consistancy use LED rotation of 135 degrees. 

The path from the top of the objective to the camera lens must be covered in order to avoid ambient light from washing out the signal. 
Adjusting the z stage to focus will slightly move the image (due to tilting), so do this first and then align x,y. 

In the fpm.reconstruction algorithm we can cheat to reconstruct an image
estimated_image = np.copy(object_cropped) # Cheating but works (pseudo-ptychography)
estimated_image = object_cropped * pupil # Correct method (actual ptychography)

Momentum may be used to speed up recovery process.

Optical parameters (everything in data library uses V2):

System parameters for V2 microscope:
LED2SAMPLE = 80 # Distance from LED array to the sample, 54/80mm (larger distance leads to closer overlapping Fourier circles)
LED_P = 3.3 # LED pitch, mm
NA = 0.1 # Objective numerical aperture
PIX_SIZE = 1025e-9 # Pixel size on object plane, m
WLENGTH = 550e-9 # Central wavelength of LED light, m
x_offset = -1*LED_P # x distance from central LED to optical axis, mm (+ve if central LED is to right of optical axis)
y_offset = 1.5*LED_P # y distance from central LED to optical axis, mm (+ve if central LED is below optical axis)

System parameters for V1 microscope:
LED2SAMPLE = 54 # Distance from LED array to the sample, 54/80mm (larger distance leads to closer overlapping Fourier circles)
LED_P = 3.3 # LED pitch, mm
NA = 0.1 # Objective numerical aperture
PIX_SIZE = 1090e-9 # Pixel size on object plane, m
WLENGTH = 550e-9 # Central wavelength of LED light, m
x_offset = -1*LED_P # x distance from central LED to optical axis, mm (+ve if central LED is to right of optical axis)
y_offset = 1*LED_P # y distance from central LED to optical axis, mm (+ve if central LED is below optical axis)