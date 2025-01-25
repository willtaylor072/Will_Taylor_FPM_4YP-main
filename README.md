Will Taylor 4th year project code for Fourier ptychographic microscopy 

KEY SCRIPTS
* reconstruction.ipynb is the Jupyter notebook version for the Fourier reconstruction
* gather_data.ipynb is the Jupyter notebook version for gathering the dataset of images

* main.py does whole process in one script
* fpm_functions.py contains functions for all scripts

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
within LED_spiral algorithm to fix. For each offset unit we reduce max grid size by 1. I.e. for 1,1 offset max grid size is 15.
For consistancy use LED rotation of 135 degrees. 

The path from the top of the objective to the camera lens must be covered in order to avoid ambient light from washing out the signal. 
Adjusting the z stage to focus will slightly move the image (due to tilting), so do this first and then align x,y. 

In the fpm.reconstruction algorithm we can cheat to reconstruct an image
estimated_image = np.copy(object_cropped) # Cheating but works (pseudo-ptychography)
estimated_image = object_cropped * pupil # Correct method (actual ptychography)

Momentum may be used to speed up recovery process.

#############################################

Optical parameters:

System parameters for V2 microscope:
LED2SAMPLE = 80
LED_P = 3.3
NA = 0.1 
PIX_SIZE = 1025e-9
WLENGTH = 550e-9
x_offset = 0
y_offset = 0

System parameters for V1 microscope:
LED2SAMPLE = 55
LED_P = 3.3
NA = 0.105
PIX_SIZE = 1050e-9
WLENGTH = 550e-9
x_initial = -2.83
y_initial = -3.39

