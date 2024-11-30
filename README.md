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

In the fpm.reconstruction algorithm we can cheat to reconstruct an image
estimated_image = np.copy(object_cropped) # Cheating but works (pseudo-ptychography)
estimated_image = object_cropped * pupil # Correct method (actual ptychography)

Momentum may be used to speed up recovery process.
