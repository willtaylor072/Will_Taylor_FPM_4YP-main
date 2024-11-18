Will Taylor 4th year project code for Fourier ptychographic microscopy 

KEY SCRIPTS
reconstruction.ipynb is the Jupyter notebook version for the Fourier reconstruction
gather_data.ipynb is the Jupyter notebook version for gathering the dataset of images

main.py should eventually do everything, all in one executable python script
(n.b. it may be necessary to split this .py into sub scripts in a main folder for readability)

gpioset gpiochip4 45=0 to turn fan on on RPi

If camera or LED matrix get stuck in on state need to run the following in either notebook or .py script:
led_matrix = RPiLedMatrix()
led_matrix.off()
camera = Picamera2()
camera.stop()
camera.close()