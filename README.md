Will Taylor 4th year project code for Fourier ptychographic microscopy 

KEY SCRIPTS
reconstruction.ipynb is the Jupyter notebook version for the Fourier reconstruction
gather_data.ipynb is the Jupyter notebook version for gathering the dataset of images

main.py does whole process in one script
(n.b. it may be necessary to split this .py into sub scripts in a main folder for readability)

gpioset gpiochip4 45=0 to turn fan on on RPi

If camera or LED matrix get stuck in on state need to run the following in either notebook or .py script:
led_matrix = RPiLedMatrix()
led_matrix.off()
camera = Picamera2()
camera.stop()
camera.close()

While the reconstruction algorithm can handle a 16x16 image dataset, this takes a lot longer and produces 
instability due to poor quality wide angle illumination images. If using 16x16 use quality_threshold and moderator_on options 
to stabilise. Alternatively, use a smaller dataset (4x4 -> 8x8) for stable reconstruction with quality_threshold = 0 and 
moderator_on = False.