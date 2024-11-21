Will Taylor 4th year project code for Fourier ptychographic microscopy 

KEY SCRIPTS
* reconstruction.ipynb is the Jupyter notebook version for the Fourier reconstruction
* gather_data.ipynb is the Jupyter notebook version for gathering the dataset of images

** main.py does whole process in one script
* fpm_functions.py contains functions for main.py (should also work in the notebooks but careful with plotting)


gpioset gpiochip4 45=0 to turn fan on on RPi

If camera or LED matrix get stuck in on state need to run the following in either notebook or .py script:
led_matrix = RPiLedMatrix()
led_matrix.off()
camera = Picamera2()
camera.stop()
camera.close()

KEY FINDINGS AND CONSIDERATIONS

While the reconstruction algorithm can handle a 16x16 image dataset, this takes a lot longer and produces 
instability due to poor quality wide angle illumination images. If using 16x16 use quality_threshold and moderator_on options to stabilise. Alternatively, use a smaller dataset (4x4 -> 8x8) for stable 
reconstruction with quality_threshold = 0 and moderator_on = False. This gives a good balance of 
performance, stability and speed.

Central LED (first LED in sequence) must be aligned with the optical axis. 
This is so that the center of the Fourier domain object is aligned with the first image in the dataset
and the dataset is symmetrical about the optical axis. To achieve this, modify the LED_spiral starting
coordinate by the required offset. N.b this will mean you can't use the entire 16x16 grid, depending
on the offset required. In my case it is only off by 1 LED unit. 