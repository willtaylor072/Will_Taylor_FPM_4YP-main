import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from PIL import Image
import sys
import numpy as np
import importlib 

# Custom functions
import fpm_functions as fpm 
importlib.reload(fpm) # Reload

# Reconstruction for FPM datasets - full frame only (use reconstruct_OLD.ipynb for cropped datasets)

##########################################################################################################

# Folders
data_folder = 'data/recent' # Get data from here
results_folder = 'results/recent' # Save results here

# Setup
grid_size = 15
img_size = 300 # Individual reconstruction size, 100-300

# If True, reconstruct entire frame by stitching together multiple reconstructions
# If False, we will select a small tile to reconstruct
full_reconstruction = False 

# Set parameters for reconstruction algorithm
options = {
    'max_iter': 8, # Number of iterations
    'alpha': 1, # Regularisation parameter for object update
    'beta': 1, # Regularisation parameter for pupil update
    'plot_mode': 1, # 0, plot only at end; 1, plot every iteration
    'LED_correction': 0, # 0, off; 1, accurate; 2, fast. Update wavevectors during reconstruction 
    'update_method': 2, # 1, PIE; 2, ePIE; 3, rPIE. Update method, ePIE reccomended
    'momentum': False, # Use momentum on alpha and beta (tuned for ePIE only)
    'intensity_correction': True, # Adjust image intensity to account for LED variation
}

# Optical system parameters
LED2SAMPLE = 50 # Distance from LED array to the sample, (shorter distance leads to greater overlap of adjacent spectra)
LED_P = 3.3 # LED pitch, mm
NA = 0.1 # Objective numerical aperture
PIX_SIZE = 1150e-9 # Pixel size on object plane, m. Directly measured for V3
x_initial = y_initial = 0 # Offset of first LED to optical axis (can be tuned slightly if reconstruction has crease like artefacts)
WLENGTH = 550e-9 # Central wavelength of LED light, m, 550nm for white or green