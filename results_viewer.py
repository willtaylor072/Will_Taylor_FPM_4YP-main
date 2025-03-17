import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from matplotlib_scalebar.scalebar import ScaleBar
import os
from PIL import Image
import numpy as np
import importlib 

# Custom functions
import fpm_functions as fpm 
importlib.reload(fpm) # Reload

# Simple image viewer for full frame reconstructions

##########################################################################################################

# Folders
results_folder = 'results/library/talia_full_field' 

# Select the reconstructions we have 
have_magnitude = True
have_phase = True

PIX_SIZE = 0.725 # Brightfield pixel size in um

# Read images from folder
brightfield = np.array(Image.open(os.path.join(results_folder,'brightfield.png')))/255
x_lim = brightfield.shape[1] # 1456
y_lim = brightfield.shape[0] # 1088

if have_magnitude:
    magnitude = np.array(Image.open(os.path.join(results_folder,'magnitude.png')))/255
if have_phase:
    phase = np.array(Image.open(os.path.join(results_folder,'phase.png')))/255

###### UI FUNCTIONS ########

# Handles arrow key crop adjustments and w/e resize
def on_key(event):
    global crop_start_x, crop_start_y,v_x,v_y,img_size
    acceleration = 3
    max_speed = 30
    match event.key:
        case 'up': 
            v_y = min(v_y + acceleration, max_speed)
            crop_start_y -= v_y
            crop_start_y = max(0,crop_start_y)
        case 'down': 
            v_y = min(v_y + acceleration, max_speed)
            crop_start_y += v_y
            crop_start_y = min(crop_start_y,y_lim-img_size)
        case 'left': 
            v_x = min(v_x + acceleration, max_speed)
            crop_start_x -= v_x
            crop_start_x = max(crop_start_x,0)
        case 'right':
            v_x = min(v_x + acceleration, max_speed)
            crop_start_x += v_x
            crop_start_x = min(crop_start_x,x_lim-img_size)
        case 'w': # Zoom out
            new_size = min(img_size + 5, 500)
            delta = (new_size - img_size) // 2
            img_size = new_size
            # Adjust crop start so we zoom from centre
            crop_start_x = max(0, min(crop_start_x - delta, x_lim - img_size))
            crop_start_y = max(0, min(crop_start_y - delta, y_lim - img_size))
        case 'e': # Zoom in
            new_size = max(img_size - 5, 30)
            delta = (new_size - img_size) // 2
            img_size = new_size
            crop_start_x = max(0, min(crop_start_x - delta, x_lim - img_size))
            crop_start_y = max(0, min(crop_start_y - delta, y_lim - img_size))

def on_release(event):
    global v_x,v_y
    v_x=v_y=0 # Stop box moving
    
# Callback to abort script
def abort_callback(event):
    global abort_script
    abort_script = True
    plt.close(fig) 

#############

# Set up plots
plt.ion() # Allow live plotting
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Fourier Ptychography - Results Viewer')

img_size = 200 # Size of region to view
crop_start_x = int(x_lim/2 - img_size/2)
crop_start_y = int(y_lim/2 - img_size/2)

upsampling_ratio = 5 # The upsampling ratio used when reconstructing the full frame object, probably 5
obj_size = upsampling_ratio * img_size # Size of cropped object to view

# Top left is full FOV brightfield
axes[0,0].set_aspect(x_lim / y_lim)  # Aspect ratio for the full frame
axes[0,0].imshow(brightfield, cmap='gray')  # Full frame plot
axes[0,0].set_title('Use arrow keys to move region, w/e to resize')

# Define cropped regions and set plots
brightfield_cropped = np.zeros((img_size,img_size))
magnitude_cropped = np.zeros((obj_size,obj_size))
phase_cropped = np.zeros((obj_size,obj_size))

brightfield_cropped_plot = axes[0,1].imshow(brightfield_cropped,cmap='gray',vmin=0,vmax=1)
axes[0,1].set_title('Brightfield')
magnitude_cropped_plot = axes[1,0].imshow(magnitude_cropped,cmap='gray',vmin=0,vmax=1)
axes[1,0].set_title('Reconstructed magnitude')
phase_cropped_plot = axes[1,1].imshow(phase_cropped,cmap='gray',vmin=0,vmax=1)
axes[1,1].set_title('Reconstructed phase')

abort_script = False
do_reconstruction = False
v_y=v_x=0 # For moving preview frame
prev_img_size = img_size # To detect change in image size
prev_obj_size = obj_size

# Connect the events to their handlers
fig.canvas.mpl_connect('key_press_event', on_key)
fig.canvas.mpl_connect('key_release_event', on_release)

button_ax_abort = fig.add_axes([0.1, 0.02, 0.15, 0.05]) 
button_abort = Button(button_ax_abort, 'Abort')
button_abort.on_clicked(abort_callback)

# Add scale bar to brightfield crop
scalebar = ScaleBar(PIX_SIZE, "um", length_fraction=0.25)
axes[0,1].add_artist(scalebar)

# Add rectangle to full FOV brightfield
rectangle = patches.Rectangle((crop_start_x, crop_start_y), img_size, img_size, linewidth=2, edgecolor='red', facecolor='none')
axes[0,0].add_patch(rectangle)

while not abort_script:
    # Get new cropped regions 
    brightfield_cropped = brightfield[crop_start_y:crop_start_y+img_size, crop_start_x:crop_start_x+img_size]
    x,y = crop_start_x*upsampling_ratio, crop_start_y*upsampling_ratio
    obj_size = upsampling_ratio * img_size
    magnitude_cropped = magnitude[y:y+obj_size, x:x+obj_size]
    phase_cropped = phase[y:y+obj_size, x:x+obj_size]

    # Update plot data
    brightfield_cropped_plot.set_data(brightfield_cropped)
    magnitude_cropped_plot.set_data(magnitude_cropped)
    phase_cropped_plot.set_data(phase_cropped)
    
    # Update extent only if size changed
    if img_size != prev_img_size:
        brightfield_cropped_plot.set_extent([0, img_size, img_size, 0])
        magnitude_cropped_plot.set_extent([0, obj_size, obj_size, 0])
        phase_cropped_plot.set_extent([0, obj_size, obj_size, 0])
    prev_img_size = img_size
    prev_obj_size = obj_size
    
    # Adjust rectangle position
    rectangle.set_xy((crop_start_x, crop_start_y))
    rectangle.set_width(img_size)
    rectangle.set_height(img_size)

    plt.pause(0.05)

