# Destriping code without using the GUI. 

from matplotlib import pyplot as plt
from skimage import io
import numpy as np
import destripe

fname = 'nacreous_domain.tif' #file name
wedgeSize = 5 #angular range (degrees) 
theta = 45 #orientation (degrees) (+/- 90 degrees)
kmin = 5 #min frequency to start missing wedge (1/px) ( < 0.5 * Length of image )
niter = 50 #number of iterations for reconstruction
a = 0.2 #Descent parameter for TV minimization
save = True # Set to true to save final reconstruction. 
show = True # Set to true to see FFT and location of missing wedge. 
#If set to false, the reconstruction will run. 

#Read input image
input_img = io.imread('sample_data/' + fname)
input_img = np.array(input_img, dtype=np.float32)

# Set parameters to destripe object
destripe_obj = destripe.destripe(input_img, niter, a, wedgeSize, theta, kmin)

if show: #Show the Original Image, FFT, and missing wedge (Region being reconstructed)
	destripe_obj.view_missing_wedge()
	plt.show()
else: #Start reconstruction. 
	destripe_obj.TV_reconstruction(save)