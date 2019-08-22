from skimage import io
import numpy as np
import destripe

fname = 'nacreous_domain.tif' #file name
wedgeSize = 5 #angular range (degrees)
theta = 0 #min frequency to start missing wedge (1/px)
kmin = 15 #orientation (degrees)
niter = 5 #number of iterations for reconstruction
a = 0.2 #Descent arameter for TV minimization
save = True # Set to true to save image 

input_img = io.imread('sample_data/' + fname)
input_img = np.array(input_img, dtype=np.float32)

destripe_obj = destripe.destripe(input_img, niter, a, wedgeSize, theta, kmin)

destripe_obj.TV_reconstruction(save)
