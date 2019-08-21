from destripe import TV_reconstruction, view_missing_wedge
from skimage import io
import numpy as np

fname = 'nacreous_domain.tif'
wedgeSize = 5 #angular range (degrees)
theta = 0 #min frequency to start missing wedge (1/px)
kmin = 15 #orientation (degrees)
niter = #number of iterations for reconstruction

input_img = io.imread(fname)
input_img = np.array(input_img, dtype=np.float32)

view_missing_wedge(input_img, wedgeSize, theta, kmin)