from matplotlib.widgets import TextBox, Button
from matplotlib import pyplot as plt
from skimage import io
import numpy as np
import destripe

fname = 'nacreous_domain.tif' #file name
wedgeSize = 5 #angular range (degrees) 
theta = 0 #orientation (degrees) (+/- 90 degrees)
kmin = 15 #min frequency to start missing wedge (1/px) ( < 0.5 * Length of image )
niter = 100 #number of iterations for reconstruction
a = 0.2 #Descent parameter for TV minimization
save = True # Set to true to save image 


#Read input image
input_img = io.imread('sample_data/' + fname)
input_img = np.array(input_img, dtype=np.float32)

# Set parameters to destripe object
destripe_obj = destripe.destripe(input_img, niter, a, wedgeSize, theta, kmin)

#Check if version of matplotlib can support GUI
destripe_obj.check_matplotlib_version()

#Submit new parameters in GUI
def submit_dtheta(wedgeSize):
	destripe_obj.edit_wedgeSize(wedgeSize)
	create_input_boxes()

def submit_theta(theta): 
	destripe_obj.edit_theta(theta)
	create_input_boxes()
 
def submit_kmin(kmin):
	destripe_obj.edit_kmin(kmin)
	create_input_boxes()

def submit_niter(niter):
	destripe_obj.edit_niter(niter)
	create_input_boxes()

def run_destripe(event):
	destripe_obj.TV_reconstruction(save)

# Create GUI
def create_input_boxes():

	(cwedgeSize, ctheta, ckmin, cniter) = destripe_obj.get_params()

	dtheta_box = plt.axes([0.15, 0.03, 0.05, 0.05])
	dtheta_input = TextBox(dtheta_box, 'Wedge Size: ', initial=str(cwedgeSize))
	dtheta_input.on_submit(submit_dtheta)

	theta_box = plt.axes([0.29, 0.03, 0.05, 0.05])
	theta_input = TextBox(theta_box, 'Theta: ', initial=str(ctheta))
	theta_input.on_submit(submit_theta)

	kmin_box = plt.axes([0.42, 0.03, 0.05, 0.05])
	kmin_input = TextBox(kmin_box, 'kmin: ', initial=str(ckmin))
	kmin_input.on_submit(submit_kmin)

	niter_box = plt.axes([0.65, 0.03, 0.05, 0.05])
	niter_input = TextBox(niter_box, 'Total Iterations: ', initial=str(cniter))
	niter_input.on_submit(submit_niter)

	recon_box = plt.axes([0.72, 0.03, 0.25, 0.05])
	recon_button = Button(recon_box, 'Start Reconstruction')
	recon_button.on_clicked(run_destripe)
	plt.show()

destripe_obj.view_missing_wedge()
create_input_boxes()