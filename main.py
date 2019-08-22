from matplotlib.widgets import TextBox, Button
from matplotlib import pyplot as plt
from skimage import io
import numpy as np
import destripe

fname = 'nacreous_domain.tif' #file name
wedgeSize = 5 #angular range (degrees)
theta = 0 #min frequency to start missing wedge (1/px)
kmin = 15 #orientation (degrees)
niter = 10 #number of iterations for rconstruction
a = 0.1 #Descent arameter for TV minimization
save = True # Set to true to save image 

input_img = io.imread('sample_data/' + fname)
input_img = np.array(input_img, dtype=np.float32)

destripe_obj = destripe.destripe(input_img, niter, a, wedgeSize, theta, kmin)

def submit_dtheta(wedgeSize):
	destripe_obj.edit_wedgeSize(wedgeSize)
	create_input_boxes(wedgeSize, theta, kmin)

def submit_theta(theta):
	destripe_obj.edit_theta(theta)
	create_input_boxes(wedgeSize, theta, kmin)
 
def submit_kmin(kmin):
	destripe_obj.edit_kmin(kmin)
	create_input_boxes(wedgeSize, theta, kmin)

def run_destripe(event):
	destripe_obj.TV_reconstruction(save)

def create_input_boxes(wedgeSize, theta, kmin):
	dtheta_box = plt.axes([0.17, 0.04, 0.1, 0.05])
	dtheta_input = TextBox(dtheta_box, 'Wedge Size: ', initial=str(wedgeSize))
	dtheta_input.on_submit(submit_dtheta)

	theta_box = plt.axes([0.37, 0.04, 0.1, 0.05])
	theta_input = TextBox(theta_box, 'Theta: ', initial=str(theta))
	theta_input.on_submit(submit_theta)

	kmin_box = plt.axes([0.57, 0.04, 0.1, 0.05])
	kmin_input = TextBox(kmin_box, 'kmin: ', initial=str(kmin))
	kmin_input.on_submit(submit_kmin)

	recon_box = plt.axes([0.7, 0.04, 0.25, 0.05])
	recon_button = Button(recon_box, 'Start Reconstruction')
	recon_button.on_clicked(run_destripe)
	plt.show()

destripe_obj.view_missing_wedge()
create_input_boxes(wedgeSize, theta, kmin)
