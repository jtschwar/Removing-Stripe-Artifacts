from numpy.fft import fftn, ifftn, fftshift, ifftshift
from matplotlib import pyplot as plt
from scipy import ndimage
import numpy as np

def TV_reconstruction(dataset, Niter = 100, a = 0.1, wedgeSize = 5, theta =0, kmin = 15): 

	#dataset – real-space image (pixels)
	#Niter – Number of iterations for reconstruction
	# a – Decent parameter (unitless, typically 0 – 0.3)
	# wedgeSize – angular range of the missing wedge (degrees)
	# theta – orientation of the missing wedge (degrees)
	# kmin – Minimum frequency to start the missing wedge (1/px)

	#Import dimensions from dataset
	(nx, ny) = dataset.shape
		
	mask = create_mask(dataset, wedgeSize, theta, kmin)

	#FFT of the Original Image
	FFT_image = fftshift(fftn(dataset)) 	

	# Reconstruction starts as random image.		
	recon_init = np.random.rand(nx,ny) 						

	#Artifact Removal Loop
	for i in range(Niter):

		#FFT of Reconstructed Image.
		FFT_recon = fftshift(fftn(recon_init)) 		
			
		#Data Constraint
		FFT_recon[mask] = FFT_image[mask] 			

		#Inverse FFT
		recon_constraint = np.real(ifftn(ifftshift(FFT_recon)))

		#Positivity Constraint 
		recon_constraint[ recon_constraint < 0 ] = 0

		if i < Niter -1:  	#ignore on last iteration	

			#TV Minimization
			# The basis for TVDerivative (20 iterations and epsilon = 1e-8) was determined by Sidky (2006). 
			recon_minTV = recon_constraint
			d = np.linalg.norm(recon_minTV - recon_init)
			for j in range(20):
				Vst = TVDerivative(recon_minTV, nx, ny)
				recon_minTV = recon_minTV - a*d*Vst

		#Initialize the Next Loop.
		reconinit = recon_minTV						

		#Return the reconstruction. 
		return recon_constraint 		
					
def TVDerivative(img):

	fxy = np.pad(img, (1,1), 'constant', constant_values = np.mean(img))
	fxnegy = np.roll(fxy, -1, axis = 0)
	fxposy = np.roll(fxy, 1, axis = 0)
	fnegxy = np.roll(fxy, -1, axis = 1)
	fposxy = np.roll(fxy, 1, axis = 1)
	fposxnegy = np.roll( np.roll(fxy, 1, axis = 1), -1, axis = 0 )
	fnegxposy = np.roll( np.roll(fxy, -1, axis = 1), 1, axis = 0)
	vst1 = (2*(fxy - fnegxy) +
			 2*(fxy - fxnegy))/np.sqrt(1e-8 + (fxy - fnegxy)**2 + (fxy - fxnegy)**2)
	vst2 = (2*(fposxy - fxy))/np.sqrt(1e-8 + (fposxy - fxy)**2 + (fposxy - fposxnegy)**2)
	vst3 = (2*(fxposy - fxy))/np.sqrt(1e-8 + (fxposy - fxy)**2 + (fxposy - fnegxposy)**2)
	vst = vst1 - vst2 - vst3
	vst = vst[1:-1, 1:-1]
	vst = vst/np.linalg.norm(vst)
	return vst

def create_mask(dataset, wedgeSize, theta, kmin):

	(nx, ny) = dataset.shape

	# Convert missing wedge size and theta to radians.
	theta = (theta+90)*(np.pi/180)
	dtheta = wedgeSize*(np.pi/180)

	#Create coordinate grid in polar 
	x = np.arange(-nx/2, nx/2-1,dtype=np.float64)
	y = np.arange(-ny/2, ny/2-1,dtype=np.float64)
	[x,y] = np.meshgrid(x,y,indexing ='xy')
	rr = (np.square(x) + np.square(y))
	phi = np.arctan2(y,x) 
	phi *= -1

	#Create the Mask
	mask = np.ones( (ny, nx), dtype = np.int8 )
	mask[np.where((phi >= (theta-dtheta/2)) & (phi <= (theta+dtheta/2)))]  = 0
	mask[np.where((phi >= (np.pi+theta-dtheta/2)) & (phi <= (np.pi+theta+dtheta/2)))] = 0
	mask[np.where((phi >= (-np.pi+theta-dtheta/2)) & (phi <= (-np.pi+theta+dtheta/2)))] = 0
	mask[np.where(rr < np.square(kmin))] = 1 # Keep values below rmin.
	mask = np.array(mask, dtype = bool)
	mask = np.transpose(mask)
	return mask

def view_missing_wedge(dataset, wedgeSize, theta, kmin):
	
	FFT_raw = np.log(np.abs(fftshift(fftn(dataset))) + 1)

	mask = create_mask(dataset, wedgeSize, theta, kmin)

	sx = ndimage.sobel(mask, axis=0)
	sy = ndimage.sobel(mask, axis=1)
	mask_edge = np.hypot(1*sx,1*sy)
	mask_edge = np.ma.masked_where(mask_edge == 0, mask_edge)
	mask_edge[mask_edge > 0] = 1

	fig, ax = plt.subplots()
	ax.imshow(FFT_raw, cmap = 'bone')
	ax.set_title('FFT of Input Image')
	ax.pcolor(mask_edge, edgecolors='y', linewidths=1)
	ax.axis('off')
	plt.show()


