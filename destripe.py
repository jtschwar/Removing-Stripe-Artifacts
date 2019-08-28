from numpy.fft import fftn, fftshift, ifftn, ifftshift
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import io
import numpy as np


class destripe:

	def __init__(self, dataset, Niter, a, wedgeSize, theta, kmin):
		self.dataset = dataset
		self.Niter = Niter
		self.a = a
		self.wedgeSize = wedgeSize
		self.theta = theta
		self.kmin = kmin

	def TV_reconstruction(self, save): 

		#dataset - real-space image (pixels)
		#Niter - Number of iterations for reconstruction
		# a - Decent parameter (unitless, typically 0 – 0.3)
		# wedgeSize - angular range of the missing wedge (degrees)
		# theta - orientation of the missing wedge (degrees)
		# kmin - Minimum frequency to start the missing wedge (1/px)

		#Import dimensions from dataset
		(nx, ny) = self.dataset.shape
			
		mask = self.create_mask()

		#FFT of the Original Image
		FFT_image = fftshift(fftn(self.dataset)) 	

		# Reconstruction starts as random image.		
		recon_init = np.random.rand(nx,ny) 						

		#Artifact Removal Loop
		for i in range(self.Niter):

			print('Iteration No.: ' + str(i+1) +'/'+str(self.Niter))

			#FFT of Reconstructed Image.
			FFT_recon = fftshift(fftn(recon_init)) 		

			#Data Constraint
			FFT_recon[mask] = FFT_image[mask] 			

			#Inverse FFT
			recon_constraint = np.real(ifftn(ifftshift(FFT_recon)))

			#Positivity Constraint 
			recon_constraint[ recon_constraint < 0 ] = 0

			if i < self.Niter -1:  	#ignore on last iteration	

				#TV Minimization
				# The basis for TVDerivative (20 iterations and epsilon = 1e-8) was determined by Sidky (2006). 
				recon_minTV = recon_constraint.copy()
				d = np.linalg.norm(recon_minTV - recon_init)
				for j in range(20):
					Vst = self.TVDerivative(recon_minTV)
					recon_minTV = recon_minTV - self.a * d * Vst

				#Initialize the Next Loop.
				recon_init = recon_minTV.copy()						

		if save:
			recon_constraint = recon_constraint/np.amax(recon_constraint)*255
			io.imsave('recon.tif', np.uint8(recon_constraint))

		#Show reconstruction
		fig, ax = plt.subplots()
		ax.imshow(recon_constraint, cmap='gray')
		ax.axis('off')
		ax.set_title('Reconstruction')	
		plt.show()
						
	def TVDerivative(self, img):

		fxy = np.pad(img, (1,1), 'constant', constant_values = np.mean(img))
		fxnegy = np.roll(fxy, -1, axis = 0)
		fxposy = np.roll(fxy, 1, axis = 0)
		fnegxy = np.roll(fxy, -1, axis = 1)
		fposxy = np.roll(fxy, 1, axis = 1)
		fposxnegy = np.roll(fxy, [-1,1], axis=(0,1))
		fnegxposy = np.roll(fxy, [1,-1], axis=(0,1))
		
		vst1 = (4*fxy - 2*(fnegxy + fxnegy))/np.sqrt(1e-8 + (fxy - fnegxy)**2 + (fxy - fxnegy)**2)
		vst2 = (2*(fposxy - fxy))/np.sqrt(1e-8 + (fposxy - fxy)**2 + (fposxy - fposxnegy)**2)
		vst3 = (2*(fxposy - fxy))/np.sqrt(1e-8 + (fxposy - fxy)**2 + (fxposy - fnegxposy)**2)
		
		vst = vst1 - vst2 - vst3
		vst = vst[1:-1, 1:-1]
		vst = vst/np.linalg.norm(vst)
		return vst

	def create_mask(self):

		(nx, ny) = self.dataset.shape

		if self.theta > 90 or self.theta < -90:
			raise ValueError('Please keep theta between +/- 90 degrees.') 

		# Convert missing wedge size and theta to radians.
		rad_theta = (self.theta+90)*(np.pi/180)
		dtheta = self.wedgeSize*(np.pi/180)

		#Create coordinate grid in polar 
		x = np.arange(-nx/2, nx/2-1,dtype=np.float64)
		y = np.arange(-ny/2, ny/2-1,dtype=np.float64)
		[x,y] = np.meshgrid(x,y,indexing ='xy')
		rr = (np.square(x) + np.square(y))
		phi = np.arctan2(y,x) 
		phi *= -1

		#Create the Mask
		mask = np.ones( (ny, nx), dtype = np.int8 )
		mask[np.where((phi >= (rad_theta-dtheta/2)) & (phi <= (rad_theta+dtheta/2)))]  = 0
		mask[np.where((phi >= (np.pi+rad_theta-dtheta/2)) & (phi <= (np.pi+rad_theta+dtheta/2)))] = 0
		mask[np.where((phi >= (-np.pi+rad_theta-dtheta/2)) & (phi <= (-np.pi+rad_theta+dtheta/2)))] = 0
		mask[np.where(rr < np.square(self.kmin))] = 1 # Keep values below rmin.
		mask = np.array(mask, dtype = bool)
		mask = np.transpose(mask)
		return mask

	def view_missing_wedge(self):
		
		FFT_raw = np.log(np.abs(fftshift(fftn(self.dataset))) + 1)

		mask = self.create_mask()

		sx = ndimage.sobel(mask, axis=0)
		sy = ndimage.sobel(mask, axis=1)
		mask_edge = np.hypot(1*sx,1*sy)
		mask_edge = np.ma.masked_where(mask_edge == 0, mask_edge)
		mask_edge[mask_edge > 0] = 1

		fig, (ax1,ax2) = plt.subplots(1,2, figsize=(7,3))
		ax1.imshow(self.dataset, cmap='bone')
		ax1.set_title('Input Image')
		ax1.axis('off')
		ax2.imshow(FFT_raw, cmap = 'bone')
		ax2.set_title('FFT of Input Image')
		ax2.pcolor(mask_edge, edgecolors='y', linewidths=1)
		ax2.axis('off')
		plt.tight_layout()
		plt.draw()

	def update_missing_wedge(self):

		plt.clf()
		FFT_raw = np.log(np.abs(fftshift(fftn(self.dataset))) + 1)

		mask = self.create_mask()

		sx = ndimage.sobel(mask, axis=0)
		sy = ndimage.sobel(mask, axis=1)
		mask_edge = np.hypot(1*sx,1*sy)
		mask_edge = np.ma.masked_where(mask_edge == 0, mask_edge)
		mask_edge[mask_edge > 0] = 1

		ax = plt.gca()
		ax1 = plt.subplot(121, frameon=False)
		ax1.imshow(self.dataset, cmap='bone')
		ax1.set_title('Input Image')
		ax1.axis('off')
		ax2 = plt.subplot(122, frameon=False)
		ax2.imshow(FFT_raw, cmap = 'bone')
		ax2.set_title('FFT of Input Image')
		ax2.pcolor(mask_edge, edgecolors='y', linewidths=1)
		ax2.axis('off')
		plt.tight_layout()
		plt.draw()

	def edit_wedgeSize(self, new_wedgeSize):
		self.wedgeSize = float(eval(new_wedgeSize))
		self.update_missing_wedge()

	def edit_theta(self, new_theta):
		self.theta = float(eval(new_theta))
		self.update_missing_wedge()

	def edit_kmin(self, new_kmin):
		self.kmin = float(eval(new_kmin))
		self.update_missing_wedge()

	def edit_niter(self, new_niter):
		self.Niter = int(new_niter)
		self.update_missing_wedge()	

	def get_params(self):
		return int(self.wedgeSize), int(self.theta), int(self.kmin), self.Niter

	def check_matplotlib_version(self):
		import matplotlib
		version = float(matplotlib.__version__[0:3])
		if version < 2.1:
			print('Matplotlib is unable to run script, please update.')
			raise ValueError('Please update matplotlib version above 2.1.') 
