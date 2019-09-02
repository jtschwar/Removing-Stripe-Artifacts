from numpy.fft import fftshift, ifftshift
from matplotlib import pyplot as plt
from skimage import io
import FFTW
import numpy as np

fname = 'nacreous_domain2.tif' #file name

input_img = io.imread('sample_data/' + fname)
input_img = np.array(input_img, dtype=np.float32)

ttffw = FFTW.WrapFFTW(input_img.shape)

fft_img = np.log(np.abs(fftshift(ttffw.fft(input_img))) +1)
# fft_img = np.log(np.abs(fftshift(np.fft.fftn(input_img))) +1)


# plt.imshow(np.real(fft_img), cmap='bone')
# plt.show()