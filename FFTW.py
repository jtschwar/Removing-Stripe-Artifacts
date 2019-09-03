# 
import numpy as np
import sys, os
try:
    import pyfftw
    hasfftw = True
except ImportError:
    print('PYFFTW not found, please run "pip install pyfftw" for up to 20x speedup')
    hasfftw = False


class WrapFFTW(object):
  def __init__(self, shape, **kwargs):
      self.shape = shape

      self._flags = kwargs.get('flags', ['FFTW_MEASURE'])
      self._threads = kwargs.get('threads', 8)

      self.data = pyfftw.empty_aligned(self.shape, n=16, dtype='complex64')
      self.data_k = pyfftw.empty_aligned(self.shape, n=16, dtype='complex64')

      self.fft_object = pyfftw.FFTW(self.data, self.data_k,
                              axes=(0,1), flags = self._flags,
                              threads = self._threads)
      self.ifft_object = pyfftw.FFTW(self.data_k, self.data,
                               direction = 'FFTW_BACKWARD',
                               axes=(0,1), flags = self._flags,
                               threads = self._threads)

  def fft(self, inp):
      self.data[:,:] = inp
      return self.fft_object().copy()

  def ifft(self, inp):
      self.data_k[:,:] = inp
      return self.ifft_object().copy()


class WrapFFTW_NUMPY(object):
  def __init__(self, shape, **kwargs):
      self.shape = shape

  def fft(self, inp):
      return np.fft.fftn(inp)

  def ifft(self, inp):
      return np.fft.ifftn(inp) 


if not hasfftw:
    WrapFFTW = WrapFFTW_NUMPY

