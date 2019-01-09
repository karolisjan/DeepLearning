'''
    Ref.:
    
        http://jseabold.net/blog/2012/02/23/wavelet-regression-in-python/
        http://connor-johnson.com/2016/01/24/using-pywavelets-to-remove-high-frequency-noise/
'''
import pywt
import numpy as np
from statsmodels.robust import mad


def wavelet_transform(
    x,
    wavelet='haar',
    level=1,
    mode='symmetric'
):
    # Decompes using Wavelet Transform
    coeff = pywt.wavedec(x, wavelet=wavelet, mode=mode)
    
    # Calculate a robust estimator of the standar deviation
    sigma_hat = mad(coeff[-level])
    
    # Calculate a universal threshold
    u_thresh = sigma_hat * np.sqrt(2 * np.log(len(x)))
    
    # Threshold the coefficients
    coeff[1:] = (pywt.threshold(i, value=u_thresh, mode='soft') for i in coeff[1:])
    
    # Reconstruct the signal using the thresholded coefficients
    denoised_x = pywt.waverec(coeff, wavelet=wavelet, mode=mode)
    
    return denoised_x
    
    
    