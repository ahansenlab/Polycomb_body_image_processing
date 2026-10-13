import numpy as np
import scipy.ndimage as ndi

from numba import jit


def to_smooth(signal, sigma=1):
    """
    Gaussian filters a signal with NaNs.

    Parameters
    ----------
    signal: numpy.ndarray
        The signal to smooth.
    sigma: float, optional
        Blurring factor.
    
    Returns
    -------
    numpy.ndarray
        Smoothed signal as array.
    """
    # find where signal has NaNs
    signal = np.copy(signal)
    invalid = np.isnan(signal)
    valid = ~invalid

    # set NaNs to zero then smooth
    signal[invalid] = 0
    signal = ndi.gaussian_filter(signal, sigma, mode="nearest")

    # normalize based on the mask and reset back to NaNs (if any)
    signal[valid] /= ndi.gaussian_filter(valid.astype(float), sigma, mode="nearest")[valid]
    signal[invalid] = np.nan

    return signal


@jit(nopython=True)
def to_msds(track):
    """
    Asdf.
    
    Parameters
    ----------
    track: numpy.ndarray
        Single track (3D) to compute MSDs for.
    
    Returns
    -------
    numpy.ndarray
        A 2D array of MSDs for each axis.
    numpy.ndarry
        An array of the number of samples for each lag.
    """
    # get max lags
    n = len(track)
    msds = np.zeros((n,3), dtype=np.float64)
    samples = np.zeros(n, dtype=np.int64)

    # compute all lags    
    for i in range(n):
        for j in range(n-i):
            # squared difference
            s = track[i+j] - track[j]
            if not np.isnan(s).any():
                msds[i] += s**2
                samples[i] += 1
        
        # normalize by samples
        if samples[i] > 0:
            msds[i] /= samples[i] 
        else:
            msds[i,:] = np.nan

    return msds, samples
