import os
import re
import h5py
import math
import yaml
import shutil
import colorsys
import argparse
import cupy as cp
import numpy as np
import random as r
import tifffile as tiff
import matplotlib.pyplot as plt
import cupyx.scipy.ndimage as ndi

from functools import wraps
from numba import jit, prange
from datetime import datetime
from aicspylibczi import CziFile
from collections.abc import Iterable
from cupyx.scipy.fft import fftn, ifftn, fftfreq


def configure_plots():
    """
    Sets matplotlib formatting.

    Parameters
    ----------
    None
    """
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["font.size"] = 15
    plt.rcParams["xtick.major.width"] = 2
    plt.rcParams["ytick.major.width"] = 2
    plt.rcParams["xtick.minor.width"] = 2
    plt.rcParams["ytick.minor.width"] = 2
    plt.rcParams["axes.linewidth"] = 2
    plt.rcParams["axes.titlesize"] = 15


def to_fmt(number):
    """
    Defines number formatting for the
    pipeline.
    
    Parameters
    ----------
    number : int
        Number to format as string.
    
    Returns
    -------
    str
        Formatted string.
    """
    digits = 3
    
    return f"{number:0{digits}d}"


def make_color():
    """
    Samples color for segmentation labels.

    Parameters
    ----------
    None
    """
    rgb = colorsys.hsv_to_rgb(r.random(), 0.5, 0.9)
    
    return np.asarray(rgb)


def read_options():
    """
    Reads command-line arguments for modules.
    
    Returns
    -------
    dict
        Dictionary containing command-line options.
    """ 
    # make parser and add options
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file")
    parser.add_argument("-b", "--beads")
    parser.add_argument("-p", "--psf")
    parser.add_argument("-c", "--config")
    parser.add_argument("-d", "--directory")
    parser.add_argument("-s", "--start", type=int, default=0)
    parser.add_argument("-e", "--end", type=int)
    parser.add_argument("-n", "--nproc", type=int, default=8)
    parser.add_argument("-g", "--gpus", type=int, default=1)
    parser.add_argument("-m", "--mode")
    parser.add_argument("-o", "--other")
    parser.add_argument("-F", "--force", action="store_true")
    
    # read and convert to dictionary
    options = vars(parser.parse_args())

    return options


def to_time(frame, dt):
    """
    Formats timestamp based on frame and
    time step.

    Parameters
    ----------
    frame : int
        Current frame index.
    dt : float
        Time step in seconds.

    Returns
    -------
    str
        Formatted timestamp.
    """ 
    h, remainder = divmod(frame * dt, 3600)
    m, s = divmod(remainder, 60)
    
    return f"Time: {int(h):02}h {int(m):02d}m {int(s):02d}s"


def to_directory(file):
    """
    Defines the default naming for the
    output directory.
    
    Parameters
    ----------
    file : str
        Path to CZI file.
    
    Returns
    -------
    str
        Path for pipeline outputs.
    """
    directory, _ = os.path.splitext(file)
    
    return directory


def to_end(file):
    """
    Gets the last frame of a CZI.

    Parameters
    ----------
    file : str
        Path to CZI file.
    
    Returns
    -------
    int
        The last frame of the CZI.
    """
    end = CziFile(file).get_dims_shape()[0]["T"][1]
    
    return end


def to_date(path, pattern=r"\d{8}", fmt="%Y%m%d"):
    """
    Converts path to datetime object.

    Parameters
    ----------
    path : str
        Path to search for date in.
    pattern : str, optional
        Pattern of the date as a regular expression.
    fmt : str, optional
        Format of the year, month, and day.
    
    Returns
    -------
    datetime.datetime
        Object representing date.
    """
    return datetime.strptime(re.search(pattern, path).group(), fmt)


def to_mip(signal, mask=None, intensity={}, color=None,
           gray=False, background=None, axis=0):
    """
    Makes a maximum intensity projection (MIP) from
    frame, applying color if desired.

    Parameters
    ----------
    signal : numpy.ndarray
        Array of signal.
    mask : numpy.ndarray, optional
        Boolean mask for frame.
    intensity : dict, optional
        Dictionary of intensity bounds.
    color : None or iterable, optional
        Color of MIP.
    gray : bool, optional
        Option to force grayscale.
    background : iterable, optional
        Color to set the background (outside mask).
    axis : int, optional
        Axis by which MIP is computed.

    Returns
    -------
    numpy.ndarray
        MIP image as a 3D array.
    """
    # make MIP and get bounds
    signal = signal.astype(float)
    if mask is not None:
        signal = signal * mask
    mip = np.max(signal, axis=axis)
    lower = intensity["lower"] if "lower" in intensity else np.min(mip)
    upper = intensity["upper"] if "upper" in intensity else np.max(mip)

    # impose bounds
    mip = np.clip(mip, lower, upper)
    mip = (mip - lower) / (upper - lower)  # normalize

    # add color and set background (if desired)
    c = 255*np.ones(3) if color is None or gray else np.array(color)
    mip = (mip[..., np.newaxis] * c).astype(np.uint8)
    if (mask is not None) and (background is not None):
        mip[~np.max(mask, axis=axis)] = background
    
    return mip


def save_config(config, directory):
    """
    Saves YAML config file in directory.
    
    Parameters
    ----------
    config : str
        Path to YAML config file.
    directory : str
        Path for pipeline outputs.
    """
    shutil.copyfile(config, os.path.join(directory, 
                    os.path.basename(config)))


def read_yaml(path):
    """
    Reads YAML file and handles missing/empty
    parameters with None.

    Parameters
    ----------
    path : str
        Path to the file.
    
    Returns
    -------
    dict
        Read dictionary.
    """ 
    def fix_yaml(item):
        # function to process read YAML file (recursively)
        if isinstance(item, dict):
            return {key: fix_yaml(value)
                    if value is not None else {}
                    for key, value in item.items()}
        elif isinstance(item, list):
            return [fix_yaml(i) for i in item]
        else:
            return item

    # read the YAML file
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return fix_yaml(data)


def write_yaml(path, data):
    """
    Writes YAML file.

    Parameters
    ----------
    path : str
        Path to the directory.
    data : dict
        Dictionary to save.
    """ 
    # write the YAML file
    with open(path, "w") as f:
        yaml.dump(data, f)


def read_beads(path):
    """
    Reads a TXT file of bead files.

    Parameters
    ----------
    path : str
        Path to TXT file of bead files.
    
    Returns
    -------
    list
        List (str) of bead files.
    """
    with open(path, "r") as f:
        paths = [line.strip() for line in f if line.strip()]
                
    return paths


def read_czi(path, frame):
    """
    Read CZI at frame.

    Parameters
    ----------
    path : str
        Path to the CZI.
    frame : int
        Index of frame to read.
    
    Returns
    -------
    numpy.ndarray
        The frame as an array.
    """
    # read frame
    czi = CziFile(path)
    signal, _ = czi.read_image(T=frame, cores=1)
    del czi

    # ensure at least 4D (channels and 3D volume)
    signal = np.squeeze(signal)
    signal = np.array(signal, ndmin=4, dtype=np.float32)

    return signal

def read_h5(path, key="data"):
    """
    Reads (compressed) h5 file to NumPy array.

    Parameters
    ----------
    path : str
        Path to the H5 file.
    key : str, optional
        Key for data.

    Returns
    -------
    numpy.ndarray
        Saved array data.
    """ 
    with h5py.File(path, "r") as f:
        return f[key][:]


def write_h5(path, data, key="data"):
    """
    Writes a (compressed) H5 file.

    Parameters
    ----------
    path : str
        Path to the H5 file.
    data : numpy.ndarray
        The data to save.
    key : str, optional
        Key for data.
    mode : str, optional
        Mode for opening the file.
    """
    with h5py.File(path, "w") as file:
        file.create_dataset(key, data=data, compression="gzip")


def read_kernel(path, lasers={}):
    """
    Read the PSF kernel from a (compressed)
    H5 file.

    Parameters
    ----------
    path : str
        Path to the PSF H5 file.
    lasers : dict
        Dictionary of lasers and channels.
    Returns
    -------
    numpy.ndarray
        PSF as a volume of intensities.
    """
    # holders for each channel
    n = len(lasers)
    kernel = [None for _ in range(n)]

    # read groups
    with h5py.File(path, "r") as f:
        for chnl, laser in lasers.items():
            key = str(laser)
            kernel[chnl] = f["kernel"][key][:]

    return np.asarray(kernel)  # concatenate


def read_shape(path, lasers={}):
    """
    Read the PSF shape from a (compressed)
    H5 file.

    Parameters
    ----------
    path : str
        Path to the PSF H5 file.
    lasers : dict
        Dictionary of lasers and channels.
    Returns
    -------
    numpy.ndarray
        Mask shape as a "covariance" matrix
    """
    # holders for each channel
    n = len(lasers)
    shape = [None for _ in range(n)]

    # read groups
    with h5py.File(path, "r") as f:
        for chnl, laser in lasers.items():
            key = str(laser)
            shape[chnl] = f["shape"][key][:]
            
    return np.asarray(shape)  # concatenate


def write_psf(path, kernel, shape, lasers={}):
    """
    Save the PSF as a (compressed) H5 file
    with both signal and mask shape.

    Parameters
    ----------
    path : str
        Path to the PSF H5 file.
    kernel : numpy.ndarray
        PSF as a volume of intensities.
    shape : numpy.ndarray
        Mask shape as a "covariance" matrix
    lasers : dict
        Dictionary of lasers and channels.
    """
    with h5py.File(path, "w") as f:
        # create groups and write each channel
        k, s = f.create_group("kernel"), f.create_group("shape")
        for chnl, laser in lasers.items():
            key = str(laser)
            k.create_dataset(key, data=kernel[chnl], compression="gzip")
            s.create_dataset(key, data=shape[chnl], compression="gzip")


def read_tiff(path):
    """
    Reads TIFF file to NumPy array.

    Parameters
    ----------
    path : str
        Path to the  file.

    Returns
    -------
    numpy.ndarray
        Saved array data.
    """ 
    return tiff.imread(path).astype(np.float32)


def write_tiff(path, data):
    """
    Writes a (compressed) TIFF file.

    Parameters
    ----------
    path : str
        Path to the TIFF file.
    data : numpy.ndarray
        The data to save.
    """
    tiff.imwrite(path, data, compression="zlib")


def proc(func):
    """
    Decorator for expanding arguments when using
    multiprocessing.

    Parameters
    ----------
    func : callable
        The function to be wrapped.
    
    Returns
    -------
    callable
        The wrapped function.
    """ 
    @wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) == 1 and isinstance(args[0], Iterable):
            return func(*args[0])
        else:
            return func(*args, **kwargs)
            
    return wrapper


def mkdir(path):
    """
    Force empty directory by removing the
    existing (if necessary).

    Parameters
    ----------
    path : str
        Path to the directory.
    
    Returns
    -------
    str
        The same path to the directory.
    """ 
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    
    return path


def where(mask):
    """
    Finds the first index where a 1D array
    is true.

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean mask.
    
    Returns
    -------
    int or None
        First index where mask is true.
    """
    idxs = np.argwhere(mask)[:,0]
    if len(idxs) != 0:
        return idxs[0]
    else:
        return None


def bar_format(unit):
    """
    Modifies tqdm's progress bar formatting to
    include units.

    Parameters
    ----------
    unit : str
        Units of the progress bar.
    
    Returns
    -------
    str
        Format for progress bar.
    """
    fmt = "{l_bar}{bar}| {n_fmt}/{total_fmt} " + unit + \
          " [{elapsed}<{remaining}, {rate_fmt}{postfix}]"

    return fmt

def to_isotropic(signal, pixel=145, step=300, order=1):
    """
    Resamples a stack to isotropic pixel sizes.

    Parameters
    ----------
    signal : numpy.ndarray
        Array to deskew.
    pixel : float, optional
        Size of the isotropic pixel (nm).
    step : float, optional
        Step length of the sheets (nm).
    order : int, optional
        Order of the interpolation.

    Returns
    -------
    numpy.ndarray
        The isotropic volume.
    """
    # get shape and dtype
    chnls, z, y, x = signal.shape
    dtype = cp.float32

    # get ratio and shape
    a = pixel/step
    shape = (math.ceil(z/a), y, x)

    # compute inverse transform
    inverse = cp.array([[a, 0, 0, (1-1/a)/2*a],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=dtype)
    #print(inverse)

    # make resampled holder
    volume = cp.empty(shape, dtype=dtype)
    resample = np.empty((chnls, *shape), dtype=dtype)
    for chnl in range(chnls):
        # apply transformation
        ndi.affine_transform(cp.array(signal[chnl]), inverse,
                             output_shape=shape, output=volume, order=order)
        resample[chnl] = volume.get()
    del volume
    cp.get_default_memory_pool().free_all_blocks()

    return resample


@jit(nopython=True)
def to_ellipsoid(shape, radius=None):
    """
    Makes mask (structuring element) defined
    by a positive semidefinite matrix.
    
    Parameters
    ----------
    shape : numpy.ndarray
        Shape of the mask as a "covariance" matrix
    radius : list, tuple, or numpy.ndarray, optional
        Radii for each axis of volume.
    
    Returns
    -------
    numpy.ndarray
        Mask (structuring element) as a NumPy array.
    """
    # get shape of volume
    shape = shape.astype(np.float64)
    if radius is None:
        m = np.ceil(np.diag(shape)**0.5).astype(np.int64)
    else:
        m = np.empty(3, dtype=np.int64)
        for axis in range(3):
            m[axis] = radius[axis]

    # fill out mask
    s = np.empty(3)
    inverse = np.linalg.inv(shape)
    mask = np.zeros((2*m[0]+1, 2*m[1]+1, 2*m[2]+1), dtype=np.bool)
    for i in range(-m[0], m[0]+1):
        s[0] = i
        for j in range(-m[1], m[1]+1):
            s[1] = j
            for k in range(-m[2], m[2]+1):
                s[2] = k
                if s.T @ inverse @ s <= 1:  # if in mask
                    mask[m[0]+i, m[1]+j, m[2]+k] = 1

    return mask


def to_log(signal, sigma):
    """
    Defines Laplacian of Gaussian (LoG)
    filter with "nearest" interpolation.

    Parameters
    ----------
    signal : cupy.ndarray
        Input signal.
    sigma : float or int
        Standard deviation of the Gaussian kernel.
    
    Returns
    -------
    cupy.ndarray
        Filtered signal.
    """
    # Laplacian of Gaussian filter
    s = ndi.gaussian_filter(signal, sigma, mode="nearest")
    s = ndi.laplace(s, mode="nearest")
    
    return s


def to_gpu(source):
    """
    Alternative method to send arrays to a device.

    Parameters
    ----------
    source : numpy.ndarray
        The array sending to device.
    
    Returns
    -------
    cupy.ndarray
        The transferred device array.
    """ 
    target = cp.empty_like(source)
    cp.cuda.runtime.memcpy(target.data.ptr, source.ctypes.data, source.nbytes,
                           cp.cuda.runtime.memcpyHostToDevice)
    return target


def deskewing(signal, angle=30, pixel=145, step=300, order=1):
    """
    Deskews and coverslip transforms a raw volume.

    Parameters
    ----------
    signal : numpy.ndarray
        Array to deskew.
    angle : float, optional
        Angle (degrees) of the sheet.
    pixel : float, optional
        Size of the isotropic pixel (nm).
    step : float, optional
        Step length of the sheets (nm).
    order : int, optional
        Order of the interpolation.
    
    Returns
    -------
    numpy.ndarray
        The deskewed volume.
    """
    # get shape and dtype
    chnls, z, y, x = signal.shape
    dtype = cp.float32

    # get ratio and shape
    a = step/pixel
    theta = angle * math.pi/180
    s, c = math.sin(theta), math.cos(theta)
    shape = (int(math.ceil(s*y)), int(math.ceil(c*y+a*z)), x)

    # compute transformation matrix
    transform = cp.array([[a*c*(0.5-s), -s, 0, s*(y-1)-0.5*a*c*(0.5-s)*(z-1)],
                          [-a*(0.5*s+c**2), -c, 0, c*(y-1)+0.5*a*(1+0.5*s+c**2)*(z-1)],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=dtype)
    inverse = cp.linalg.inv(transform)  # inverse transform

    # make deskew holder
    deskew = np.empty((chnls, shape[0], shape[2], shape[1]), dtype=dtype)
    for chnl in range(chnls):
        # apply transformation
        affine = ndi.affine_transform(to_gpu(signal[chnl]), inverse,
                                      output_shape=shape, order=order)
        deskew[chnl] = np.transpose(affine.get(), (0,2,1))  # change axes
        del affine
    cp.get_default_memory_pool().free_all_blocks()
    
    return deskew


def deconvolution(signal, kernel, iters=10, alpha=5e-3):
    """
    Deconvolves a volume with a PSF using a Richardson-Lucy
    method with total-variation regulariztion.

    Parameters
    ----------
    signal : numpy.ndarray
        signal to deconvolve.
    kernel : numpy.ndarray
        Kernel describing the PSF in each channel.
    iters : int, optional
        Number of iterations.
    alpha : float, optional
        Regularization constant. 
    
    Returns
    -------
    numpy.ndarray
        The deconvolved signal.
    """    
    # get shape and dtype
    if len(signal.shape) == 4:
        chnls, z, y, x = signal.shape
    elif len(signal.shape) == 3:
        z, y, x = signal.shape
        chnls = 1
    dtype, eps = cp.float32, 1e-12
    
    # get mirrored PSF
    kernel = cp.array(kernel, dtype=dtype)
    kernel_reverse = cp.copy(kernel)
    for axis in range(1,4):
        kernel_reverse = cp.flip(kernel_reverse, axis=axis)

    # make holders
    shape = (z, y, x)
    ratio, conv = cp.zeros(shape, dtype=dtype), cp.zeros(shape, dtype=dtype)
    grads, estimate = cp.empty((3, *shape), dtype=dtype), cp.empty(shape, dtype=dtype)

    # run deconvolution
    deconvolve = np.empty_like(signal)
    for chnl in range(chnls):
        # get channel and mask
        s = np.copy(signal[chnl])
        mask = s > 0
        s = to_gpu(s)  # send to GPU
        
        # run iterations
        estimate[:] = s
        for i in range(iters):
            # convolve with PSFs
            conv = ndi.convolve(estimate, kernel[chnl], mode="nearest")
            ratio = s / cp.maximum(conv, eps)
            ratio = ndi.convolve(ratio, kernel_reverse[chnl], mode="nearest")
            
            # get normalized gradients
            for axis in range(3):
                grads[axis] = cp.gradient(estimate, axis=axis)
            grads /= cp.maximum(cp.linalg.norm(grads, axis=0), eps)
    
            # compute divergence
            for axis in range(3):
                grads[axis] = cp.gradient(grads[axis], axis=axis)
            divergence = cp.sum(grads, axis=0)
    
            # perform update
            estimate *= ratio / cp.maximum(1 - alpha*cp.abs(divergence), eps)
            del divergence
            estimate[estimate < eps] = eps
            
        # enforce zeros and update
        deconvolve[chnl] = mask * estimate.get()
        del s
    del ratio
    del conv
    del grads
    del estimate
    cp.get_default_memory_pool().free_all_blocks()
        
    return deconvolve


@jit(nopython=True, parallel=True)
def to_fill(signal):
    """
    Fills in wedges in a deskewed volume.

    Parameters
    ----------
    signal : numpy.ndarray
        Volume to fill in.
    
    Returns
    -------
    numpy.ndarray
        The filled volume.
    """
    # find lowest non-zero for padding
    lowest = -1
    for i in range(signal.shape[0]):
        for j in range(signal.shape[1]):
            for k in range(signal.shape[2]):
                # update non-zero lowest
                if signal[i,j,k] > 0:
                    if lowest == -1:
                        lowest = signal[i,j,k]
                    else:
                        if signal[i,j,k] < lowest:
                            signal[i,j,k] = lowest
    
    # run the following in parallel
    for i in prange(signal.shape[0]):
        for j in range(signal.shape[1]):
            # find region start/end
            start, end = -1, -1
            for k in range(signal.shape[2]):
                if signal[i,j,k] != 0:
                    if start == -1:
                        start = k
                    end = k

            # nearest non-zero (from left)
            fill = signal[i,j,start]
            if fill != 0:
                signal[i,j,:start] = fill
            else:
                signal[i,j,:start] = lowest

            # nearest non-zero (from right)
            fill = signal[i,j,end]
            if fill != 0:
                signal[i,j,end+1:] = fill
            else:
                signal[i,j,end+1:] = lowest
    
    return signal


@jit(nopython=True)
def to_volume(signal, position, radius=5):
    """
    Makes volume subset.

    Parameters
    ----------
    signal : numpy.ndarray
        Array to subset.
    position : numpy.ndarray
        Position for subset center.
    radius : list, tuple, or numpy.ndarray, optional
        Radii for each axis of subsetted volume.
    
    Returns
    -------
    numpy.ndarray
        Subset of signal.
    numpy.ndarray
        Lower bounds of subset.
    numpy.ndarray
        Upper bounds of subset.
    """
    # get bounds
    position = np.rint(position).astype(np.int64)
    lower, upper  = position-radius, position+radius+1

    # apply bounds
    lower[lower < 0] = 0
    upper[upper < 0] = 0
    for axis in range(3):
        bound = signal.shape[axis]
        if lower[axis] > bound:
            lower[axis] = bound
        if upper[axis] > bound:
            upper[axis] = bound

    # get volume
    volume = signal[lower[0]:upper[0],lower[1]:upper[1],lower[2]:upper[2]]

    return np.copy(volume), lower, upper


@jit(nopython=True)
def to_pwrs():
    """
    Makes a polynomial basis for regression.
    
    Returns
    -------
    numpy.ndarray
        Powers of the polynomial basis.
    """
    # polynomial basis (order two)
    pwrs = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1],
                     [2,0,0], [1,1,0], [1,0,1], [0,2,0],
                     [0,1,1], [0,0,2]])
    return pwrs


@jit(nopython=True)
def mapping(x):
    """
    Makes a polynomial basis for a single input.

    Parameters
    ----------
    x : numpy.ndarray
        Input position.
    
    Returns
    -------
    numpy.ndarray
        The mapped position.
    """
    # make mapping
    pwrs = to_pwrs()
    mapped= np.empty(len(pwrs), dtype=x.dtype)
    for i in range(len(pwrs)):
        mapped[i] = np.prod(np.power(x, pwrs[i]))
    
    return mapped


@jit(nopython=True)
def mappings(x):
    """
    Makes a polynomial basis for multiple inputs.

    Parameters
    ----------
    x : numpy.ndarray
        Input positions
    
    Returns
    -------
    numpy.ndarray
        The mapped positions.
    """
    # make mappings
    pwrs = to_pwrs()
    mapped = np.empty((len(x), len(pwrs)), dtype=x.dtype)
    for i in range(len(x)):
        for j in range(len(pwrs)):
            mapped[i,j] = np.prod(np.power(x[i], pwrs[j]))
    
    return mapped


@jit(nopython=True)
def ridge(a, b, alpha=1e-3):
    """
    Runs ridge regression on inputs.

    Parameters
    ----------
    a : numpy.ndarray
        Source for regression.
    b : numpy.ndarray
        Target for regression.
    alpha : float
        Ridge parameter.
    
    Returns
    -------
    numpy.ndarray
        Weights of the fit.
    """
    l = alpha * np.identity(a.shape[1])
    l = l.astype(a.dtype)  # force type
    l[0,0] = 0  # ignore bias
    w = np.linalg.inv(a.T @ a + l) @ a.T @ b
        
    return w


@jit(nopython=True)
def to_intensity(signal, position, shape, mask=None, 
                 radius=5, alpha=1e-3):
    """
    Estimates intensity of inner shell with
    a polynomial fit from an outer shell.

    Parameters
    ----------
    signal : numpy.ndarray
        Signal subset.
    position : numpy.ndarray
        Center of shells.
    shape : numpy.ndarray
        Shape of PSF mask for channel.
    mask : numpy.ndarray, optional
        Mask subset.
    radius : int, optional
        Radius of subsetted volume.
    alpha : float, optional
        Ridge parameter.

    Returns
    -------
    float
        Background estimated intensity.
    """
    # make volume (and force dtype)
    volume, lower, _ = to_volume(signal, position, radius)
    dtype = volume.dtype
    position, lower = position.astype(dtype), lower.astype(dtype)

    # apply mask if given
    if mask is not None:
        mask, _, _ = to_volume(mask, position, radius)
        volume *= mask

    # hold signals and positions
    signals = np.empty(volume.size, dtype=dtype)
    positions = np.empty((volume.size, 3), dtype=np.int32)

    # fill holders
    idx = 0
    for i in range(volume.shape[0]):
        for j in range(volume.shape[1]):
            for k in range(volume.shape[2]):
                # make sure pixel is valid
                if volume[i,j,k] > 0:
                    signals[idx] = volume[i,j,k]
                    positions[idx] = (i,j,k)
                    idx += 1
    # remove extra
    signals = signals[:idx]
    positions = positions[:idx]

    # find points in inside/outside regions
    s = positions - (position - lower)
    inv = np.linalg.inv(6.25*shape).astype(dtype)
    inner = np.sum((s @ inv) * s, axis=1) < 1
    outer = ~inner

    # fit mapping to outer points
    mapped = mappings(s)
    w = ridge(mapped[outer], signals[outer], alpha)

    # estimate inner background and subtract
    background = np.sum(mapped[inner] @ w)
    intensity = np.sum(signals[inner]) - background
    
    return intensity, background


@jit(nopython=True)
def fracshift(signal, position, radius=5, iters=30, subtract=False,
              mask=None, shape=np.diag((9,9,9)), alpha=1e-3):
    """
    Method for running fracshift on a volume
    of intensities.

    Parameters
    ----------
    signal : numpy.ndarray
        Array to subset and run fracshift on.
    position : numpy.ndarray
        Position for subset center.
    radius : int, optional
        Buffer radius of around position.
    iters : int, optional
        Maximum fracshift iterations.
    subtract : bool, optional
        If using background subtraction.
    mask : numpy.ndarray, optional
        Subset of nucleus mask.
    alpha : float, optional
        Ridge parameter.
    
    Returns
    -------
    numpy.ndarray
        Estimated subpixel position.
    bool
        If estimated position stayed in bounds.
    """
    # make volume (and force dtype)
    volume, lower, _ = to_volume(signal, position, radius)
    dtype = volume.dtype
    position, lower = position.astype(dtype), lower.astype(dtype)

    # if using background subtraction
    if subtract:
        # hold signals and positions
        signals = np.empty(volume.size, dtype=dtype)
        positions = np.empty((volume.size, 3), dtype=np.int32)
    
        # fill holders
        idx = 0
        for i in range(volume.shape[0]):
            for j in range(volume.shape[1]):
                for k in range(volume.shape[2]):
                    # make sure pixel is valid
                    if volume[i,j,k] > 0:
                        signals[idx] = volume[i,j,k]
                        positions[idx] = (i,j,k)
                        idx += 1
        # remove extra
        signals = signals[:idx]
        positions = positions[:idx]

        # find points in inside/outside regions
        s = positions - (position - lower)
        inv = np.linalg.inv(6.25*shape).astype(dtype)
        inner = np.sum((s @ inv) * s, axis=1) < 1
        outer = ~inner
        
        # fit mapping to outer points
        mapped = mappings(s)
        w = ridge(mapped[outer], signals[outer], alpha)
    
        # subtract estimates
        signals -= mapped @ w
        signals[signals < 0] = 0
        for i in range(len(positions)):
            z, y, x = positions[i]
            volume[z, y, x] = signals[i]

    # mask shape (twice covariance of 488nm PSF)
    # shape = np.array([[17.0, 0., -3.7],
    #                   [ 0., 8.6,  0.],
    #                   [-3.7, 0., 12.7]], dtype=dtype)
    shape = np.array([[25.0, 0., 0.],
                      [0., 9.0,  0.],
                      [0., 0., 9.0]], dtype=dtype)
    mask = to_ellipsoid(shape)
    center = 1/2*(np.array(mask.shape)-1)

    # get integer/float positions of mask
    integers = np.argwhere(mask) - center.astype(np.int32)
    floats = integers.astype(dtype)
    intensities = np.empty(len(integers), dtype=dtype)

    # run fracshift
    inbounds = True  # if mask went out of bounds
    bounds = np.array(volume.shape)-1
    position = np.copy(position) - lower  # position in volume
    for _ in range(iters):
        # get center and shifts
        center = position.astype(np.int32)
        dz, dy, dx = position - center
        for idx in range(len(integers)):
            # get indices
            ijk = integers[idx] + center
            lmn = ijk + 1

            # make sure in bounds
            ijk[ijk < 0] = 0
            lmn[lmn < 0] = 0
            for axis in range(3):
                if ijk[axis] > bounds[axis]:
                    ijk[axis] = bounds[axis]
                if lmn[axis] > bounds[axis]:
                    lmn[axis] = bounds[axis]
                
            # expand indices
            i, j, k = ijk
            l, m, n = lmn
            
            # compute resampled intensity
            intensities[idx] = (
                volume[i, j, k] * (1-dz) * (1-dy) * (1-dx) +
                volume[i, j, n] * (1-dz) * (1-dy) * dx +
                volume[i, m, k] * (1-dz) * dy * (1-dx) +
                volume[i, m, n] * (1-dz) * dy * dx +
                volume[l, j, k] * dz * (1-dy) * (1-dx) +
                volume[l, j, n] * dz * (1-dy) * dx +
                volume[l, m, k] * dz * dy * (1-dx) +
                volume[l, m, n] * dz * dy * dx
            )
        
        # position update
        s = np.sum(intensities)
        if s != 0:
            shift = intensities @ floats / s
            position += shift

            # make sure in bounds
            for axis in range(3):
                if (position[axis] < 0) or (position[axis] > bounds[axis]):
                    inbounds = False
                    return position - shift + lower, inbounds
        else:
            break 
            
    return position + lower, inbounds


def to_shift(reference, target):
    """
    Registration function for computing the
    shift between two channels.

    Parameters
    ----------
    reference : numpy.ndarray
        Reference signal.
    target : numpy.ndarray
        Signal being shifted.

    Returns
    -------
    numpy.ndarray
        Computed shift to move moving to reference.
    """
    # handle for missing pixels
    reference, target = cp.array(reference), cp.array(target)
    for signal in [reference, target]:
        mask = signal > 0
        signal[~mask] = cp.min(signal[mask])

    # compute shift
    shift = phase_cross_correlation(reference, target, upsample_factor=100)
    cp.get_default_memory_pool().free_all_blocks()
    
    return np.clip(shift, -3, 3)  # handle for bad


def to_shifted(signal, shift):
    """ 
    Shifts volume in Fourier space.

    Parameters
    ----------
    signal : numpy.ndarray
        Signal to shift.
    shift : numpy.ndarray
        Vector of shifts along each axis.
    
    Returns
    -------
    numpy.ndarray
        Shifted volume.
    """ 
    # use Fourier transform to shift
    fourier = ndi.fourier_shift(fftn(cp.array(signal)), shift)
    signal = cp.abs(ifftn(fourier)).get()
    cp.get_default_memory_pool().free_all_blocks()
    
    return signal


def phase_cross_correlation(reference_image, moving_image, upsample_factor=1):
    """
    Compute the shift between two 3D images (adapted
    from Scikit-learn to run with CuPy).

    Parameters
    ----------
    reference_image : array
        Reference image.
    moving_image : array
        Image to register. Must be same dimensionality as
        ``reference_image``.
    upsample_factor : int, optional
        Upsampling factor. Images will be registered to within
        ``1 / upsample_factor`` of a pixel. For example
        ``upsample_factor == 20`` means the images will be registered
        within 1/20th of a pixel. Default is 1 (no upsampling).

    Returns
    -------
    numpy.ndarray
        Shift vector (in pixels) required to register ``moving_image``
        with ``reference_image``. Axis ordering is consistent with
        the axis order of the input array.
    """
    # take fourier transform of inputs
    dtype = reference_image.dtype
    reference_freq, moving_freq = fftn(reference_image), fftn(moving_image)
    product = reference_freq * moving_freq.conj()
    del reference_freq
    del moving_freq

    # get cross-correlation
    eps = cp.finfo(product.real.dtype).eps
    product /= cp.maximum(cp.abs(product), 100 * eps)
    cross_correlation = ifftn(product)
    
    # compute (pre-upsampled) shift
    maxima = cp.unravel_index(cp.argmax(cp.abs(cross_correlation)), cross_correlation.shape)
    shift = cp.stack(maxima).astype(dtype, copy=False)

    # handle shift
    shape = cp.array(product.shape)
    midpoint = (shape/2).astype(int)
    shift[shift > midpoint] -= shape[shift > midpoint]

    # continue if subpixel shift
    if upsample_factor > 1:
        # get initial shift estimate
        upsample_factor = cp.array(upsample_factor, dtype=dtype)
        shift = cp.round(shift * upsample_factor) / upsample_factor

        # make upsampling region and get center/offset
        upsample_size = cp.ceil(upsample_factor * 1.5)
        upsample_region = cp.arange(upsample_size)
        center = upsample_size // 2.0
        offset = center - shift * upsample_factor
        
        # matrix multiply DFT around shift estimate
        cross_correlation = product.conj()
        for i in reversed(range(3)):
            # compute kernel
            kernel = cp.outer(upsample_region - offset[i], fftfreq(product.shape[i], upsample_factor))
            kernel = cp.exp(-2j * cp.pi * kernel)
            kernel = kernel.astype(cp.complex64, copy=False)

            # update cross-correlation
            cross_correlation = cp.tensordot(kernel, cross_correlation, axes=(1,-1))

        # compute (upsampled) shift
        maxima = cp.unravel_index(cp.argmax(cp.abs(cross_correlation)), cross_correlation.shape)
        up_shift = cp.stack(maxima).astype(dtype, copy=False)
        shift += (up_shift - center) / upsample_factor

    return shift.get()
