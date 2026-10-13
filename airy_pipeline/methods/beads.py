import os
import math
import warnings
import cupy as cp
import numpy as np
import cupyx.scipy.ndimage as ndi

from numba import jit
from itertools import combinations
from scipy.optimize import minimize
from .utilities import proc, read_czi, to_isotropic, read_h5, write_h5, to_gpu, \
     mkdir, to_log, fracshift, to_date, ridge, mappings, save_config, \
     to_directory, to_end, to_fmt, to_volume, to_shifted, deskewing

# handles for no bead pairs
warnings.filterwarnings("ignore", message="loadtxt: input contained no data")


@proc
def finding_proc(file, config, c, device, locks):
    """
    Process job detecting beads in a single CZI.

    Parameters
    ----------
    file : str
        Path to bead file.
    config : str
        Path to YAML file.
    c : dict
        Dictionary for config.
    device : int
        GPU device to use.
    locks : list
        List of GPU locks.
    """
    # make directory and save config
    directory = mkdir(to_directory(file))
    save_config(config, directory)
    
    # set CSV format
    fmt = ["%d"] + 6*["%.3f"]
    header = "frame,z (px),y (px),x (px),z (px),y (px),x (px)"

    # make holders for kernel and number of samples
    n, k = max(c["beads"]["finding"]["channels"]) + 1, 25  # kernel size of 15 pixels
    size, radius = 3*(k,), (k - 1) / 2
    kernel, samples = np.zeros((n,*size)), np.zeros(n)
    
    # process frames
    frames_path = os.path.join(directory, "frames")
    for frame in range(to_end(file)):
        # read raw and get lock
        raw = read_czi(file, frame)
        resampled = to_isotropic(raw, **c["isotropic"]) #Added on July 7, 2026 for isotropic volumes
        locks[device].acquire()
        with cp.cuda.Device(device):
            # find beads ## July 7, 2026 got rid of the deskewing step
            positions, pairs = finding(resampled, **c["beads"]["finding"])
        locks[device].release()
            
        # make frame path and get channel pairs
        frame_path = mkdir(os.path.join(frames_path, to_fmt(frame)))
        for pair in pairs:
            # add frame
            info = np.concatenate((frame*np.ones((len(pairs[pair]),1)),
                                   pairs[pair]), axis=1)
            # save as CSV
            name = to_fmt(pair[0]) + "-" + to_fmt(pair[1]) + ".csv"
            np.savetxt(os.path.join(frame_path, name), info,
                       delimiter=",", fmt=fmt, header=header, comments="")
        
        # average beads for each channel
        for chnl in c["beads"]["finding"]["channels"]:
            for i in range(len(positions[chnl])):
                # get bead volume
                position = positions[chnl][i]
                volume, lower, _ = to_volume(resampled[chnl], position, radius)
            
                # make sure full shape then add shifted volume
                if volume.shape == size:
                    shift = (position - lower) - radius
                    kernel[chnl] += to_shifted(volume, -shift)
                    samples[chnl] += 1

    # make path then save bead PSF (kernel) for channels
    channels_path = os.path.join(directory, "channels")
    for chnl in c["beads"]["finding"]["channels"]:
        # make channel path
        chnl_path = mkdir(os.path.join(channels_path, to_fmt(chnl)))

        # write (not normalized) kernel/samples as H5 and CSV
        write_h5(os.path.join(chnl_path, "kernel.h5"), kernel[chnl])
        np.savetxt(os.path.join(chnl_path, "samples.csv"), [samples[chnl]],
                   delimiter=",", comments="")
        
    return kernel, samples


def finding(resampled, channels={}, separation=10.0, search=5.0):
    """
    Finds beads between channel pairs.

    Parameters
    ----------
    resampled : numpy.ndarray
        Resampled signal.
    channels : dict, optional
        A dictionary of bead channel parameters.
    separation : float, optional
        Minimum separation between nearby beads.
    search : float, optional
        Maximum offset between channels.
    
    Returns
    -------
    numpy.ndarray
        Estimated positions of paired beads.
    """
    # find beads in each channel
    positions = {}
    for chnl in channels.keys():
        positions[chnl] = find(resampled[chnl], separation=separation,
                               **channels[chnl])
    # connect channels
    pairs = {}
    for pair in combinations(channels.keys(), 2):
        pairs[pair] = connect(positions[pair[0]], positions[pair[1]],
                              search=search)
    return positions, pairs


def find(resampled, separation=10.0, radius=2.0, response=500):
    """
    Finds beads in a channel.

    Parameters
    ----------
    resampled : numpy.ndarray
        Resampled signal.
    radius : float, optional
        Approximate radius (in pixels) of PSF.
    response : float, optional
        Minimum LoG response for bead.
    separation : float, optional
        Minimum separation between nearby beads.
    
    Returns
    -------
    numpy.ndarray
        Estimated positions of beads.
    """
    # compute normalized Laplacian of Gaussian (LoG) filter
    sigma = radius / math.sqrt(2)
    logs = -sigma**2 * to_log(to_gpu(resampled), sigma)

    # find local maximums of LoG response
    maxed = ndi.maximum_filter(logs, footprint=cp.ones((3,3,3)), mode="nearest")
    dots = (logs == maxed) * (response["lower"] < logs) * (logs < response["upper"])
    del logs
    positions = cp.argwhere(dots).astype(cp.float32).get()
    del dots
    cp.get_default_memory_pool().free_all_blocks()
    
    # mark nearby then get subpixel positions
    keep = mark(positions, separation)
    for i in range(len(positions)):
        # refine position (mark if went out of volume)
        positions[i], inbounds = fracshift(resampled, positions[i],
                                           subtract=False, radius=12)
        keep[i] *= inbounds

    return positions[keep]


@jit(nopython=True)
def mark(positions, separation=10.0):
    """
    Removes beads that are too close to help
    connect beads between channels and compute
    the PSF(s).
    
    Parameters
    ----------
    positions : numpy.ndarray
        Positions of features.
    separation : float, optional
        Minimum separation between nearby beads.

    Returns
    -------
    numpy.ndarray
        Boolean mask of indices to keep.
    """
    # get number of detected beads
    n = len(positions)
    keep = np.ones(n, dtype=np.bool)  # mask to keep

    # iterate though all pairs
    for i in range(n):
        for j in range(i+1,n):            
            # remove both if close
            if np.sum((positions[i] - positions[j])**2)**0.5 < separation:
                keep[i] = 0
                keep[j] = 0

    return keep


@jit(nopython=True)
def connect(x, y, search=5.0):
    """
    Connects bead detections between two channels.

    Parameters
    ----------
    x : numpy.ndarray
        Detection positions for one channel.
    y : numpy.ndarray
        Detection positions for another channel.
    search : float
        Maximum offset between channels.
    
    Returns
    -------
    numpy.ndarray
        Positions of paired detections.
    """
    # make holder for maximum pairs
    n, m = len(x), len(y)
    pairs = np.empty((max(n,m), 6))

    # find nearest detection within radius
    idx = 0
    for i in range(len(x)):
        nearest, nearest_distance = -1, np.inf
        for j in range(len(y)):
            # handle based on distance
            distance = np.sum((x[i] - y[j])**2)**0.5
            if distance < search:
                if (nearest == -1) or (distance < nearest_distance):
                    nearest, nearest_distance = j, distance
        
        # add to pairs      
        if nearest != -1:
            pairs[idx,0:3], pairs[idx,3:6] = x[i], y[nearest]
            idx += 1
            
    return pairs[:idx]  # remove unused


def to_shape(kernel, angle=0): #July 8 2026 changed default angle to 0 for Airyscan
    """ 
    Fits a PSF volume with an anisotropic
    Gaussian (ellipsoidal mask)
    
    Parameters
    ----------
    kernel : numpy.ndarray
        Volume of intensities to fit PSF.
    angle : float, optional
        Angle (degrees) of the sheet.
    
    Returns
    -------
    numpy.ndarray
        Mask shape as "covariance" matrix.
    """
    # get positions and intensities
    positions = np.argwhere(np.ones(kernel.shape))
    intensities = kernel.flatten()

    # make rotation matrix and map positions
    theta = angle * math.pi/180
    s, c = math.sin(theta), math.cos(theta)
    rotation = np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])
    radius = (np.asarray(kernel.shape) - 1) / 2
    mapped = (positions - radius) @ rotation

    # loss function
    def loss(u):
        # sum of squared error
        inverse = np.diag(1/np.exp(u[1:]))
        estimate = np.exp(u[0]) * np.exp(-0.5*np.sum((mapped @ inverse) * mapped, axis=1))
        return np.sum((estimate - intensities)**2)

    # fit Gaussian
    u0 = np.log(np.array([np.max(intensities), 4.0, 4.0, 4.0]))
    u = minimize(loss, u0, method="BFGS").x
    u = np.exp(u)
    
    # make shape
    shape = rotation @ np.diag(u[1:]) @ rotation.T
    return shape


def fitting_proc(file, directory, beads, c):
    """
    Process job for fitting beads.

    Parameters
    ----------
    file : str
        Path to CZI file.
    directory : str
        Path for pipeline outputs.
    beads : str
        Path to TXT file of bead files.
    c : dict
        Dictionary for config.
    """
    # make path and run fitting
    channels = list(c["beads"]["finding"]["channels"].keys())
    fitting(channels, file, directory, beads, **c["beads"]["fitting"])


def fitting(channels, file, directory, beads, days=2, alpha=1e-2):
    """
    Fitting method for bead aberrations. 

    Parameters
    ----------
    channels : list
        Detected bead channels.
    file : str
        Path to CZI file.
    directory : str
        Path for pipeline outputs.
    beads : str
        Path to TXT file of bead files.
    days : int, optional
        Number of days before or after CZI.
    alpha : float
        Ridge parameter.
    """
    # get window of bead paths and make path
    window = to_window(file, beads, days=days)
    intensities_path = mkdir(os.path.join(directory, "corrections", "intensities"))
    aberrations_path = mkdir(os.path.join(directory, "corrections", "aberrations"))

    # compute bead PSF (kernel) for channels
    for chnl in channels:
        kernel, samples = 0, 0
        for path in window:
            # get path to channel
            chnl_path = os.path.join(to_directory(path), "channels", to_fmt(chnl))

            # update (not normalized) kernel/samples
            kernel += read_h5(os.path.join(chnl_path, "kernel.h5"))
            samples += np.loadtxt(os.path.join(chnl_path, "samples.csv"),
                                  delimiter=",")
        # save max as CSV
        if samples > 0:
            bead = (np.max(kernel) - np.min(kernel)) / samples
            np.savetxt(os.path.join(intensities_path, f"{to_fmt(chnl)}.csv"), [bead],
                       delimiter=",", comments="")

    # iterate through pairs of channels
    for pair in combinations(channels, 2):
        # read all positions
        positions = np.empty((0,6))
        for path in window:
            # go through frames
            frames_path = os.path.join(to_directory(path), "frames")
            for item in os.listdir(frames_path):
                if item.isnumeric():
                    # read paired positions and add
                    name = to_fmt(pair[0]) + "-" + to_fmt(pair[1]) + ".csv"
                    info = np.loadtxt(os.path.join(frames_path, item, name),
                                      delimiter=",", skiprows=1, ndmin=2)
                    if len(info) > 0:
                        positions = np.concatenate((positions, info[:,1:]), axis=0)

        if len(positions) > 0:
            # compute shifts (and medians) between channels
            shifts = positions[:,3:6] - positions[:,0:3]
            median = np.median(shifts, axis=0)
    
            # filter bad shifts
            s = shifts - median
            cutoff = 2*np.median(np.abs(s), axis=0)
            mask = np.all(s < cutoff, axis=1) * np.all(s > -cutoff, axis=1)
            shifts, positions = shifts[mask], positions[mask]
            
            # for both mappings
            if len(positions) > 0:
                for i in range(2):            
                    # fit the shifts to the positions 
                    mapped = mappings(positions[:,3*(1-i):3*(1-i)+3])
                    weights = ridge(mapped, (-1)**i*shifts, alpha)
            
                    # save as CSV
                    name = to_fmt(pair[1-i]) + "-" + to_fmt(pair[i]) + ".csv"
                    np.savetxt(os.path.join(aberrations_path, name), weights,
                            delimiter=",", comments="")

    
def to_window(file, beads, days=5):
    """
    Finds nearby bead files within some
    amount of days.

    Parameters
    ----------
    file : str
        Path to CZI file.
    beads : str
        Path to TXT file of bead files.
    days : int, optional
        Number of days before or after date.
    
    Returns
    -------
    list
        List of nearby bead paths.
    """
    # get the date of target and hold
    date = to_date(file)
    closest = [None, None]  # block, days
    
    # read paths into blocks
    blocks, block = [], []
    with open(beads, "r") as f:
        # separate blocks based on breaks
        for path in f:
            path = path.strip()
            if path == "":
                if block:
                    blocks.append(block)
                    block = []
            else:
                # get the days away from target
                distance = abs(date - to_date(path)).days
                block.append((path, distance))

                # if in window, try to find closest block
                if distance <= days:
                    if (closest[0] is None) or (distance < closest[1]):
                        closest[0], closest[1] = len(blocks), distance
        if block:
            blocks.append(block)

    # make window of paths
    window = []
    if closest[0] is not None:
        block = blocks[closest[0]]
        for path, distance in block:
            if distance <= days:
                window.append(path)

    return window
