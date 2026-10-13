import os
import math
import cupy as cp
import numpy as np
import cupyx.scipy.ndimage as ndi

from numba import jit
from .utilities import proc, read_h5, mkdir, to_log, \
     to_fmt, fracshift


@proc
def detection_proc(nucleus, shape, directory, c, device):
    """
    Process job for detecting features within
    a nucleus.

    Parameters
    ----------
    nucleus : int
        Index for the tracked nucleus.
    shape : numpy.ndarray
        Shape of PSF masks for each channel.
    directory : str
        Path to the directory for input/outputs.
    c : dict
        Dictionary for config.
    device : int, optional
        GPU device to use.
    """
    # set CSV format
    fmt = 2*["%d"] + 6*["%.3f"] + ["%d"]
    header = "frame,label,z_map (px),y_map (px),x_map (px)," + \
             "z_sub (px),y_sub (px),x_sub (px),sisters"

    # make paths
    frames_path = os.path.join(directory, "frames")
    nuclei_path = os.path.join(directory, "nuclei")

    # get path to nucleus and read track
    nucleus_path = os.path.join(nuclei_path, to_fmt(nucleus))
    nucleus_track = np.loadtxt(os.path.join(nucleus_path, "info.csv"),
                               delimiter=",", skiprows=1, ndmin=2)
    # go through channels
    chnls_info = {}  # hold channel feature info
    chnls_path = mkdir(os.path.join(nucleus_path, "channels"))
    for chnl in c["detection"]["channels"]:
        chnls_info[chnl] = np.empty((0,9))
        mkdir(os.path.join(chnls_path, to_fmt(chnl)))

    # iterate through track
    for i in range(len(nucleus_track)):
        # get frame/label and path
        frame, label = int(nucleus_track[i,0]), int(nucleus_track[i,1])
        frame_path = os.path.join(frames_path, to_fmt(frame))
        
        # read mask/deskew/deconvolve for nucleus
        lbl = to_fmt(label)
        mask = read_h5(os.path.join(frame_path, "masks.h5"), lbl)
        dsk = read_h5(os.path.join(frame_path, "deskews.h5"), lbl)
        dcv = read_h5(os.path.join(frame_path, "deconvolves.h5"), lbl)
                
        # detect features for each channel
        with cp.cuda.Device(device):
            features = detection(dcv, dsk, mask, shape, **c["detection"])
    
        # iterate through channels
        for chnl in c["detection"]["channels"]:
            # add frame and nucleus label (for tracking)
            info = np.empty((len(features[chnl]),9))
            info[:,0], info[:,1] = frame, label
            info[:,2:] = features[chnl]

            # append to info
            chnls_info[chnl] = np.concatenate((chnls_info[chnl], info), axis=0)
        
    # iterate through channels and save as CSV
    for chnl in c["detection"]["channels"]:
        info = chnls_info[chnl]
        np.savetxt(os.path.join(chnls_path, to_fmt(chnl), "info.csv"), info,
                   delimiter=",", fmt=fmt, header=header, comments="")


def detection(dcv, dsk, mask, shape, channels={}, nearby=10.0, sister=5.0):
    """
    Defines detection method and filters any
    detected features.

    Parameters
    ----------
    dcv : numpy.ndarray
        Deconvolved signal subset around nucleus.
    dsk : numpy.ndarray
        Deskewed signal subset around nucleus.
    mask : numpy.ndarray
        Mask subset for nucleus.
    shape : numpy.ndarray
        Shape of PSF masks for each channel.
    channels : dict, optional
        Channels (with parameters) to detect features.
    nearby : float, optional
        Minimum separation between nearby features.
    sister : float, optional
        Maximum distance to designate sisters.

    Returns
    -------
    dict
        A dictionary with the features detected
        in each channel.
    """
    # get centroid
    centroid = np.mean(np.argwhere(mask).astype(float), axis=0)
        
    # find features in each channel
    features = dict.fromkeys(channels.keys())
    for chnl in channels:
        # handle channel config
        if "nearby" not in channels[chnl]:
            channels[chnl]["nearby"] = nearby
        if "sister" not in channels[chnl]:
            channels[chnl]["sister"] = sister
            
        # run SIFT-like detection and handle
        positions, sisters = detect(dcv[chnl], dsk[chnl], mask, 
                                    shape[chnl], **channels[chnl])
        if len(positions.shape) == 2:
            mapped = positions - centroid  # center positions
            features[chnl] = np.concatenate((mapped, positions, sisters.reshape(-1,1)), axis=1)
        else:
            features[chnl] = np.empty((0,7))
    
    return features


def detect(dcv, dsk, mask, shape, radius={}, response=300, nearby=10.0, sister=3.0):
    """
    SIFT-like method to finding features/dots
    of within nuclei.
    
    Parameters
    ----------
    dcv : numpy.ndarray
        Deconvolved (channel) signal subset around nucleus.
    dsk : numpy.ndarray
        Deskewed (channel) signal subset around nucleus.
    mask : numpy.ndarray
        Mask subset for nucleus.
    shape : numpy.ndarray
        Shape of PSF mask for channel.
    radius : dict, optional
        A dictionary containing a window of radii.
    response : float, optional
        Threshold for LoG filter response.
    nearby : float, optional
        Minimum separation between nearby features.
    sister : float, optional
        Maximum distance to designate sisters.

    Returns
    -------
    numpy.ndarray
        Positions of the detected features.
    numpy.ndarray
        Mask for sister detection of features.
    """
    # set up range of sigmas
    n = 10
    lower = radius["lower"] if "lower" in radius else 1.0
    upper = radius["upper"] if "upper" in radius else 3.0
    k = (upper/lower)**(1/(n-1))
    sigmas = lower/math.sqrt(2) * cp.array([(k**i) for i in range(0,n)])
    
    # move to GPU and estimate background
    dcv, mask = cp.array(dcv), cp.array(mask)
    background = cp.median(dcv[mask])
    # dcv[dcv == 0] = background  # testing
    
    # apply normalized LoG filters
    logs = cp.zeros((n, *dcv.shape), dtype=cp.float32)
    for i in range(n):
        logs[i] = -sigmas[i]**2 * to_log(dcv, sigmas[i])

    # find local maxima
    maxed = ndi.maximum_filter(logs, footprint=cp.ones((3,3,3,3)), mode="nearest")
    dots = (logs == maxed) * mask

    # get positions/responses and filter
    responses, positions = logs[dots], cp.argwhere(dots).astype(cp.float32)[:,1:]
    keep = responses >= (response * background / 1000)  # scale with background
    responses, positions = responses[keep].get(), positions[keep].get()

    # get best detections and pad deskew if nuclear localized
    ordering = np.argsort(responses)
    positions, responses = positions[ordering][-5:], responses[ordering][-5:]

    # get mask and mark nearby before refining positions
    mask = mask.get()
    sisters, keep = mark(positions, responses, nearby, sister)
    for i in range(len(positions)):   
        # refine position (mark if went out of volume)
        positions[i], inbounds = fracshift(dsk, positions[i], subtract=True,
                                           mask=mask, shape=shape, radius=7)
        keep[i] *= inbounds

    # filter detections
    positions, sisters = positions[keep], sisters[keep]
    cp.get_default_memory_pool().free_all_blocks()
    
    return positions, sisters


@jit(nopython=True)
def mark(positions, responses, nearby=10.0, sister=5.0):
    """
    Marks features to handle for nearby and sister
    detections.
    
    Parameters
    ----------
    positions : numpy.ndarray
        Positions of features.
    responses : dict, optional
        LoG responses for initially detected.
    nearby : float, optional
        Minimum separation between nearby features.
    sister : float, optional
        Maximum distance to designate sisters.

    Returns
    -------
    numpy.ndarray
        Mask of features with detected sisters.
    numpy.ndarray
        Mask of indices to keep.
    """
    # number of detected features
    n = len(positions)
    keep = np.ones(n, dtype=np.bool)
    sisters = np.zeros(n, dtype=np.bool)

    # iterate through neighboring features
    for i in range(n):
        for j in range(i+1, n):
            # keep the stronger response
            distance = np.sum((positions[i] - positions[j])**2)**0.5
            if distance <= nearby:
                if responses[i] > responses[j]:
                    keep[j] = 0
                else:
                    keep[i] = 0

                # mark sisters
                if distance <= sister:
                    sisters[i], sisters[j] = 1, 1

    return sisters, keep
