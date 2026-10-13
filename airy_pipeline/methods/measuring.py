import os
import warnings
import cupy as cp
import numpy as np
import h5py
from scipy import ndimage
#from skimage.segmentation import watershed
#from skimage.feature import peak_local_max

from .utilities import proc, read_h5, mkdir, to_fmt

warnings.filterwarnings("ignore", message="loadtxt: input contained no data")

@proc
def measuring_proc(nucleus, directory, c, device):
    """
    Process job for measuring features in nuclei.

    Parameters
    ----------
    nucleus : int
        Index for the tracked nucleus.
    directory : str
        Path to the directory for input/outputs.
    c : dict
        Dictionary for config.
    device : int
        GPU device to use.
    """
    # set CSV format
    fmt = ["%d"] + 3*["%.3f"] + ["%d"] + ["%.3f"] + ["%d"] + ["%.3f"] + ["%d"]
    header = "body_index, z_cen (px), y_cen (px), x_cen (px), body_volume (px), mean_intensity (au), nuc_volume (px), nuc_mean_intensity (au), nuc_index"

    # make holder
    info = []
    
    # get paths
    frames_path = os.path.join(directory, "frames")
    nuclei_path = os.path.join(directory, "nuclei")
    bodies_path = mkdir(os.path.join(nuclei_path, to_fmt(nucleus), "bodies"))

    # get path to nucleus and read track
    nucleus_path = os.path.join(nuclei_path, to_fmt(nucleus))
    nucleus_track = np.loadtxt(os.path.join(nucleus_path, "info.csv"),
                               delimiter=",", skiprows=1, ndmin=2)
    # get frame/label
    frame, label = int(nucleus_track[0,0]), int(nucleus_track[0,1])
    lbl = to_fmt(label)

    # read mask/resample/deconvolve for nucleus
    frame_path = os.path.join(frames_path, to_fmt(frame))
    mask = read_h5(os.path.join(frame_path, "masks.h5"), lbl)
    rsp = read_h5(os.path.join(frame_path, "resampled.h5"), lbl)
    dcv = read_h5(os.path.join(frame_path, "deconvolves.h5"), lbl)

    # detect polycomb bodies
    with cp.cuda.Device(device):
        info, body_labels = particles_segment(dcv, rsp, mask, **c["measuring"]["bodies"]) ## Make sure this matches
    
    info_nucindex = np.column_stack((info, np.repeat(nucleus, info.shape[0]))) #add nucleus number to the end of each column

    # save as CSV
    np.savetxt(os.path.join(bodies_path, "info.csv"), info_nucindex,
               fmt=fmt, header=header, delimiter=",", comments="")

    # make writers
    body_labels_file = h5py.File(os.path.join(bodies_path, "body_labels.h5"), "w")
    body_labels_file.create_dataset("data", data=body_labels, compression="gzip")
    body_labels_file.close()



# Watershed segmentation to segment particles. 
def particles_segment(dcv, rsp, mask, channel = 0, sigma=5.0, response=3.0, pct=99.0):
    """
    Defines segmentation method for particles (i.e. nuclear bodies).

    Parameters
    ----------
    dcv : numpy.ndarray
        Deconvolved signal subset around nucleus.
    rsp : numpy.ndarray
        Resampled signal subset around nucleus.
    mask : numpy.ndarray
        Mask subset for nucleus.
    channel: integer
        Channel to process bodies in.
    sigma : float,
        sigma for gaussian blurring (default: 5.0).
    response : float,
        response for standard deviation based thresholding (default: 3.0).
    pct : float, 
        percentile for thresholding (default: 99.0).

    Returns
    -------
    numpy.ndarray
        Positions of the detected pacticles.
    numpy.ndarray
        Mask for detected particles.
    """
    #print(dcv.shape, rsp.shape, mask.shape)
    
    # Background subtraction
    # Define background as gaussian blurred resampled image. Subtract this from deconvolved image.
    background = ndimage.gaussian_filter(rsp[channel].astype(np.float32), sigma=sigma, mode='nearest')
    processed_img = np.maximum(dcv[channel] - background, 0) * mask

    # Determine threshold
    mask_threshold = np.nanpercentile(dcv[channel] * mask, pct)
    mask_median = np.median(dcv[channel] * mask)
    mask_std = np.std(dcv[channel] * mask)

    final_threshold = np.mean([mask_threshold, mask_median + response * mask_std])
    #final_threshold = (mask_median + threshold_weight * mask_threshold)/(threshold_weight + 1) # Weighted average of 99th percentile and median of nuc
    particle_mask = processed_img > final_threshold

    struct = ndimage.generate_binary_structure(3, 1)  # 6-connectivity -- seems to work fine in testing a few images
    labels, n = ndimage.label(particle_mask, structure=struct)
    nuc_volume = np.sum(mask)
    nuc_mean_intensity = np.sum(rsp[channel][mask])/nuc_volume
    info = []

    info = []
    for index in range(1, n+1):
        body_mask = labels == index
        body_volume = np.sum(body_mask)
        centroid = np.mean(np.argwhere(body_mask), axis=0)
        mean_intensity = np.sum(rsp[channel][body_mask]) / body_volume

        data = [index-1, *centroid, body_volume, mean_intensity, nuc_volume, nuc_mean_intensity]
        info.append(data)

    if len(info) == 0:
        # no bodies detected — return an empty (0, 8) array so downstream
        # column_stack / savetxt still see the right number of columns
        info = np.zeros((0, 8))
    else:
        info = np.asarray(info)
    labels -= 1

    return info, labels
