import os
import h5py
import cupy as cp
import numpy as np
import cupyx.scipy.ndimage as ndi

from .utilities import proc, read_czi, write_tiff, to_isotropic, to_fmt, \
     to_gpu, mkdir, to_log, to_ellipsoid, deskewing, deconvolution


@proc
def segmentation_proc(frame, file, directory, c, kernel, device, locks):
    """
    Process job for segmenting nuclei in a single frame.

    Parameters
    ----------
    frame : int
        Index of the frame.
    file : str
        Path to CZI file.
    directory : str
        Path to the output directory.
    c : dict
        Dictionary for config.
    kernel : numpy.ndarray
        Array describing PSF for each channel.
    device : int
        GPU device to use.
    locks : list
        List of GPU locks.
    """
    # set CSV format
    fmt = 2*["%d"] + 3*["%.3f"] + 7*["%d"]
    header = "frame,label,z_cen (px),y_cen (px),x_cen (px),z_low (px),y_low (px)," + \
             "x_low (px),z_upp (px),y_upp (px),x_upp (px),volume (px)"
    
    # make frame directory
    frame_path = mkdir(os.path.join(directory, "frames", to_fmt(frame)))
    
    # read raw and get lock
    raw = read_czi(file, frame)

    locks[device].acquire()
    with cp.cuda.Device(device):
        # get deskew/deconvolve (and save)
        resampled = to_isotropic(raw, **c["isotropic"])[:,:-5] #Added on July 7, 2026 for isotropic volumes #shave 
        deconvolve = deconvolution(resampled, kernel, **c["deconvolution"])

        # segment on deconvolve
        labels, info = segmentation(deconvolve, **c["segmentation"])
    locks[device].release()

    # (optionally) save as TIFFs
    write_tiff(os.path.join(frame_path, "deconvolve.tiff"), deconvolve.astype(np.uint16))
    # write_tiff(os.path.join(frame_path, "resampled.tiff"), resampled.astype(np.uint16))

    # make writers
    masks_file = h5py.File(os.path.join(frame_path, "masks.h5"), "w")
    rsps_file = h5py.File(os.path.join(frame_path, "resampled.h5"), "w")
    dcvs_file = h5py.File(os.path.join(frame_path, "deconvolves.h5"), "w")

    # iterate over nuclei labels
    for label in range(len(info)):
        # get lower and upper slices
        l, u = info[label,4:7].astype(int), info[label,7:10].astype(int)
        mask = labels[l[0]:u[0],l[1]:u[1],l[2]:u[2]] == label
        rsp = resampled[:,l[0]:u[0],l[1]:u[1],l[2]:u[2]]
        dcv = deconvolve[:,l[0]:u[0],l[1]:u[1],l[2]:u[2]]

        # write each subset as key
        lbl = to_fmt(label)
        masks_file.create_dataset(lbl, data=mask, compression="gzip")
        rsps_file.create_dataset(lbl, data=rsp, compression="gzip")
        dcvs_file.create_dataset(lbl, data=dcv, compression="gzip")

    # close H5 writers
    for file in [masks_file, rsps_file, dcvs_file]:
        file.close()
    
    # add frame for tracking and save as CSV
    info = np.concatenate((frame*np.ones((len(info),1)), info), axis=1)
    np.savetxt(os.path.join(frame_path, "info.csv"), info, delimiter=",",
               fmt=fmt, header=header, comments="")


def segmentation(deconvolve, channel=0, intensity={}, kernel=3, sigma=5.0,
                 erode=15, dilate=15, scale=0.333, volume={}, pad=5):
    """
    Segmentation method and handling for filtering nuclei.

    Parameters
    ----------
    deconvolve : numpy.ndarray
        Deconvolved frame to segment.
    channel : int, optional
        Channel to base the nuclear segmentation on.
    intensity : dict, optional
        Dictionary of lower/upper intensity thresholds.
    kernel : int, optional
        Window size for median filter.
    sigma : float, optional
        Sigma for Gaussian filter.
    erode : int, optional
        Radius for erosion structuring element.
    dilate : int, optional
        Radius for dilation structuring element.
    scale : float, optional
        Factor to reduce structuring element radius along first axis.
    volume : dict, optional
        Dictionary of lower/upper volume thresholds.
    pad : int, optional
        Padding of nucleus signal subsets.
    
    Returns
    -------
    numpy.ndarray
        Integer labels of nuclei (0 to n-1) and background (-1).
    numpy.ndarray
        Info regarding size, bounds, etc. for each nucleus.
    """
    # pad and send nuclear signal to GPU
    nuclear = deconvolve[channel]
    nuclear = to_gpu(np.pad(nuclear, pad, mode="constant"))
    region, shape = nuclear > 0, np.asarray(nuclear.shape)
    
    # apply bounds to intensity
    if "lower" in intensity:
        nuclear[nuclear < intensity["lower"]] = intensity["lower"]
    if "upper" in intensity:
        nuclear[nuclear > intensity["upper"]] = intensity["upper"]

    # apply median and LoG filters
    nuclear = ndi.median_filter(nuclear, size=3*(kernel,))
    nuclear = to_log(nuclear, sigma)
    
    # get zero-crossings
    positive, negative = nuclear > 0, nuclear < 0
    del nuclear
    crossing = cp.ones((3,3,3), dtype=bool)
    crossing[1,1,1] = 0
    neg_pos = ndi.convolve(positive, crossing, output=bool) * negative
    pos_neg = ndi.convolve(negative, crossing, output=bool) * positive
    del positive
    del negative
    mask = cp.logical_or(neg_pos, pos_neg)
    del neg_pos
    del pos_neg

    # fill mask shells
    for _ in range(3):
        # done in 2D to handle for odd shapes
        for i in range(mask.shape[0]):
            mask[i] = ndi.binary_fill_holes(mask[i])
        mask = cp.rollaxis(mask, 2)

    # enforce non-zero region and remove padding
    mask *= region
    del region
    mask = mask[pad:-pad,pad:-pad,pad:-pad]

    # make structuring elements for watershed-like labeling
    s = np.diag((scale**2, 1, 1))
    erosion = to_ellipsoid(erode**2 * s)
    dilation = to_ellipsoid(dilate**2 * s)

    # label after eroding
    mask = ndi.binary_erosion(mask, structure=cp.array(erosion))
    labels, n = ndi.label(mask, cp.ones((3,3,3), dtype=bool), cp.int16)

    # compute volumes and find where labels are
    volumes = ndi.sum_labels(mask.astype(cp.int32), labels, cp.arange(1,n+1))
    indices = ndi.value_indices(labels, ignore_value=0)

    # filter out small (makes the dilation fast)
    remaining = list(range(1,n+1))
    for label in range(1,n+1):
        if volumes[label-1] < 1000:
            labels[indices[label]] = 0
            remaining.remove(label)
    
    # grey dilation to recover nuclei
    labels = ndi.grey_dilation(labels, footprint=cp.array(dilation))
    mask = (labels > 0).astype(cp.int32)  # re-make mask

    # compute volumes/centroids
    volumes = ndi.sum_labels(mask, labels, cp.arange(1,n+1))
    centroids = ndi.center_of_mass(mask, labels, cp.arange(1,n+1))

    # find where objects are
    indices = ndi.value_indices(labels, ignore_value=0)
    slices = ndi.find_objects(labels)
    labels -= 1  # shift labels down to match indices
    del mask

    # filter out nuclei
    info, idx = [], 0
    
    # Get the actual unpadded 3D shape of the label volume
    # shape was originally cp.asarray(nuclear.shape) which included the padding
    # Let's get the true unpadded dimensions:
    volume_shape = deconvolve[channel].shape 

    for label in remaining:
        v = volumes[label-1].get()
        if (("lower" not in volume) or (v >= volume["lower"])) and \
           (("upper" not in volume) or (v <= volume["upper"])):

            # find lower/upper bounds of the slice
            l = np.array([slices[label-1][axis].start for axis in range(3)])
            u = np.array([slices[label-1][axis].stop for axis in range(3)])
            # Get centroid coordinates for the nucleus
            c = centroids[label-1].get()

            # Check if the bounding box touches the true edges (0 or true_shape)
            touches_boundary = (
                l[0] == 0 or u[0] == volume_shape[0] or
                l[1] == 0 or u[1] == volume_shape[1] or
                l[2] == 0 or u[2] == volume_shape[2]
            )
            
            if not touches_boundary:
                l = np.clip(l - pad, 0, shape)
                u = np.clip(u + pad, 0, shape)
                   
                # add nucleus info and update labels, then update idx
                info.append([idx, *c, *l, *u, v])
                #info.append([idx, c[0], c[1], c[2], l[0], l[1], l[2], u[0], u[1], u[2], v])
                #print(f"Label {label}: Centroid at {c}, Lower bounds {l}, Upper bounds {u}, Volume {v}, Index {idx}")
                labels[indices[label]] = idx
                idx += 1

            else:
                labels[indices[label]] = -1
        else:
            labels[indices[label]] = -1

   
    # for label in remaining:
    #     # add if correct volume
    #     v = volumes[label-1].get()
    #     if (("lower" not in volume) or (v >= volume["lower"])) and \
    #        (("upper" not in volume) or (v <= volume["upper"])):

    #         # find lower/upper of subset with padding
    #         l = np.array([slices[label-1][axis].start for axis in range(3)])
    #         u = np.array([slices[label-1][axis].stop for axis in range(3)])
    #         l = np.clip(l - pad, 0, shape)
    #         u = np.clip(u + pad, 0, shape)
            
    #         # only keep if nucleus does not touch the edge of the volume
    #         if (l[0] != 0) and (u[0] != deconvolve.shape[0] + pad) and \
    #            (l[1] != 0) and (u[1] != deconvolve.shape[1] + pad) and \
    #            (l[2] != 0) and (u[2] != deconvolve.shape[2] + pad):
                
    #             # re-label and pull centroid
    #             labels[indices[label]] = idx
    #             c = centroids[label-1].get()
        
    labels = labels.get()
    del slices
    del volume
    del centroids
    del indices
    cp.get_default_memory_pool().free_all_blocks()

    return labels, np.asarray(info)
