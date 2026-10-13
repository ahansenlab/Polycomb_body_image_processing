import os
import cv2
import numpy as np

from PIL import Image, ImageDraw, ImageFont
from .utilities import proc, read_tiff, read_h5, \
     to_mip, to_time, to_fmt, mkdir


@proc
def labels_proc(frame, file, directory, c, lookup):
    """
    Process job for MIP-ing a frame with 
    nuclei labels.

    Parameters
    ----------
    frame : int
        Index of the movie frames.
    file : str
        Path to CZI file.
    directory : str
        Path to the directory for input/outputs.
    c : dict
        Dictionary for config.
    lookup : dict
        Dictionary of nuclei tracks.
    """
    # get frame path and read info
    frame_path = os.path.join(directory, "frames", to_fmt(frame))
    info = np.loadtxt(os.path.join(frame_path, "info.csv"),
                      delimiter=",", skiprows=1, ndmin=2)
    # get deconvolve
    deconvolve = read_tiff(os.path.join(frame_path, "deconvolve.tiff"))

    # make MIP
    chnl = c["segmentation"]["channel"]
    mip = to_mip(deconvolve[chnl], gray=True, axis=0, **c["movies"]["channels"][chnl])
    mip = np.clip(mip, 0, 200)  # clip for labels
    
    # add color labels to MIP
    text = []
    available = np.ones(mip.shape[0:2], dtype=bool)  # remaining unlabeled region
    masks = np.zeros(mip.shape[0:2], dtype=bool)
    for label in range(len(info)):
        key = (frame, label)
        if key in lookup:
            # read mask subset
            l, u = info[label,5:8].astype(int), info[label,8:11].astype(int)
            mask = read_h5(os.path.join(frame_path, "masks.h5"), to_fmt(label))

            # add color to nucleus (on remaining region)
            masks[l[1]:u[1],l[2]:u[2]] = np.max(mask, axis=0)
            mip[np.logical_and(masks, available)] += (55*lookup[key][1]).astype(np.uint8)
            available[masks] = 0
            masks *= False

            # append nucleus label
            x, y = int(info[label,3]), int(info[label,4])
            text.append([x, y, lookup[key][0]])
    mip = np.transpose(mip, axes=(1,0,2))  # transpose

    # add text labels to MIP
    h = mip.shape[0]
    mip = Image.fromarray(mip)  # convert for writing text
    draw = ImageDraw.Draw(mip)
    for i in range(len(text)):
        x, y, label = text[i]
        draw.text(xy=(x-20,y-15),text=to_fmt(label),
                  font=ImageFont.truetype(font="Arial", size=25), fill=(0,0,0))
    draw.text(xy=(20,h-50), text=to_time(frame, c["general"]["time"]),
                  font=ImageFont.truetype(font="Arial", size=30), fill=(255,255,255))

    return cv2.cvtColor(np.array(mip), cv2.COLOR_RGB2BGR)


# @proc
# def channels_proc(frame, file, directory, c, lookup):
#     """
#     Process job for MIP-ing a frame.
    
#     Parameters
#     ----------
#     frame : int
#         Index of the movie frames.
#     file : str
#         Path to CZI file.
#     directory : str
#         Path to the directory for input/outputs.
#     c : dict
#         Dictionary for config.
#     lookup : dict
#         Dictionary of nuclei tracks.
#     """
#     # get frame path and read info
#     frame_path = os.path.join(directory, "frames", to_fmt(frame))
#     info = np.loadtxt(os.path.join(frame_path, "info.csv"),
#                       delimiter=",", skiprows=1, ndmin=2)
#     # get deconvolve
#     deconvolve = read_tiff(os.path.join(frame_path, "deconvolve.tiff"))

#     # make mask from tracked nuclei
#     masks = np.zeros(deconvolve.shape[1:4], dtype=bool)
#     for label in range(len(info)):
#         key = (frame, label)
#         if key in lookup:
#             # read mask subset
#             l, u = info[label,5:8].astype(int), info[label,8:11].astype(int)
#             mask = read_h5(os.path.join(frame_path, "masks.h5"), to_fmt(label))

#             # add mask
#             mask = np.logical_or(masks[l[0]:u[0],l[1]:u[1],l[2]:u[2]], mask)
#             masks[l[0]:u[0],l[1]:u[1],l[2]:u[2]] = mask

#     # make MIPs
#     mips = dict()
#     mips["all"] = None  # for merging channels
#     for chnl in c["movies"]["channels"]:
#         # make MIP for channel
#         mip = to_mip(deconvolve[chnl], mask=masks, axis=0, **c["movies"]["channels"][chnl])
#         mip = np.transpose(mip, axes=(1,0,2))  # transpose
#         mips[chnl] = mip

#         # add to merged
#         if mips["all"] is None:
#             mips["all"] = np.copy(mip) 
#         else:
#             mips["all"] = np.maximum(mips["all"], mip)

#     # add time to MIPs
#     h = mip.shape[0]
#     for item in mips:
#         mip = Image.fromarray(mips[item])  # convert for writing text
#         draw = ImageDraw.Draw(mip)
#         draw.text(xy=(20,h-50), text=to_time(frame, c["general"]["time"]),
#                   font=ImageFont.truetype(font="Arial", size=30),
#                   fill=(255,255,255))
#         mips[item] = cv2.cvtColor(np.array(mip), cv2.COLOR_RGB2BGR)

#     return mips


@proc
def nuclei_proc(nucleus, directory, c):
    """
    Process job for MIP-ing nucleus channels 
    for a single nucleus track.

    Parameters
    ----------
    nucleus : int
        Index for the tracked nucleus.
    directory : str
        Path for pipeline outputs.
    c : dict
        Dictionary for config.
    """
    # parameters
    background = (255,255,255)  # white
    text = (0,0,0)  # black

    # get channels to show and channel widths/heights (in pixels)
    chnls = [item for item in c["movies"]["channels"]]
    l, s, border = 200, 60, 10
    w, h = (len(chnls)+1)*(l+s)+2*border, l+s+4*border
    
    # get paths
    frames_path = os.path.join(directory, "frames")
    nuclei_path = os.path.join(directory, "nuclei")
    mode_path = os.path.join(directory, "movies", "nuclei")

    # get path to nucleus and read track
    nucleus_path = os.path.join(nuclei_path, to_fmt(nucleus))
    nucleus_track = np.loadtxt(os.path.join(nucleus_path, "info.csv"),
                               delimiter=",", skiprows=1, ndmin=2)
    # compute scale
    pad = c["segmentation"]["pad"] if "pad" in c["segmentation"] else 5
    shapes = nucleus_track[:,8:11] - nucleus_track[:,5:8] - 2*pad
    scale = np.min(np.array([s,l,l]) / np.max(shapes, axis=0))
    half = np.array([s,l,l]) / 2

    # get info for first frame of track
    frame, label = int(nucleus_track[0,0]), int(nucleus_track[0,1])
    frame_path = os.path.join(frames_path, to_fmt(frame))
    bounds = {}  # hold intensity bounds per channel
    
    # read mask/deconvolve for nucleus and compute intensity bounds
    lbl = to_fmt(label)
    mask = read_h5(os.path.join(frame_path, "masks.h5"), lbl)
    dcv = read_h5(os.path.join(frame_path, "deconvolves.h5"), lbl)
    for chnl in chnls:
        lower, upper = 100, 3.0*np.median(dcv[chnl,mask])
        bounds[chnl] = {"lower": lower, "upper": upper}

    # make video writer and iterate through track
    video = cv2.VideoWriter(os.path.join(mode_path, to_fmt(nucleus) + ".mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), c["movies"]["fps"], (w, h))
    n = len(nucleus_track)
    for i in range(n):
        # get frame/label and path
        frame, label = int(nucleus_track[i,0]), int(nucleus_track[i,1])
        frame_path = os.path.join(frames_path, to_fmt(frame))
        
        # read mask/deconvolve for nucleus then remove padding
        lbl = to_fmt(label)
        mask = read_h5(os.path.join(frame_path, "masks.h5"), lbl)
        dcv = read_h5(os.path.join(frame_path, "deconvolves.h5"), lbl)
        mask = mask[pad:-pad,pad:-pad,pad:-pad]
        dcv = dcv[:,pad:-pad,pad:-pad,pad:-pad]
        
        # make empty image and get (scaled) shape/centroid
        mips = np.zeros((h, w, 3), dtype=np.uint8)
        for j in range(3):
            mips[:,:,j] = background[j]
        centroid = scale * np.mean(np.argwhere(mask), axis=0)

        # determine shifts
        center = scale * np.array(mask.shape) / 2
        shape_diff, mass_diff = half - center, center - centroid
        mass_diff = np.clip(mass_diff, -shape_diff, shape_diff)
        z = int(border + shape_diff[0] - mass_diff[0])  # flipped
        y = int(border + shape_diff[1] + mass_diff[1])
        x = int(border + shape_diff[2] + mass_diff[2])

        # add each channel
        z_merge, y_merge, x_merge = None, None, None  # for merging channels
        for j in range(len(chnls)):
            # get channel and offset
            chnl, off = chnls[j], j*(l+s)
            intensity, color = bounds[chnl], c["movies"]["channels"][chnl]["color"]
            
            # get mip along z-axis
            z_mip = to_mip(dcv[chnl], mask=mask, background=background,
                           intensity=intensity, color=color, axis=0)
            z_mip = cv2.resize(cv2.transpose(z_mip), None,
                               fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            z_merge = np.copy(z_mip) if z_merge is None else np.maximum(z_merge, z_mip)

            # get mip along y-axis
            y_mip = to_mip(dcv[chnl], mask=mask, background=background,
                           intensity=intensity, color=color, axis=1)
            y_mip = cv2.resize(cv2.rotate(y_mip, cv2.ROTATE_90_CLOCKWISE), None,
                               fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            y_merge = np.copy(y_mip) if y_merge is None else np.maximum(y_merge, y_mip)

            # get mip along x-axis
            x_mip = to_mip(dcv[chnl], mask=mask, background=background,
                           intensity=intensity, color=color, axis=2)
            x_mip = cv2.resize(cv2.flip(x_mip, 0), None,
                               fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            x_merge = np.copy(x_mip) if x_merge is None else np.maximum(x_merge, x_mip)

            # add MIPs to image
            mips[x:x+z_mip.shape[0], y+off:y+off+z_mip.shape[1]] = z_mip
            mips[x:x+y_mip.shape[0], l+z+off:l+z+off+y_mip.shape[1]] = y_mip
            mips[l+z:l+z+x_mip.shape[0], y+off:y+off+x_mip.shape[1]] = x_mip

        # add merge to image
        off = len(chnls)*(l+s)
        mips[x:x+z_mip.shape[0], y+off:y+off+z_mip.shape[1]] = z_merge
        mips[x:x+y_mip.shape[0], l+z+off:l+z+off+y_mip.shape[1]] = y_merge
        mips[l+z:l+z+x_mip.shape[0], y+off:y+off+x_mip.shape[1]] = x_merge
        
        # add time to image and write
        mips = Image.fromarray(mips)  # convert for writing text
        draw = ImageDraw.Draw(mips)
        draw.text(xy=(10, h-25),
                  text=to_time(frame, c["general"]["time"]),
                  font=ImageFont.truetype(font="Arial", size=18), fill=text)
        video.write(cv2.cvtColor(np.array(mips), cv2.COLOR_RGB2BGR))
    video.release()


@proc
def evaluate_proc(nucleus, directory, c):
    """
    Process job for MIP-ing nucleus channels 
    for a single nucleus track.

    Parameters
    ----------
    nucleus : int
        Index for the tracked nucleus.
    directory : str
        Path for pipeline outputs.
    c : dict
        Dictionary for config.
    """
    # parameters
    background = (255,255,255)  # white
    text = (0,0,0)  # black
    radius, edge = 12, 2

    # get channels to show and channel widths/heights (in pixels)
    chnls = [item for item in c["movies"]["channels"]]
    l, s, border = 200, 60, 10
    w, h = (len(chnls)+1)*(l+s)+2*border, l+s+4*border
    
    # get paths
    frames_path = os.path.join(directory, "frames")
    nuclei_path = os.path.join(directory, "nuclei")
    mode_path = mkdir(os.path.join(directory, "movies", "evaluate", to_fmt(nucleus)))

    # get path to nucleus and read track
    nucleus_path = os.path.join(nuclei_path, to_fmt(nucleus))
    nucleus_track = np.loadtxt(os.path.join(nucleus_path, "info.csv"),
                               delimiter=",", skiprows=1, ndmin=2)
    # compute scale
    pad = c["segmentation"]["pad"] if "pad" in c["segmentation"] else 5
    shapes = nucleus_track[:,8:11] - nucleus_track[:,5:8] - 2*pad
    scale = np.min(np.array([s,l,l]) / np.max(shapes, axis=0))
    half = np.array([s,l,l]) / 2

    # get info for first frame of track
    frame, label = int(nucleus_track[0,0]), int(nucleus_track[0,1])
    frame_path = os.path.join(frames_path, to_fmt(frame))
    bounds = {}  # hold intensity bounds per channel
    
    # read mask/deconvolve for nucleus and compute intensity bounds
    lbl = to_fmt(label)
    mask = read_h5(os.path.join(frame_path, "masks.h5"), lbl)
    dcv = read_h5(os.path.join(frame_path, "deconvolves.h5"), lbl)
    for chnl in chnls:
        lower, upper = 100, 3.0*np.median(dcv[chnl,mask])
        bounds[chnl] = {"lower": lower, "upper": upper}

    # go through tracks in nucleus
    tracks_path = os.path.join(nucleus_path, "tracks")
    for item in os.listdir(tracks_path):
        if item[:-4].isnumeric():
            # make video writer
            video = cv2.VideoWriter(os.path.join(mode_path, f"{item[:-4]}.mp4"),
                                    cv2.VideoWriter_fourcc(*"mp4v"),
                                    c["movies"]["fps"], (w, h))
            # read track from CSV
            track = np.loadtxt(os.path.join(tracks_path, item), delimiter=",",
                               skiprows=1, ndmin=2)
        
            # iterate through track frames
            n = len(track)
            for i in range(n):
                # get frame/label and path
                frame, label = int(nucleus_track[i,0]), int(nucleus_track[i,1])
                frame_path = os.path.join(frames_path, to_fmt(frame))
                
                # read mask/deconvolve for nucleus then remove padding
                lbl = to_fmt(label)
                mask = read_h5(os.path.join(frame_path, "masks.h5"), lbl)
                dcv = read_h5(os.path.join(frame_path, "deconvolves.h5"), lbl)
                mask = mask[pad:-pad,pad:-pad,pad:-pad]
                dcv = dcv[:,pad:-pad,pad:-pad,pad:-pad]
                
                # make empty image and get (scaled) shape/centroid
                mips = np.zeros((h, w, 3), dtype=np.uint8)
                for j in range(3):
                    mips[:,:,j] = background[j]
                centroid = scale * np.mean(np.argwhere(mask), axis=0)

                # determine shifts
                center = scale * np.array(mask.shape) / 2
                shape_diff, mass_diff = half - center, center - centroid
                mass_diff = np.clip(mass_diff, -shape_diff, shape_diff)
                z = int(border + shape_diff[0] - mass_diff[0])  # flipped
                y = int(border + shape_diff[1] + mass_diff[1])
                x = int(border + shape_diff[2] + mass_diff[2])

                # add each channel
                z_merge, y_merge, x_merge = None, None, None  # for merging channels
                for j in range(len(chnls)):
                    # get channel and offset
                    chnl, off = chnls[j], j*(l+s)
                    intensity, color = bounds[chnl], c["movies"]["channels"][chnl]["color"]
                    
                    # get mip along z-axis
                    z_mip = to_mip(dcv[chnl], mask=mask, background=background,
                                intensity=intensity, color=color, axis=0)
                    z_mip = cv2.resize(cv2.transpose(z_mip), None,
                                    fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    z_merge = np.copy(z_mip) if z_merge is None else np.maximum(z_merge, z_mip)

                    # get mip along y-axis
                    y_mip = to_mip(dcv[chnl], mask=mask, background=background,
                                intensity=intensity, color=color, axis=1)
                    y_mip = cv2.resize(cv2.rotate(y_mip, cv2.ROTATE_90_CLOCKWISE), None,
                                    fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    y_merge = np.copy(y_mip) if y_merge is None else np.maximum(y_merge, y_mip)

                    # get mip along x-axis
                    x_mip = to_mip(dcv[chnl], mask=mask, background=background,
                                intensity=intensity, color=color, axis=2)
                    x_mip = cv2.resize(cv2.flip(x_mip, 0), None,
                                    fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    x_merge = np.copy(x_mip) if x_merge is None else np.maximum(x_merge, x_mip)
        
                    # get subset position and draw circle over dots
                    position = track[i,3+chnl*6:6+chnl*6]
                    if not np.isnan(position).all():
                        z_sub, y_sub, x_sub = np.rint(scale * (position - pad)).astype(int)
                        z_mip = cv2.circle(z_mip, (y_sub, x_sub), radius, (0,0,0), edge)   
                        y_mip = cv2.circle(y_mip, (y_mip.shape[1]-z_sub, x_sub), radius, (0,0,0), edge)
                        x_mip = cv2.circle(x_mip, (y_sub, x_mip.shape[0]-z_sub), radius, (0,0,0), edge)

                    # add MIPs to image
                    mips[x:x+z_mip.shape[0], y+off:y+off+z_mip.shape[1]] = z_mip
                    mips[x:x+y_mip.shape[0], l+z+off:l+z+off+y_mip.shape[1]] = y_mip
                    mips[l+z:l+z+x_mip.shape[0], y+off:y+off+x_mip.shape[1]] = x_mip

                # add merge to image
                off = len(chnls)*(l+s)
                mips[x:x+z_mip.shape[0], y+off:y+off+z_mip.shape[1]] = z_merge
                mips[x:x+y_mip.shape[0], l+z+off:l+z+off+y_mip.shape[1]] = y_merge
                mips[l+z:l+z+x_mip.shape[0], y+off:y+off+x_mip.shape[1]] = x_merge
                
                # add time to image and write
                mips = Image.fromarray(mips)  # convert for writing text
                draw = ImageDraw.Draw(mips)
                draw.text(xy=(10, h-25),
                        text=to_time(frame, c["general"]["time"]),
                        font=ImageFont.truetype(font="Arial", size=18), fill=text)
                video.write(cv2.cvtColor(np.array(mips), cv2.COLOR_RGB2BGR))
            video.release()


from .utilities import read_czi

@proc
def chnls_proc(frame, file, directory, c, lookup):
    """
    Process job for MIP-ing a frame.
    
    Parameters
    ----------
    frame : int
        Index of the movie frames.
    file : str
        Path to CZI file.
    directory : str
        Path to the directory for input/outputs.
    c : dict
        Dictionary for config.
    lookup : dict
        Dictionary of nuclei tracks.
    """ 
    # get deconvolve
    name = os.path.splitext(file)[0]
    deconvolve = read_czi(f"{name}_LatticelightsheetDeconvolved(5.0).czi", frame)

    # make MIPs
    mips = dict()
    for chnl in [1]:
        # make MIP for channel
        mip = to_mip(deconvolve[chnl], axis=0, **c["movies"]["channels"][chnl])
        mip = np.transpose(mip, axes=(1,0,2))  # transpose
        mips[chnl] = mip

    # add time to MIPs
    h = mip.shape[0]
    for item in mips:
        mip = Image.fromarray(mips[item])  # convert for writing text
        draw = ImageDraw.Draw(mip)
        draw.text(xy=(20,h-50), text=to_time(frame, c["general"]["time"]),
                  font=ImageFont.truetype(font="Arial", size=30),
                  fill=(255,255,255))
        mips[item] = cv2.cvtColor(np.array(mip), cv2.COLOR_RGB2BGR)

    return mips