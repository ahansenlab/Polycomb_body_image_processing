import os
import trackpy
import numpy as np
import pandas as pd

from .utilities import proc, read_h5, mkdir, fracshift, \
     where, to_fmt

# turn off logs and messages
trackpy.ignore_logging()
trackpy.quiet(suppress=True)


def nuclei_proc(positions, c):
    """
    Process job for tracking nuclei.

    Parameters
    ----------
    positions : pandas.DataFrame
        Positions of nuclei across frames.
    c : dict
        Dictionary for config.
    """
    return nuclei(positions, **c["tracking"]["nuclei"])


def nuclei(positions, search=25, length=60):
    """
    Runs linking on a data frame of nuclei
    positions.

    Parameters
    ----------
    positions : pandas.DataFrame
        Positions of nuclei across frames.
    search : float, optional
        Search radius for nuclei (in pixels).
    length : int, optional
        Minimum length of tracks.

    Returns
    -------
    list
        A list of data frames for each track.
    """
    columns = ["z_cen (px)", "y_cen (px)", "x_cen (px)"]
    
    # use trackpy to link nuclei
    positions = trackpy.link(positions, pos_columns=columns,
                             search_range=search)
    positions = trackpy.filtering.filter_stubs(positions, threshold=length)
    positions.reset_index(drop=True, inplace=True)

    # get tracks as separate data frames
    tracks = []
    for _, particle in positions.groupby("particle"):
        track = particle.drop(columns=["particle"])  # drop particle column
        tracks.append(track)
    
    return tracks


@proc
def features_proc(nucleus, shape, directory, c):
    """
    Process job for tracking features within a
    single nucleus.

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
    """
    # set CSV format
    fmt = 2*["%d"] + 6*["%.3f"] + 2*["%d"]
    header = "frame,label,z_map (px),y_map (px),x_map (px)," + \
             "z_sub (px),y_sub (px),x_sub (px),sisters,detected"
    
    # make paths and read track
    frames_path = os.path.join(directory, "frames")
    nucleus_path = os.path.join(directory, "nuclei", to_fmt(nucleus))
    nucleus_track = np.loadtxt(os.path.join(nucleus_path, "info.csv"),
                               delimiter=",", skiprows=1, ndmin=2)
    # go through channels
    chnls_path = os.path.join(nucleus_path, "channels")
    for chnl in c["detection"]["channels"]:
        # get paths
        chnl_path = os.path.join(chnls_path, to_fmt(chnl))
        tracks_path = mkdir(os.path.join(chnl_path, "tracks"))

        # read and track
        info = pd.read_csv(os.path.join(chnl_path, f"info.csv"))
        if len(info) > 0:
            # run linking and iterate through each
            tracks = features(info, **c["tracking"]["features"])
            for i in range(len(tracks)):
                track = tracks[i].to_numpy()

                # break up track to fill in (if necessary)
                info = []
                for j in range(len(track)):
                    # add frame
                    line = np.empty(10)
                    line[0:9] = track[j,0:9]
                    line[9] = 1  # mark detected
                    info.append(line)
                    if j == len(track) - 1:
                        break  # break if end of track
                        
                    # fill gap (if exists)
                    start, end = int(track[j,0]), int(track[j+1,0])
                    if end - start > 1:
                        # compute slope and iterate over gap
                        slope = (track[j+1,2:5] - track[j,2:5]) / (start - end)
                        for frame in range(start+1, end):
                            # get path to frame and row in nucleus track
                            frame_path = os.path.join(frames_path, to_fmt(frame))
                            idx = where(nucleus_track[:,0] == frame)
                            label = int(nucleus_track[idx,1])  # get frame label
        
                            # read mask/deskew for nucleus
                            lbl = to_fmt(label)
                            mask = read_h5(os.path.join(frame_path, "masks.h5"), lbl)
                            dsk = read_h5(os.path.join(frame_path, "deskews.h5"), lbl)
                            centroid = np.mean(np.argwhere(mask).astype(float), axis=0)
        
                            # make line
                            line = np.empty(10)
                            line[0], line[1] = frame, label
                            line[2:5] = track[j,2:5] + slope * (frame - start)
                            line[5:8] = line[2:5]
                            line[5:8] += centroid  # subset position
        
                            # get subpixel position
                            p = np.rint(line[5:8]).astype(int)  # position in subset
                            if (p >= 0).all() and (p < mask.shape).all() and (mask[p[0],p[1],p[2]]):
                                # use fracshift to refine position
                                initial = np.copy(line[5:8])
                                line[5:8], inbounds = fracshift(dsk[chnl], initial,
                                                                subtract=True, mask=mask,
                                                                shape=shape[chnl], radius=7)
                                line[2:5] += line[5:8] - initial  # adjust mapped
                            else:
                                inbounds = False

                            # bad interpolation set to NaNs
                            if not inbounds:
                                line[2:8] = np.nan
                            line[8:10] = 0  # no sisters or detection
                            info.append(line)

                # save as CSV
                np.savetxt(os.path.join(tracks_path, to_fmt(i) + ".csv"), np.asarray(info),
                           delimiter=",", header=header, fmt=fmt, comments="")


def features(positions, search=10, length=15, memory=5):
    """
    Runs tracking on a dataframe of feature
    positions.

    Parameters
    ----------
    positions : pandas.DataFrame
        Dataframe of positions across frames.
    search : float, optional
        Isotropic search radius for features.
    length : int, optional
        Minimum length of feature tracks.
    memory : int, optional
        Maxmimum allowed frames to skip.

    Returns
    -------
    list
        A list of dataframes for each track.
    """
    columns = ["z_map (px)", "y_map (px)", "x_map (px)"]
        
    # use trackpy to link nuclei
    positions = trackpy.link(positions, pos_columns=columns,
                             search_range=search, memory=memory)
    positions = trackpy.filtering.filter_stubs(positions, threshold=length)
    positions.reset_index(drop=True, inplace=True)

    # get tracks as separate data frames
    tracks = []
    for _, particle in positions.groupby("particle"):
        track = particle.drop(columns=["particle"])  # drop particle column
        tracks.append(track)
    
    return tracks
