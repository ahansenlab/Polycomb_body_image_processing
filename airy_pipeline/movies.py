import os
import cv2
import numpy as np

from tqdm import tqdm
from multiprocessing import set_start_method, Pool
from airy_pipeline.methods.movies import labels_proc, chnls_proc, \
     nuclei_proc, evaluate_proc
from airy_pipeline.methods import read_options, read_yaml, save_config, \
     bar_format, mkdir, to_fmt, make_color, to_directory

set_start_method("spawn", force=True)  # helps with GPUs


def run(file=None, beads=None, psf=None, directory=None, config=None, start=0,
        end=None, nproc=8, gpus=1, mode=None, other=None, force=False):
    """
    Runs the movies module to make movies
    of segmentation and tracking.

    Parameters
    ----------
    file : str, optional
        Path to CZI file.
    beads : str, optional
        Path to TXT file of bead files.
    psf : str, optional
        Path to H5 file of PSFs.3
    directory : str, optional
        Path for pipeline outputs.
    config : str, optional
        Path to YAML file.
    start : int, optional
        Starting frame (python indexed).
    end : int, optional
        Ending frame (python indexed).
    nproc : int, optional
        Number of processes used.
    gpus : int, optional
        Number of GPUs to used.
    mode : str, optional
        Module specific mode.
    other : any, optional
        For pipeline testing.
    force: bool, optional
        If forcing overwrite.
    """
    # argument handling
    if directory is None:
        if file is not None:
            directory = to_directory(file)
        else:
            print("Directory is required for this module!") 
            exit()
    
    # get paths and set modes
    frames_path = os.path.join(directory, "frames")
    nuclei_path = os.path.join(directory, "nuclei")
    movies_path = os.path.join(directory, "movies")
    modes = ["labels", "channels", "nuclei", "evaluate"]

    # handle for missing frames/nuclei
    # if (not os.path.exists(frames_path)) or (not os.path.exists(nuclei_path)):
        # print("Nuclei must be segmented and tracked before this module!")
        # exit()
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(movies_path):
        os.makedirs(movies_path)  # make directory
    
    # make movies of frames
    if mode in ["channels", "labels"]:
        # argument handling
        if file is None:
            print("File is required for this mode!") 
            exit()
            
        # get all frames
        frames = []
        for item in os.listdir(frames_path):
            # assume step is done if exists
            if os.path.exists(os.path.join(frames_path, item, "info.csv")):
                # add frame (if desired)
                frame = int(item)
                if (start is not None) and (frame < start):
                    continue
                if (end is not None) and (frame >= end):
                    continue
                frames.append(frame)
        frames.sort()

        # handle existing
        prompt = "Movie(s) already exist! Do you want to overwrite? (y/n): "
        if mode == "labels":
            labels_path = os.path.join(movies_path, "labels.mp4")
            if os.path.exists(labels_path):
                if force or (input(prompt) == "y"):
                    os.remove(labels_path)
                else:
                    exit()
        else:
            channels_path = os.path.join(movies_path, "channels")
            if os.path.exists(channels_path):
                if force or (input(prompt) == "y"):
                    pass
                else:
                    exit()
            mkdir(channels_path)

       # handle config
        if config is not None:
            c = read_yaml(config)
            if "movies" not in c:
                c["movies"] = {}
        else:
            c = {"movies": {}}
        save_config(config, directory)

        # make lookup for nuclei tracks
        lookup = dict()
        for item in os.listdir(nuclei_path):
            if item.isnumeric():                
                # get path to nucleus and read track
                nucleus_path = os.path.join(nuclei_path, to_fmt(int(item)))
                nucleus_track = np.loadtxt(os.path.join(nucleus_path, "info.csv"),
                                           delimiter=",", skiprows=1, ndmin=2)
                # get color and add
                color = make_color()
                for i in range(len(nucleus_track)):
                    frame, label = int(nucleus_track[i,0]), int(nucleus_track[i,1])
                    lookup[(frame, label)] = (int(item), color)

        # start bar and make process jobs
        n = len(frames)
        progress = tqdm(desc="Movie-ing", bar_format=bar_format("frames"), total=n)
        args = list(zip(frames, n*[file], n*[directory], n*[c], n*[lookup]))

        # run jobs for labels mode
        if mode == "labels":
            # make first frame to initialize writer
            mip = labels_proc(*args[0])
            video = cv2.VideoWriter(labels_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                    c["movies"]["fps"], mip.shape[:2][::-1])
            video.write(mip)
            progress.update()
            
            # use process pool
            with Pool(min(n,nproc), maxtasksperchild=5) as pool:
                for mip in pool.imap(labels_proc, args[1:]):
                    video.write(mip)  # write frame
                    progress.update()
            video.release()  # close writer

        # run jobs for channels mode
        else:
            # make first frame to initialize writers
            mips = chnls_proc(*args[0])
            videos = dict()
            for item in mips:
                mip = mips[item]
                name = to_fmt(item) + ".mp4" if type(item) is int else item + ".mp4"
                videos[item] = cv2.VideoWriter(os.path.join(channels_path, name),
                                               cv2.VideoWriter_fourcc(*"mp4v"),
                                               c["movies"]["fps"], mip.shape[:2][::-1])
                videos[item].write(mip)
            progress.update()
            
            # use process pool
            with Pool(min(n,nproc), maxtasksperchild=5) as pool:
                for mips in pool.imap(chnls_proc, args[1:]):
                    for item in mips:
                        videos[item].write(mips[item])  # write frame
                    progress.update()
            for item in mips:
                videos[item].release()  # close writers

    # make movies for subsets and evaluating
    elif mode in ["nuclei", "evaluate"]:        
        # make path and handle existing
        mode_path = os.path.join(movies_path, mode)
        if os.path.exists(mode_path):
            prompt = "Movies already exist! Do you want to overwrite? (y/n): "
            if force or (input(prompt) == "y"):
                pass
            else:
                exit()
        mkdir(mode_path)

       # handle config
        if config is not None:
            c = read_yaml(config)
            if "movies" not in c:
                c["movies"] = {}
        else:
            c = {"movies": {}}
        save_config(config, directory)

        # configure process for mode and get nuclei
        mode_proc = nuclei_proc if mode == "nuclei" else evaluate_proc
        nuclei = [int(item) for item in os.listdir(nuclei_path) if item.isnumeric()]
        
        # get number of frames and make process jobs
        n = len(nuclei)
        args = list(zip(nuclei, n*[directory], n*[c]))
        
        # use process pool
        with Pool(min(n,nproc), maxtasksperchild=5) as pool:
            for _ in tqdm(pool.imap_unordered(mode_proc, args), desc="Movie-ing",
                          bar_format=bar_format("nuclei"), total=n):
                pass
    
    # if invalid mode
    elif mode is None:
        print("Module requires a mode! Please use \"--mode\" to indicate.")
    else:
        s = ", "
        print(f"That mode doesn't exist! Available modes: {s.join(modes)}")
        

if __name__ == "__main__":
    # read args from command-line and run
    kwargs = read_options()
    run(**kwargs)
