import os
import shutil
import numpy as np

from tqdm import tqdm
from multiprocessing import set_start_method, Manager, Pool
from airy_pipeline.methods.segmentation import segmentation_proc
from airy_pipeline.methods import read_options, to_directory, to_end, \
     bar_format, read_yaml, read_kernel, save_config

set_start_method("spawn", force=True)  # helps with GPUs


def run(file=None, beads=None, psf=None, directory=None, config=None, start=0,
        end=None, nproc=8, gpus=1, mode=None, other=None, force=False):
    """
    Runs the segmentation module to detect nuclei
    within each frame.

    Parameters
    ----------
    file : str, optional
        Path to CZI file.
    beads : str, optional
        Path to TXT file of bead files.
    psf : str, optional
        Path to H5 file of PSFs.
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
    if file is None:
        print("File is required for this module!") 
        exit()
    if directory is None:
        directory = to_directory(file)

    # make path and get all frames
    frames_path = os.path.join(directory, "frames")
    if end is None:
        end = to_end(file)
    frames = list(range(start, end))

    # handle existing
    if os.path.exists(frames_path):
        prompt = "Segmented nuclei already exist! Do you want to overwrite? (y/n): "
        if force or ((ans := input(prompt)) == "y"):
            # remove existing
            shutil.rmtree(frames_path)
            os.makedirs(frames_path)
            
        elif ans == "n":
            ans = input("Do you want segment missing (or additional) frames? (y/n): ")
            if ans == "y":
                for item in os.listdir(frames_path):
                    # if info exists and attempt to remove
                    if os.path.exists(os.path.join(frames_path, item, "info.csv")):
                        frame = int(item)
                        if frame in frames:
                            frames.remove(frame)
            else:
                exit()
        else:
            exit()
    else:
        os.makedirs(frames_path)

    # handle config
    if config is None:
        c = {"segmentation": {}}
    else:
        c = read_yaml(config)
        if "segmentation" not in c:
            c["segmentation"] = {}
        save_config(config, directory)

    # get number of frames and read kernel(s)
    n = len(frames)
    print(n)
    kernel = None if psf is None else read_kernel(psf, c["general"]["lasers"])
    
    # use manager for locking GPUs
    with Manager() as manager:
        # make process jobs
        devices = np.arange(n) % gpus
        locks = [manager.Lock() for _ in range(gpus)]
        args = list(zip(frames, n*[file], n*[directory], n*[c],
                        n*[kernel], devices, n*[locks]))
        
        # use process pool
        with Pool(min(n,nproc), maxtasksperchild=5) as pool:
            for _ in tqdm(pool.imap_unordered(segmentation_proc, args), desc="Segmenting",
                          bar_format=bar_format("frames"), total=n):
                pass  # update progress
        

if __name__ == "__main__":
    # read args from command-line and run
    kwargs = read_options()
    run(**kwargs)
