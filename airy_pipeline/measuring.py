import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from multiprocessing import set_start_method, Pool
from airy_pipeline.methods.measuring import measuring_proc
from airy_pipeline.methods import read_options, to_directory, configure_plots, \
     bar_format, read_yaml, read_kernel, read_shape, save_config, mkdir, where

set_start_method("spawn", force=True)  # helps with GPUs
configure_plots()  # matplotlib config


def run(file=None, beads=None, psf=None, directory=None, config=None, start=0,
        end=None, nproc=8, gpus=1, mode=None, other=None, force=False):
    """
    Runs the measuring module to measure features
    within nuclei.

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
    if psf is None:
        print("PSF(s) are required for this module!") 
        exit()
    if directory is None:
        directory = to_directory(file)

    # get paths
    frames_path = os.path.join(directory, "frames")
    nuclei_path = os.path.join(directory, "nuclei")

    # handle for missing frames/nuclei
    if (not os.path.exists(frames_path)) or (not os.path.exists(nuclei_path)):
        print("Nuclei must be segmented and tracked before this module!")
        exit()

    # get nuclei to process
    nuclei, overwrite = [], force
    for item in os.listdir(nuclei_path):
        if item.isnumeric():
            # get path to nucleus
            nucleus_path = os.path.join(nuclei_path, item)
            nuclei.append(int(item))

            # get path and handle existing
            tracks_path = os.path.join(nucleus_path, "tracks")
            if not overwrite and os.path.exists(tracks_path):
                ans = input("Features already measured! Do you want to overwrite? (y/n): ")
                if ans == "y":
                    overwrite = True
                else:
                    exit()

    # handle config
    if config is None:
        c = {"measuring": {}}
    else:
        c = read_yaml(config)
        if "measuring" not in c:
            c["measuring"] = {}
        save_config(config, directory)
    n = len(nuclei)  # number of nuclei

    # -------------- measure polycomb bodies --------------
    
    # make jobs and use process pool
    devices = np.arange(n) % gpus
    args = list(zip(nuclei, n*[directory], n*[c], devices))
    with Pool(min(n,nproc), maxtasksperchild=5) as pool:
        for _ in tqdm(pool.imap_unordered(measuring_proc, args), desc="Measuring",
                      bar_format=bar_format("nuclei"), total=n):
            pass


if __name__ == "__main__":
    # read args from command-line and run
    kwargs = read_options()
    run(**kwargs)
