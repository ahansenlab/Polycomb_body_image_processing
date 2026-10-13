import os
import numpy as np

from tqdm import tqdm
from multiprocessing import set_start_method, Pool
from airy_pipeline.methods.detection import detection_proc
from airy_pipeline.methods import read_options, to_directory, read_yaml, \
     save_config, bar_format, read_shape

set_start_method("spawn", force=True)  # helps with GPUs


def run(file=None, beads=None, psf=None, directory=None, config=None, start=0,
        end=None, nproc=8, gpus=1, mode=None, other=None, force=False):
    """
    Runs the detection module to identify features
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
    if psf is None:
        print("PSF(s) are required for this module!") 
        exit()
    if directory is None:
        if file is not None:
            directory = to_directory(file)
        else:
            print("Directory is required for this module!") 
            exit()

    # make paths
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
            channels_path = os.path.join(nucleus_path, "channels")
            if not overwrite and os.path.exists(channels_path):
                ans = input("Features already detected! Do you want to overwrite? (y/n): ")
                if ans == "y":
                    overwrite = True
                else:
                    exit()

    # handle config
    if config is None:
        c = {"detection": {"channels": {}}}
    else:
        c = read_yaml(config)
        if "detection" not in c:
            c["detection"] = {}
        if "channels" not in c["detection"]:
            c["detection"]["channels"] = {}
        save_config(config, directory)

    # get number of nuclei and read PSF mask shapes(s)
    n = len(nuclei)
    shape = None if psf is None else read_shape(psf, c["general"]["lasers"])
    
    # make args and use process pool
    args = list(zip(nuclei, n*[shape], n*[directory], n*[c], np.arange(n) % gpus))
    with Pool(min(n,nproc), maxtasksperchild=5) as pool:
        for _ in tqdm(pool.imap_unordered(detection_proc, args), desc="Detecting",
                      bar_format=bar_format("nuclei"), total=n):
            pass  # update progress


if __name__ == "__main__":
    # read args from command-line and run
    kwargs = read_options()
    run(**kwargs)
