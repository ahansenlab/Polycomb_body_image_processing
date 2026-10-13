import os
import shutil
import pandas as pd

from tqdm import tqdm
from multiprocessing import set_start_method, Pool
from airy_pipeline.methods.tracking import nuclei_proc, features_proc
from airy_pipeline.methods import read_options, to_directory, read_yaml, \
     save_config, mkdir, bar_format, to_fmt, read_shape

set_start_method("spawn", force=True)  # helps with GPUs


def run(file=None, beads=None, psf=None, directory=None, config=None, start=0,
        end=None, nproc=8, gpus=1, mode=None, other=None, force=False):
    """
    Runs the tracking module to track segmented
    nuclei and detected features.

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
    if directory is None:
        if file is not None:
            directory = to_directory(file)
        else:
            print("Directory is required for this module!") 
            exit()

    # make paths and set modes
    frames_path = os.path.join(directory, "frames")
    nuclei_path = os.path.join(directory, "nuclei")
    modes = ["nuclei", "features"]

    # tracking nuclei
    if mode == "nuclei":
        # handle for missing frames
        if not os.path.exists(frames_path):
            print("Nuclei must be segmented before they are tracked!")
            exit()
            
        # handle existing
        if os.path.exists(nuclei_path):
            prompt = "Nuclei already tracked! Do you want to overwrite? (y/n): "
            if force or (input(prompt) == "y"):
                shutil.rmtree(nuclei_path)
                os.makedirs(nuclei_path)
            else:
                exit()

        # handle config
        if config is not None:
            c = read_yaml(config)
            if "tracking" not in c:
                c["tracking"] = {}
            if "nuclei" not in c["tracking"]:
                c["tracking"]["nuclei"] = {}
        else:
            c = {"tracking": {"nuclei": {}}}
        save_config(config, directory)
        
        # make progress bar
        fmt = "{l_bar}{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        bar = tqdm(total=1, desc="Tracking", bar_format=fmt)
        
        # merge all positions
        positions = pd.DataFrame()
        for item in os.listdir(frames_path):
            if item.isnumeric():
                # read CSV
                info = pd.read_csv(os.path.join(frames_path, item, "info.csv"))
                if not info.empty:
                    positions = pd.concat([positions, info], ignore_index=True)

        # run tracking
        if len(positions) > 0:
            # make and iterate through tracks
            tracks = nuclei_proc(positions, c)
            for label in range(len(tracks)):
                # make directory and save as CSV
                nucleus_path = mkdir(os.path.join(nuclei_path, to_fmt(label)))
                tracks[label].to_csv(os.path.join(nucleus_path, "info.csv"), index=False)
        bar.update()
        bar.close()

    # tracking features (dots)
    elif mode == "features":
        # handle for missing frames/nuclei
        if (not os.path.exists(frames_path)) or (not os.path.exists(nuclei_path)):
            print("Nuclei must be segmented and tracked before this module.")
            exit()
        
        # get nuclei to process
        nuclei, overwrite = [], force
        for item in os.listdir(nuclei_path):
            if item.isnumeric():
                # get path to channel(s)
                channels_path = os.path.join(nuclei_path, item, "channels")
                nuclei.append(int(item))

                # iterate through channels
                for subitem in os.listdir(channels_path):
                    if subitem.isnumeric():
                        # get path and handle existing
                        tracks_path = os.path.join(channels_path, subitem, "tracks")
                        if not overwrite and os.path.exists(tracks_path):
                            ans = input("Features already tracked! Do you want to" \
                            " overwrite? (y/n): ")
                            if ans == "y":
                                overwrite = True
                            else:
                                exit()
                    else:
                        continue

        # handle config
        if config is not None:
            c = read_yaml(config)
            if "tracking" not in c:
                c["tracking"] = {}
            if "features" not in c["tracking"]:
                c["tracking"]["features"] = {}
        else:
            c = {"tracking": {"features": {}}}
        save_config(config, directory)

        # get number of nuclei and read PSF mask shapes(s)
        n = len(nuclei)
        shape = None if psf is None else read_shape(psf, c["general"]["lasers"])

        # make args and use process pool
        args = list(zip(nuclei, n*[shape], n*[directory], n*[c]))
        with Pool(min(n,nproc), maxtasksperchild=5) as pool:
            for _ in tqdm(pool.imap_unordered(features_proc, args), desc="Tracking",
                          bar_format=bar_format("nuclei"), total=n):
                pass  # update progress

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
