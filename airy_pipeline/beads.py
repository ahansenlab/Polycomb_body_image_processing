import os
import numpy as np

from tqdm import tqdm
from multiprocessing import Manager, Pool
from multiprocessing import set_start_method
from airy_pipeline.methods.beads import finding_proc, fitting_proc, to_shape
from airy_pipeline.methods import read_options, to_directory, save_config, \
     read_yaml, bar_format, write_psf

set_start_method("spawn", force=True)  # helps with GPUs
    

def run(file=None, beads=None, psf=None, directory=None, config=None, start=0,
        end=None, nproc=8, gpus=1, mode=None, other=None, force=False):
    """
    Runs the correction module to correct for 
    chromatic aberrations.

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
    if beads is None:
        print("Beads are required for this module!") 
        exit()
    if mode == "fitting":
        if file is None:
            print("File is required for this module!") 
            exit()
        if directory is None:
            directory = to_directory(file)

    # set modes
    modes = ["finding", "fitting"]

    # detecting beads
    if mode == "finding":
        # read bead file
        with open(beads, "r") as f:
            files = [line.strip() for line in f if line.strip()]
        n = len(files)
            
        # process each line 
        prompt = "Bead detections already exist! Do you want to overwrite? (y/n): "
        for f in files:
            # handle existing directory
            if os.path.exists(to_directory(f)):
                if force or (input(prompt) == "y"):
                    break
                else:
                    exit()

        # handle config
        if config is None:
            c = {"beads": {"finding": {}}}
        else:
            c = read_yaml(config)
            if "beads" not in c:
                c["beads"] = {}
            if "finding" not in c["beads"]:
                c["beads"]["finding"] = {}
       
        # use manager for locking GPUs
        kernel, samples = 0, 0  # hold for estimating after
        with Manager() as manager:
            # make process jobs
            devices = np.arange(n) % gpus
            locks = [manager.Lock() for _ in range(gpus)]
            args = list(zip(files, n*[config], n*[c], devices, n*[locks]))
    
            # use process pool
            with Pool(min(n,nproc), maxtasksperchild=5) as pool:
                for k, s in tqdm(pool.imap_unordered(finding_proc, args), desc="Finding",
                                 bar_format=bar_format("movies"), total=n):
                    # update kernel/samples
                    kernel += k
                    samples += s

        # handle PSF saving
        prompt = "Do you want to save the PSF(s)? (y/n): "
        if force or (input(prompt) == "y"):
            while True:
                # get path (and check ending)
                if psf is None:
                    prompt = "Where should the PSF (.h5) file be saved? "
                    psf = input(prompt)
                ext = ".h5"
                if not psf.endswith(ext):
                    psf += ext
    
                # handle existing
                if os.path.exists(psf):
                    prompt = f"{psf} already exists! Do you want to overwrite? (y/n): "
                    if force or (input(prompt) == "y"):
                        break
                    else:
                        psf = None
                else:
                    break

            # hold shape for each channel
            n = max(c["general"]["lasers"]) + 1
            shape = [None for _ in range(n)]

            # get kernel and shape for each channel
            for chnl in c["general"]["lasers"]:
                # make PSF integrate to one
                kernel[chnl] /= samples[chnl]
                kernel[chnl] -= np.min(kernel[chnl])
                kernel[chnl] /= np.sum(kernel[chnl])

                # compute mask shape matrix
                shape[chnl] = to_shape(kernel[chnl], angle=c["deskewing"]["angle"])

            # save PSF as H5
            shape = np.asarray(shape)
            write_psf(psf, kernel, shape, c["general"]["lasers"])

    # fitting chromatic aberrations
    elif mode == "fitting":
        # handle existing
        aberrations_path = os.path.join(directory, "corrections", "aberrations")
        if os.path.exists(aberrations_path):
            prompt = "Bead corrections already exist! Do you want to overwrite? (y/n): "
            if force or (input(prompt) == "y"):
                pass
            else:
                exit()
        
        # handle config
        if config is None:
            c = {"beads": {"finding": {}}}
        else:
            c = read_yaml(config)
            if "beads" not in c:
                c["beads"] = {}
            if "finding" not in c["beads"]:
                c["beads"]["finding"] = {}
            save_config(config, directory)
        
        # make progress bar
        fmt = "{l_bar}{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        bar = tqdm(total=1, desc="Fitting", bar_format=fmt)

        # run fitting
        fitting_proc(file, directory, beads, c)
        bar.update()
        bar.close()
    
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
