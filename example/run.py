from airy_pipeline import beads, measuring, segmentation, detection, tracking, movies
from airy_pipeline.methods import read_options


def run():
    """
    Example function for running multiple modules.
    """
    # read command-line options (as dict)
    kwargs = read_options()
    kwargs.pop("mode")
    
    # run the main modules
    segmentation.run(**kwargs)
    tracking.run(**kwargs, mode="nuclei")
    ##detection.run(**kwargs)
    ##tracking.run(**kwargs, mode="features")
    ##beads.run(**kwargs, mode="fitting")
    measuring.run(**kwargs)

    # make outputs
    movies.run(**kwargs, mode="labels")
#    movies.run(**kwargs, mode="channels")
    movies.run(**kwargs, mode="nuclei")
#    movies.run(**kwargs, mode="evaluate")

    
if __name__ == "__main__":
    run()
