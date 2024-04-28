import subprocess
import os


def main():

    try:
        prefix = os.environ["CONDA_PREFIX"]
    except KeyError:
        raise RuntimeError(
            "CONDA_PREFIX is not defined in the environment. It needs to be defined and set to the root of the conda environment."
        )
    if not os.path.isdir(prefix):
        raise RuntimeError("CONDA prefix does not exists.")
    clone_dir = os.path.join(prefix, "share", "eccodes-cosmo-resources")
    if not os.path.exists(clone_dir):
        subprocess.run(
            [
                "git",
                "clone",
                "-b",
                "grib1_mswiss",
                "git@github.com:cosunae/eccodes-cosmo-resources",
                clone_dir,
            ],
            check=True,
        )
