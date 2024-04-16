import subprocess
import os


def main():

    prefix = os.environ["CONDA_PREFIX"]
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
