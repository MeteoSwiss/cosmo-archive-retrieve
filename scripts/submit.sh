#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --partition=pp-long
#SBATCH --time=2-00:00:00  # Set a maximum runtime of 2 days

python ../cosmo_archive_retrieve/create_zarr_archive.py -n 14 --tempdir /scratch/cosuna/temp/


