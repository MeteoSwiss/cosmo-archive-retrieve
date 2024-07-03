#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=postproc
#SBATCH --time=1-00:00:00  # Set a maximum runtime of 2 days

python ../cosmo_archive_retrieve/convert_grib_to_p.py -n 40 --tempdir /scratch/cosuna/temp/


