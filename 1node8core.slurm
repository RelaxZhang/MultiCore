#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=0-12:00:00
#SBATCH --output=1node8core.txt

# Load required modules
module load python/3.7.4
module load mpi4py/3.0.2-timed-pingpong
module load numpy/1.18.0-python-3.7.4

time srun  -n 8 python3 twitterProcessor.py