#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=2node8core.txt
#SBARCH --error=1-1err.txt

module load python/3.7.4
module load mpi4py/3.0.2-timed-pingpong
module load numpy/1.18.0-python-3.7.4

time srun  -n 8 python3 twitterProcessor.py 