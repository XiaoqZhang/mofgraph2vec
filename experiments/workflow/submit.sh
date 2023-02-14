#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name ch4+rsm,qmof
#SBATCH --get-user-env
#SBATCH --ntasks    1
#SBATCH --cpus-per-task   8
#SBATCH --mem 32G
#SBATCH --time       24:00:00

module purge
python sweep.py