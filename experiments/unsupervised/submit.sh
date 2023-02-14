#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name  featurize_hMOF
#SBATCH --get-user-env
#SBATCH --ntasks    1
#SBATCH --cpus-per-task   8
#SBATCH --mem 32G
#SBATCH --time       48:00:00

module purge
python train.py