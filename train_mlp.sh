#!/bin/bash

#SBATCH --job-name=25751GalParamSearsh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --output=slurm_output_%A.out

python optuna_search.py