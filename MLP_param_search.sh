#!/bin/bash
#SBATCH --job-name=MLP_mrmr # Name of the job, can be useful for identifying your job later
#SBATCH --cpus-per-task=4      # Maximum amount of CPU cores (per MPI process)
#SBATCH --mem=16G              # Maximum amount of system memory (RAM) 
#SBATCH --gres=gpu:v100:1           # Request 1 GPU
#SBATCH --gres=shard:8          # Request 8 GB of VRAM

#SBATCH --time=0-03:00         # Time limit (DD-HH:MM)
#SBATCH --nice=100             # Allow other priority jobs to go first

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"


wandb login $WANDB_API_KEY

srun nice -n 15 python MLP_wandb_cv.py --dataset data/csv/rds_under_8_severe_v2.csv --data_dir data/fetched/25751 --sweep_config configs/MLP_adam.yaml