#!/bin/bash
# SLURM batch script to train the NeRF model on an A100 GPU.
#SBATCH --job-name=nerf-train-v100
#SBATCH --partition=GPU-v100  # queue with V100 GPUs
#SBATCH --gres=gpu:v100:1     # request one GPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "Starte Training um: $(date)"
echo "Node: $(hostname)"
echo "Job-ID: $SLURM_JOB_ID"

# Activate Python virtual environment
source venv/bin/activate

# Starte das Training
# Launch the actual training script
python train.py --data_dir scene/Simulation_20/ --config configs/default.json

echo "Fertig um: $(date)"
