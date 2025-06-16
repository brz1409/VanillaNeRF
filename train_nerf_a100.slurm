#!/bin/bash
#SBATCH --job-name=nerf-train-v100
#SBATCH --partition=GPU-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "Starte Training um: $(date)"
echo "Node: $(hostname)"
echo "Job-ID: $SLURM_JOB_ID"

# Aktiviere Python-Umgebung
source venv/bin/activate

# Starte das Training
python train.py --data_dir scene/Simulation_20/ --config configs/default.json

echo "Fertig um: $(date)"
