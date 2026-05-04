#!/bin/bash
#SBATCH --job-name=p2-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=$PWD

python -m scripts.train_base --idx ${SLURM_ARRAY_TASK_ID} --n-cand 50
