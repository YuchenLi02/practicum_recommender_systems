#!/bin/bash
#SBATCH --job-name=p2-rerank
#SBATCH --partition=macula
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rerank_%j.out
#SBATCH --error=logs/rerank_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=$PWD

python -m scripts.run_rerank --workers 8
python -m scripts.aggregate
