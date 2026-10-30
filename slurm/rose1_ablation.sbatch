#!/bin/bash
# SLURM job: runs the full ROSE-1 ablation (scripts/rose1_ablation.py) on one GPU.
# Submit from the repo root:   sbatch slurm/rose1_ablation.sbatch
# Logs go to logs/ (an empty tracked folder, so it exists after git pull).
# Conda path and project directory are specific to the egottfri Oscar account.
#SBATCH -J rose1_ablation
#SBATCH -p gpu
#SBATCH --gres=gpu:1           # one GPU
#SBATCH -n 4                   # CPU cores
#SBATCH --mem=32G
#SBATCH -t 08:00:00            # generous; tighten using the run log in docs/RESULTS.md
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edward_gottfried@brown.edu

set -eo pipefail
export PYTHONUNBUFFERED=1
source /users/egottfri/miniconda3/etc/profile.d/conda.sh
conda activate octa-seg
cd /users/egottfri/code/octa-segmentation
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) job=${SLURM_JOB_ID}"
python scripts/rose1_ablation.py
