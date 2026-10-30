#!/bin/bash
# SLURM job: runs the full OCTA-500 6mm large-vessel ablation
# (scripts/octa500_lv_ablation.py) on one GPU.
# Submit from the repo root:   sbatch slurm/octa500_lv_ablation.sbatch
# Logs go to logs/ (an empty tracked folder, so it exists after git pull).
# Conda path and project directory are specific to the egottfri Oscar account.
#SBATCH -J octa500_lv
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH -t 24:00:00
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
python scripts/octa500_lv_ablation.py
