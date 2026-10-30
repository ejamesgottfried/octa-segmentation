#!/bin/bash
# EDIT: one line on what this runs. One array task = one config.
# Check the array range matches:   python scripts/EDIT_NAME.py --list-tasks
# Submit from the repo root, after:  mkdir -p logs
#   Test one task first:  sbatch --array=0 slurm/EDIT_NAME.sbatch
#   Then the rest:        sbatch slurm/EDIT_NAME.sbatch
#SBATCH -J EDIT_NAME
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --array=0-EDIT_LAST%8
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=edward_gottfried@brown.edu

set -eo pipefail
export PYTHONUNBUFFERED=1
source /users/egottfri/miniconda3/etc/profile.d/conda.sh
conda activate octa-seg
cd /users/egottfri/code/octa-segmentation

echo "host=$(hostname) commit=$(git rev-parse --short HEAD) job=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/EDIT_NAME.py --task-id "${SLURM_ARRAY_TASK_ID}"
