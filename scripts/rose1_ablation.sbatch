#!/bin/bash
# SLURM job: runs the full ROSE-1 ablation (scripts/rose1_ablation.py) on one GPU.
# EDIT ME: conda path, project directory, and mail-user below are all
# specific to the original developer's cluster account — update them for yours.
#SBATCH -J rose1_ablation
#SBATCH -p gpu
#SBATCH --gres=gpu:1           # one GPU
#SBATCH -n 4                   # CPU cores
#SBATCH --mem=32G
#SBATCH -t 08:00:00            # PLACEHOLDER: generous; tighten after first run
#SBATCH -o rose1_ablation-%j.out
#SBATCH -e rose1_ablation-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edward_gottfried@brown.edu   # optional

source /users/egottfri/miniconda3/etc/profile.d/conda.sh
conda activate octa-seg
cd /users/egottfri/code/octa-segmentation
python scripts/rose1_ablation.py