#!/bin/bash
# SLURM job: runs the full OCTA-500 large-vessel ablation
# (scripts/octa500_lv_ablation.py) on one GPU.
# EDIT ME: conda path, project directory, and mail-user below are all
# specific to the original developer's cluster account — update them for yours.
#SBATCH -J octa500_lv
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o octa500_lv-%j.out
#SBATCH -e octa500_lv-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_REAL_EMAIL   # fill in or delete this line

export PYTHONUNBUFFERED=1
source /users/egottfri/miniconda3/etc/profile.d/conda.sh
conda activate octa-seg
cd /users/egottfri/code/octa-segmentation
python scripts/octa500_lv_ablation.py