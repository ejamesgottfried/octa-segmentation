"""Architecture ablation on ROSE-1 SVC: trains all 5 FlexUNet variants
(see src/model.py's MODEL_REGISTRY) on both raw and CLAHE+median-preprocessed
images, 5-fold CV each. Resumable: a (condition, variant) combo whose 5 fold
checkpoints already exist is skipped, so a killed/requeued job picks up where
it left off. Intended to run via rose1_ablation.sbatch on a SLURM cluster,
or directly with `python scripts/rose1_ablation.py`.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import torch
import cv2
from train import run_variant_sweep

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
N_FOLDS    = 5
VARIANTS   = ["unet", "attention_unet", "res_unet", "se_unet", "att_res_unet"]
CONDITIONS = ["raw", "preprocessed"]

# EDIT ME: point these at your own ROSE-1 copy / preprocessed output / results dir.
RESULTS_DIR = Path("/users/egottfri/code/octa-segmentation/results/rose1_ablation")
ROSE1_BASE = Path("/files22_lrsresearch/ENG_Lee-Lab_Shared/group/data/public/rose_dataset/ROSE-1")
PREP_BASE  = Path("/users/egottfri/code/octa-segmentation/data/preprocessed")

train_masks = sorted((ROSE1_BASE / "SVC/train/gt").glob("*.tif"))
test_masks  = sorted((ROSE1_BASE / "SVC/test/gt").glob("*.tif"))
IMAGES = {
    "raw": {
        "train": sorted((ROSE1_BASE / "SVC/train/img").glob("*.tif")),
        "test":  sorted((ROSE1_BASE / "SVC/test/img").glob("*.tif")),
    },
    "preprocessed": {
        "train": sorted((PREP_BASE / "SVC/train/img").glob("*.tif")),
        "test":  sorted((PREP_BASE / "SVC/test/img").glob("*.tif")),
    },
}

print(f"device: {DEVICE}", flush=True)
# Sanity-check the data before burning GPU time on it: same image/mask counts,
# and dimensions divisible by 16 (required by FlexUNet's 4 downsampling stages).
for cond in CONDITIONS:
    assert len(IMAGES[cond]["train"]) == len(train_masks), f"{cond} train count mismatch"
    assert len(IMAGES[cond]["test"])  == len(test_masks),  f"{cond} test count mismatch"
h, w = cv2.imread(str(IMAGES["raw"]["train"][0]), cv2.IMREAD_GRAYSCALE).shape
assert h % 16 == 0 and w % 16 == 0, f"image {h}x{w} not divisible by 16"
print(f"counts ok; image {h}x{w}", flush=True)

# Full sweep: {raw, preprocessed} x 5 model variants.
for condition in CONDITIONS:
    run_variant_sweep(VARIANTS, IMAGES[condition]["train"], train_masks,
                       IMAGES[condition]["test"], test_masks,
                       RESULTS_DIR / condition, n_splits=N_FOLDS, device=DEVICE,
                       label=condition)
print("all configs done", flush=True)