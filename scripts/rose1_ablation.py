import os, glob, sys
sys.path.append('/users/egottfri/code/octa-segmentation/src')

import torch
import cv2
from pathlib import Path
from train import run_kfold

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
N_FOLDS    = 5
VARIANTS   = ["unet", "attention_unet", "res_unet", "se_unet", "att_res_unet"]
CONDITIONS = ["raw", "preprocessed"]
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
for cond in CONDITIONS:
    assert len(IMAGES[cond]["train"]) == len(train_masks), f"{cond} train count mismatch"
    assert len(IMAGES[cond]["test"])  == len(test_masks),  f"{cond} test count mismatch"
h, w = cv2.imread(str(IMAGES["raw"]["train"][0]), cv2.IMREAD_GRAYSCALE).shape
assert h % 16 == 0 and w % 16 == 0, f"image {h}x{w} not divisible by 16"
print(f"counts ok; image {h}x{w}", flush=True)

for condition in CONDITIONS:
    tr_imgs = IMAGES[condition]["train"]
    te_imgs = IMAGES[condition]["test"]
    for name in VARIANTS:
        out_dir = RESULTS_DIR / condition / name
        done = len(glob.glob(str(out_dir / "best_model_fold*.pth")))
        if done >= N_FOLDS:
            print(f"[skip]  {condition} {name} ({done} folds)", flush=True)
            continue
        print(f"[train] {condition} / {name}", flush=True)
        run_kfold(tr_imgs, train_masks, te_imgs, test_masks,
                  model_name=name, n_splits=N_FOLDS, device=DEVICE,
                  output_dir=str(out_dir))
print("all configs done", flush=True)