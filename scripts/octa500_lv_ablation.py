"""Architecture ablation on OCTA-500's 6mm large-vessel ground truth: trains
all 5 FlexUNet variants (see src/model.py's MODEL_REGISTRY), 5-fold CV each.
Resumable: a variant whose 5 fold checkpoints already exist is skipped.
Intended to run via octa500_lv_ablation.sbatch on a SLURM cluster, or
directly with `python scripts/octa500_lv_ablation.py`.
"""

import os, glob, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import torch, cv2
from train import run_kfold
from dataset import get_octa500_largevessel_6mm

DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
N_FOLDS     = 5
VARIANTS    = ["unet", "attention_unet", "res_unet", "se_unet", "att_res_unet"]
# EDIT ME: point this at your own results directory.
RESULTS_DIR = Path("/users/egottfri/code/octa-segmentation/results/octa500_lv_ablation")

train_imgs, train_masks, test_imgs, test_masks = get_octa500_largevessel_6mm()
print(f"device: {DEVICE}", flush=True)
print(f"train: {len(train_imgs)}  test: {len(test_imgs)}", flush=True)
# Sanity-check the data before burning GPU time on it: same image/mask counts,
# files actually exist, and dimensions divisible by 16 (FlexUNet's 4 downsampling stages).
assert len(train_imgs)==len(train_masks) and len(test_imgs)==len(test_masks)
assert Path(train_imgs[0]).exists(), f"missing {train_imgs[0]}"
assert Path(train_masks[0]).exists(), f"missing {train_masks[0]}"
h,w = cv2.imread(str(train_imgs[0]), cv2.IMREAD_GRAYSCALE).shape
assert h%16==0 and w%16==0, f"{h}x{w} not divisible by 16"
print(f"paths ok; image {h}x{w}", flush=True)

for name in VARIANTS:
    out = RESULTS_DIR / name
    if len(glob.glob(str(out/"best_model_fold*.pth"))) >= N_FOLDS:
        print(f"[skip]  {name}", flush=True); continue
    print(f"[train] {name}", flush=True)
    run_kfold(train_imgs, train_masks, test_imgs, test_masks,
              model_name=name, n_splits=N_FOLDS, device=DEVICE, output_dir=str(out))
print("all done", flush=True)