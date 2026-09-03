# OCTA Vessel Segmentation

Benchmarking deep learning and classical computer vision methods for retinal
vessel segmentation in **OCTA** (Optical Coherence Tomography Angiography)
images.

OCTA is a non-invasive imaging modality that visualizes retinal
microvasculature without injected dye. Automatically segmenting vessels from
OCTA scans is a step toward quantitative biomarkers for diseases such as
diabetic retinopathy and Alzheimer's disease. This project compares a plain
U-Net against several lightweight architectural additions (attention gates,
residual blocks, squeeze-and-excitation) and a classical adaptive-thresholding
baseline, evaluated with k-fold cross-validation across two public datasets.

## Datasets

| Dataset | FOV | Resolution | Format | Subjects | Notes |
|---|---|---|---|---|---|
| [ROSE-1](https://imed.nimte.ac.cn/dataofrose.html) | 3×3mm | 304×304 | `.tif` | 39 (26 AD, 13 healthy) | SVC / DVC / SVC+DVC layers |
| [OCTA-500](https://ieee-dataport.org/open-access/octa-500) | 3×3mm & 6×6mm | 304×304 / 400×400 | `.bmp` | 200–500 | capillary & large-vessel ground truth |

Full field/label layout notes (which subfolder maps to which layer, ID
ranges, etc.) are in [notes.md](notes.md). These datasets are not distributed
in this repo — see notes.md for where they were sourced from, and update the
paths described below to point at your own copy.

> **Note on paths:** this project was developed on Brown University's Oscar
> HPC cluster, so several scripts default to absolute filesystem paths (e.g.
> `/files22_lrsresearch/...`, `/users/egottfri/...`). Search for these paths
> (`ROSE1_BASE`, `PREP_BASE`, `RESULTS_DIR`, and the dataset paths in
> [src/dataset.py](src/dataset.py)) and update them to your own data/output
> locations before running anything.

## Repo layout

```
src/
  dataset.py             PyTorch Dataset, augmentations, dataset-specific train/test/k-fold splits
  model.py                FlexUNet: a U-Net with toggleable attention/residual/SE blocks, plus a name -> config registry
  train.py                 Losses (Dice, BCE+Dice), train/validate loop, k-fold training driver
  evaluate.py              Segmentation metrics: Dice, IoU, sensitivity, specificity, precision, AUPRC
  preprocess.py            CLAHE + median-filter preprocessing, applied once and cached to disk
  classical_baseline.py    Non-learned baseline: Otsu/adaptive thresholding + morphological cleanup
  plotting.py              Training-curve plotting
  test_model.py            Loads saved fold checkpoints and evaluates them on a held-out test set
scripts/
  rose1_ablation.py         Ablation sweep: 5 FlexUNet variants x {raw, preprocessed} on ROSE-1 SVC
  octa500_lv_ablation.py    Ablation sweep: 5 FlexUNet variants on OCTA-500 6mm large-vessel GT
  *.sbatch                   SLURM job scripts that run the two ablation scripts on a GPU node
notebooks/
  01_explore_rose1.ipynb     Exploratory data analysis of the ROSE-1 dataset
notes.md                     Raw notes on dataset paths, folder layouts, and label conventions
environment.yml / requirements.txt   Reproducible Python environment
```

`data/` and `results/` are empty (gitignored) placeholders for local dataset
copies and training outputs — everything under them is machine-specific and
not checked in.

## Method

**Preprocessing** ([src/preprocess.py](src/preprocess.py)): CLAHE (contrast-limited
adaptive histogram equalization) followed by a median filter, to see whether
enhancing local contrast before training helps segmentation.

**Model** ([src/model.py](src/model.py)): `FlexUNet` is a standard
encoder-decoder U-Net where three architectural additions can each be toggled
independently:
- `attention` — additive attention gates on the skip connections ([Oktay et al., 2018](https://arxiv.org/abs/1804.03999))
- `residual` — a residual shortcut inside each conv block
- `se` — squeeze-and-excitation channel attention inside each conv block ([Hu et al., 2018](https://arxiv.org/abs/1709.01507))

With all three off, `FlexUNet` is architecturally identical to a plain U-Net
([Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)) — this makes the
ablation clean, since every variant shares the same backbone and only one
component changes at a time. `MODEL_REGISTRY` in model.py names the five
variants actually swept: `unet`, `attention_unet`, `res_unet`, `se_unet`,
`att_res_unet`.

**Training** ([src/train.py](src/train.py)): a combined Dice + BCE loss (BCE for
stable per-pixel gradients, Dice to counter the class imbalance from thin
vessels being a small fraction of pixels), Adam, early stopping on validation
Dice, and 5-fold cross-validation via [src/dataset.py](src/dataset.py) (the
dataset's own train/test split is preserved; folds are drawn from the
training portion only, so the test set never leaks into model selection).

**Baseline** ([src/classical_baseline.py](src/classical_baseline.py)): Otsu or
adaptive Gaussian thresholding with optional CLAHE/median preprocessing and
morphological closing, with hyperparameters selected via cross-validated grid
search — a non-learned reference point for how much the U-Nets actually buy
you.

**Evaluation** ([src/evaluate.py](src/evaluate.py)): Dice, IoU, sensitivity
(recall), specificity, precision, and AUPRC (area under the precision-recall
curve, computed from the continuous probability map before thresholding).

## Setup

```bash
conda env create -f environment.yml
conda activate octa-seg
```

Alternatively, with an existing Python 3.11 environment:

```bash
pip install -r requirements.txt
```

## Usage

All commands assume you're running from the repo root, and that you've
updated the dataset paths as described above.

**1. (Optional) preprocess images:**

```bash
python src/preprocess.py
```

**2. Train one model on one dataset** (edit `dataset_name` / `model_name` at
the bottom of `train.py` and run `python src/train.py`, or import `run_kfold`
directly — modules under `src/` import each other by flat name, e.g.
`train.py` does `from model import build_model`, so `src/` itself must be on
`sys.path`, not just the repo root):

```python
import sys; sys.path.append("src")
from train import run_kfold
from dataset import get_octa500_split

train_imgs, train_masks, test_imgs, test_masks = get_octa500_split()
run_kfold(train_imgs, train_masks, test_imgs, test_masks,
          model_name="attention_unet", n_splits=5, device="cuda",
          output_dir="results/attention_unet_octa500")
```

**3. Run a full architecture ablation** (all 5 variants, resumable —
already-finished configs are skipped):

```bash
python scripts/rose1_ablation.py          # ROSE-1 SVC, raw + preprocessed
python scripts/octa500_lv_ablation.py     # OCTA-500 6mm large-vessel
```

On a SLURM cluster, submit the equivalent batch jobs instead:

```bash
sbatch scripts/rose1_ablation.sbatch
sbatch scripts/octa500_lv_ablation.sbatch
```

**4. Evaluate saved checkpoints on the held-out test set:**

```python
import sys; sys.path.append("src")
from test_model import evaluate_on_test

evaluate_on_test(test_imgs, test_masks,
                  model_paths=["results/unet_octa500/best_model_fold1.pth", ...],
                  device="cuda")
```

Each `run_kfold` call writes, per fold, a checkpoint (`best_model_fold*.pth`,
containing the model name + kwargs needed to rebuild it) and a training-curve
plot to `output_dir`.

## Results

Ablation results land under `results/<condition>/<model_name>/` as
per-fold checkpoints and training curves; aggregate metrics are printed to
stdout by `run_kfold` (mean ± std Dice across folds) and `test_model.py`
(mean ± std across folds on the test set, for every metric in
`evaluate.py`). Nothing is auto-aggregated into a single results table yet —
that step is manual.
