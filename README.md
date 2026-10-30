# OCTA Vessel Segmentation

Benchmarking deep learning and classical computer vision methods for retinal vessel
segmentation in **OCTA** (Optical Coherence Tomography Angiography) images.

OCTA is a non-invasive imaging modality that visualizes retinal microvasculature without
injected dye. Automatically segmenting vessels from OCTA scans is a step toward quantitative
biomarkers for diseases such as diabetic retinopathy and Alzheimer's disease. This project
compares a plain U-Net against several lightweight architectural additions (attention gates,
residual blocks, squeeze-and-excitation) and a classical adaptive-thresholding baseline,
evaluated with k-fold cross-validation across two public datasets. Lee Lab, Brown University.

## Datasets

| Dataset | FOV | Resolution | Format | Subjects | Notes |
|---|---|---|---|---|---|
| [ROSE-1](https://imed.nimte.ac.cn/dataofrose.html) | 3×3 mm | 304×304 | `.tif` | 39 (26 AD, 13 healthy) | SVC / DVC / SVC+DVC layers |
| [OCTA-500](https://ieee-dataport.org/open-access/octa-500) | 3×3 mm and 6×6 mm | 304×304 / 400×400 | `.bmp` | 200 to 500 | capillary and large-vessel ground truth |

Curated dataset notes (layouts, ID ranges, label provenance) are in
[docs/DATA.md](docs/DATA.md); the original working notes are in [docs/notes.md](docs/notes.md).
The datasets are not distributed in this repo.

> **Note on paths:** this project was developed on Brown University's Oscar HPC cluster, so
> several scripts default to absolute paths (`/files22_lrsresearch/...`, `/users/egottfri/...`).
> Search for `ROSE1_BASE`, `PREP_BASE`, `RESULTS_DIR`, and the dataset paths in
> [src/dataset.py](src/dataset.py) and point them at your own copies before running anything.

## Repo layout

```
src/
  dataset.py             PyTorch Dataset, augmentations, dataset-specific train/test/k-fold splits
  model.py               FlexUNet: a U-Net with toggleable attention/residual/SE blocks + registry
  train.py               Losses (Dice, BCE+Dice), train/validate loop, k-fold training driver
  evaluate.py            Metrics: Dice, IoU, sensitivity, specificity, precision, AUPRC
  preprocess.py          CLAHE + median-filter preprocessing, applied once and cached to disk
  classical_baseline.py  Non-learned baseline: Otsu/adaptive thresholding + morphological cleanup
  plotting.py            Training-curve plotting
  test_model.py          Loads saved fold checkpoints and evaluates them on a held-out test set
scripts/                 One experiment per file (rose1_ablation.py, octa500_lv_ablation.py, ...)
slurm/                   One SLURM .sbatch file per script
tests/                   CPU smoke tests on synthetic data (python -m pytest tests/ -q)
notebooks/               datasets/, preprocessing/, training/, evaluating/, figures/ (thin, analysis only)
reports/                 Small result CSVs and final figures that results are cited from
docs/                    DATA.md, RESULTS.md (every reported number), ROADMAP.md (planned work)
logs/                    SLURM output (contents gitignored)
data/, results/          Local data copies and training outputs (mostly gitignored)
private/                 Gitignored; never committed
CLAUDE.md, .claude/      Instructions, rules, and skills for Claude Code
```

## Method

**Preprocessing** ([src/preprocess.py](src/preprocess.py)): CLAHE (contrast-limited adaptive
histogram equalization) followed by a median filter, to test whether enhancing local contrast
before training helps segmentation.

**Model** ([src/model.py](src/model.py)): `FlexUNet` (paper name TogUNet) is a standard
encoder-decoder U-Net where three additions can each be toggled independently:
- `attention`: additive attention gates on the skip connections ([Oktay et al., 2018](https://arxiv.org/abs/1804.03999))
- `residual`: a residual shortcut inside each conv block
- `se`: squeeze-and-excitation channel attention inside each conv block ([Hu et al., 2018](https://arxiv.org/abs/1709.01507))

With all three off, `FlexUNet` is architecturally identical to a plain U-Net
([Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)), so every variant shares the
same backbone and only one component changes at a time. `MODEL_REGISTRY` names the five
variants swept so far: `unet`, `attention_unet`, `res_unet`, `se_unet`, `att_res_unet`.

**Training** ([src/train.py](src/train.py)): combined Dice + BCE loss (BCE for stable per-pixel
gradients, Dice for the class imbalance of thin vessels), Adam, early stopping on validation
Dice, and 5-fold cross-validation. The dataset's own train/test split is preserved; folds are
drawn from the training portion only, so the test set never influences model selection.

**Baseline** ([src/classical_baseline.py](src/classical_baseline.py)): Otsu or adaptive Gaussian
thresholding with optional CLAHE/median preprocessing and morphological closing, with
hyperparameters selected by cross-validated grid search on the training images.

**Evaluation** ([src/evaluate.py](src/evaluate.py)): Dice, IoU, sensitivity (recall),
specificity, precision, and AUPRC (from the continuous probability map before thresholding).

## Setup

```bash
conda env create -f environment.yml
conda activate octa-seg
python -m pytest tests/ -q        # about a minute on a laptop CPU
```
Or, in an existing Python 3.11 environment: `pip install -r requirements.txt pytest`.

## Usage

All commands run from the repo root, after updating the dataset paths.

**1. (Optional) preprocess images:** `python src/preprocess.py`

**2. Train one model on one dataset.** Modules under `src/` import each other by flat name
(`train.py` does `from model import build_model`), so `src/` itself must be on `sys.path`:

```python
import sys; sys.path.append("src")
from train import run_kfold
from dataset import get_octa500_split

train_imgs, train_masks, test_imgs, test_masks = get_octa500_split()
run_kfold(train_imgs, train_masks, test_imgs, test_masks,
          model_name="attention_unet", n_splits=5, device="cuda",
          output_dir="results/attention_unet_octa500")
```

**3. Run a full architecture ablation** (all 5 variants; resumable, finished configs are skipped):

```bash
python scripts/rose1_ablation.py          # ROSE-1 SVC, raw + preprocessed
python scripts/octa500_lv_ablation.py     # OCTA-500 6mm large-vessel
# on a SLURM cluster:
sbatch slurm/rose1_ablation.sbatch
sbatch slurm/octa500_lv_ablation.sbatch
```

**4. Evaluate saved checkpoints on the held-out test set:**

```python
import sys; sys.path.append("src")
from test_model import evaluate_on_test

evaluate_on_test(test_imgs, test_masks,
                 model_paths=["results/unet_octa500/best_model_fold1.pth", ...],
                 device="cuda")
```

Each `run_kfold` call writes, per fold, a checkpoint (`best_model_fold*.pth`, with the model
name and kwargs needed to rebuild it) and a training-curve plot to `output_dir`.

## Results

**ROSE-1 SVC, held-out test set** (best classical config vs. plain FlexUNet):

| Metric | Classical (raw) | U-Net (raw) | Change |
|---|---|---|---|
| Dice | 0.6330 ± 0.0222 | 0.7741 ± 0.0018 | +0.1411 |
| IoU | 0.4635 ± 0.0238 | 0.6341 ± 0.0023 | +0.1706 |
| Sensitivity | 0.7045 ± 0.0561 | 0.7610 ± 0.0109 | +0.0565 |
| Specificity | 0.8854 ± 0.0166 | 0.9545 ± 0.0032 | +0.0691 |
| Precision | 0.5795 ± 0.0396 | 0.7926 ± 0.0094 | +0.2131 |
| AUPRC | n/a | 0.8658 ± 0.0018 | n/a |

Classical SD is over the 9 test images; U-Net SD is over the 5 fold models. The full
per-variant breakdown (5 FlexUNet configs × raw/preprocessed) is in
[reports/rose1_ablation/summary.csv](reports/rose1_ablation/summary.csv), and the status of
every finding, with its caveats, is in [docs/RESULTS.md](docs/RESULTS.md).

## Development workflow (laptop + Oscar + Claude Code)

```
 Laptop (VS Code + Claude Code)                 Oscar
 ------------------------------                 -----
 1. ask Claude for a script       ──git push──▶ 3. git pull
 2. pytest / --smoke on CPU                     4. sbatch (one test task, then the rest)
                                                5. evaluate into reports/<experiment>/
 7. git pull, stats + figures     ◀──git push── 6. commit reports/ CSVs
 8. rigor-review, update docs/RESULTS.md
```

Code moves laptop to Oscar. Small CSVs move Oscar to laptop. Checkpoints and raw data never move.
Anything that produces a number you will cite, or runs over about 10 minutes, is a script
submitted with sbatch. Jupyter (Open OnDemand) is for looking at predictions and quick plots.

Claude Code reads [CLAUDE.md](CLAUDE.md) every session. Project skills in `.claude/skills/`:

| Skill | Use it for | Example prompt |
|---|---|---|
| `/new-experiment` | a new training script + sbatch | "new experiment: 3 seeds of unet and res_unet on rose1_raw" |
| `/oscar-run` | commands to paste on Oscar, or a failed job | "here is my .err file, why did task 4 die?" |
| `/evaluate-run` | checkpoints to per-image CSV | "evaluate rose1_seeds" |
| `/stats-compare` | paired tests, CIs, seed variance | "is res_unet better than unet on raw?" |
| `/paper-figure` | figures in house style | "heatmap for the CLAHE grid search" |
| `/rigor-review` | audit a claim before a slide | "/rigor-review the abstract draft" |

Habits: one task per session and `/clear` between tasks; plan mode (Shift+Tab) for changes
touching several files; paste real error text and `sacct` output; when Claude repeats a
mistake, add one line to CLAUDE.md. Personal notes go in `CLAUDE.local.md` (gitignored).

**Oscar cheat sheet**

```bash
cd /users/egottfri/code/octa-segmentation && git pull
python scripts/<name>.py --list-tasks
sbatch --array=0 slurm/<name>.sbatch            # test one task
sbatch --array=1-<N-1>%8 slurm/<name>.sbatch    # the rest
squeue -u egottfri
sacct -j <JOBID> --format=JobID%20,State,Elapsed,MaxRSS,ExitCode
tail -f logs/<name>-<JOBID>_<TASK>.out
```

## Data privacy

The lab's private ARIAS cohort is human-subjects data. It is never committed to this public
repository. Only a de-identified subject table, kept locally in the gitignored `private/`
folder, is used for analysis planning.
