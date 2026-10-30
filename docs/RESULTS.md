# Results

The single source of truth for numbers. Every number on a slide or poster must trace to a
line here, and every line here to a CSV in `reports/` or a SLURM job. Newest first.

## Pre-registered comparisons

Write these BEFORE running the analysis. Anything not listed here is exploratory.

| ID | Dataset | Metric | Baseline | Variants | Test | Primary? | Registered |
|---|---|---|---|---|---|---|---|
| R1 | ROSE-1 raw | Dice | unet | res_unet | per-image paired t, Holm over 4 variants | yes | (date) |
| R2 | ROSE-1 raw | Dice | unet | attention_unet, se_unet, att_res_unet | same family as R1 | no | (date) |

## Findings to date

Status key: **solid** (supported, caveats stated), **provisional** (needs E-item), **superseded**.

### ROSE-1 SVC, 9 test images, mean ± SD across 5 fold models

| Method | Raw Dice | Preprocessed Dice | Status |
|---|---|---|---|
| Classical (adaptive threshold, CV-tuned) | 0.633 ± 0.022 | 0.671 ± 0.028 | provisional (E3 nested CV) |
| U-Net | 0.774 ± 0.002 | 0.769 ± 0.002 | provisional (seeds, E4) |

- Deep learning beats classical by about 0.10 to 0.14 Dice. Solid in direction.
- Preprocessing (CLAHE + median) helps classical, slightly hurts U-Net. Provisional:
  params tuned with the classical scorer; confirm which tile size produced the
  preprocessed folder; E7 tests milder settings.
- Both methods: high specificity, lower sensitivity (thin vessels missed).
- Classical SD is over test images; U-Net SD is over fold models. Not comparable.

### FlexUNet ablation (ROSE-1, 5 variants x raw/preprocessed x 5 folds)

Test set (9 images), mean ± SD across the 5 fold models. Source: `reports/rose1_ablation/summary.csv`.

| Condition | Model | Dice | Sensitivity | Precision | AUPRC (trapz) |
|---|---|---|---|---|---|
| raw | unet | 0.7744 ± 0.0023 | 0.7587 ± 0.0110 | 0.7962 ± 0.0158 | 0.8658 ± 0.0009 |
| raw | attention_unet | 0.7732 ± 0.0021 | 0.7626 ± 0.0090 | 0.7888 ± 0.0128 | 0.8647 ± 0.0017 |
| raw | res_unet | 0.7802 ± 0.0008 | 0.7411 ± 0.0088 | 0.8287 ± 0.0100 | 0.8718 ± 0.0007 |
| raw | se_unet | 0.7720 ± 0.0041 | 0.7586 ± 0.0117 | 0.7915 ± 0.0182 | 0.8612 ± 0.0018 |
| raw | att_res_unet | 0.7799 ± 0.0018 | 0.7425 ± 0.0053 | 0.8261 ± 0.0062 | 0.8712 ± 0.0015 |
| preprocessed | unet | 0.7704 ± 0.0011 | 0.7498 ± 0.0097 | 0.7980 ± 0.0125 | 0.8618 ± 0.0020 |
| preprocessed | attention_unet | 0.7680 ± 0.0019 | 0.7624 ± 0.0087 | 0.7789 ± 0.0116 | 0.8615 ± 0.0020 |
| preprocessed | res_unet | 0.7758 ± 0.0009 | 0.7405 ± 0.0019 | 0.8199 ± 0.0040 | 0.8702 ± 0.0006 |
| preprocessed | se_unet | 0.7672 ± 0.0024 | 0.7652 ± 0.0047 | 0.7747 ± 0.0094 | 0.8603 ± 0.0013 |
| preprocessed | att_res_unet | 0.7737 ± 0.0017 | 0.7428 ± 0.0063 | 0.8126 ± 0.0071 | 0.8667 ± 0.0019 |

- Trade-off worth reporting: residual variants gain precision (+0.02 to +0.03 vs unet) and
  lose sensitivity (-0.01 to -0.02). Their Dice gain comes from fewer false positives, not
  from finding more vessels. Relevant to the loss-function experiment (E6).

- Residual connections: about +0.006 Dice over plain U-Net. Attention and SE: no clear effect.
- September fold-level paired t-tests (n = 5 folds, **unadjusted**): res_unet p = 0.016 (raw),
  0.002 (prep); att_res_unet p = 0.018 (raw), 0.063 (prep); attention and SE not different.
  **Superseded by E1** (per-image unit, Holm). Do not use "significant" until E1 is done.
- Null architecture effect matches OCTA-500 literature (U-Net 90.63, UNet++ 90.62,
  Attention U-Net 90.59 on capillaries, Li et al.).
- Only 5 of 8 flag combinations tested (E4 fills the gap).

### Grid search (ROSE-1, classical scorer, 5-fold CV on training images)

- CLAHE: best clip 2.0, tile 4, CV Dice 0.678.
- Adaptive threshold: best block 101, offset -30, CV Dice 0.679.
- Median kernel never swept (E2).

### Cross-dataset and ARIAS

- ROSE-1 to OCTA-500 transfer is poor; attributed to label granularity (GT_Capillary is
  model-generated), not domain shift. Provisional until E5's label-swap experiment.
- OCTA-500 6 mm large-vessel res_unet: Dice 0.88 in domain (SLURM 4380271).
  Transfers to ARIAS without fine-tuning (qualitative: arcades traced continuously).
- ROSE-1 model over-estimates vessel width on ARIAS (3 mm to 6 mm caliber mismatch).

## Run log (for setting SLURM time limits)

| Date | Experiment | Config | GPU | Minutes per fold | Notes |
|---|---|---|---|---|---|
| | | | | | |
