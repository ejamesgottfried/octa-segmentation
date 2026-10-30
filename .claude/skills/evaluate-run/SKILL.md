---
name: evaluate-run
description: Turn trained fold checkpoints into a per-image metrics CSV and summary table on the held-out test set. Use when James asks to evaluate an experiment, score checkpoints, compare variants on the test set, get AUPRC, or make results for a table or poster.
argument-hint: "[experiment-name]"
---

# Evaluate an experiment on its test set

One script, `scripts/evaluate_experiment.py <experiment>`, evaluates every finished
config under `results/<experiment>/`. If it does not exist yet, build it to this spec
(it replaces copy-pasted evaluation notebook cells).

## What it must do

1. For each config folder with `fold_val_dice.json`, read `config.json` to get the
   dataset name, then `dataset.get_split(name)` to rebuild the SAME test set.
2. For each `best_model_fold{k}.pth`: rebuild with `build_model(ckpt["model_name"],
   **ckpt["model_kwargs"])`, `model.eval()`, `torch.no_grad()`, sigmoid on logits.
3. Per test image, compute at threshold 0.5:
   dice, iou, sensitivity, specificity, precision (from `src/evaluate.py`), plus
   `ap = sklearn.metrics.average_precision_score(gt, prob)` (the AUPRC to report).
   Keep the old trapezoid `auprc` column too, named `auprc_trapz`, for continuity.
4. Also score the **fold ensemble**: average the k probability maps, then threshold.
   Write these rows with `fold = "ens"`. This is the candidate "deployed model" for
   ARIAS transfer and answers "which checkpoint do you use?".
5. Guard degenerate images: if `evaluate()` raises (empty prediction or empty GT),
   record NaN for the affected metrics and set `note` to the error text. Never drop rows silently.

## Output (long format, one row per image per fold per config)

`results/<experiment>/per_image.csv` and a copy in `reports/<experiment>/per_image.csv`:

| column | example |
|---|---|
| experiment | rose1_seeds |
| config_id | dataset-rose1_raw__model-res_unet__seed-1 |
| dataset, model, seed, (any other GRID keys) | rose1_raw, res_unet, 1 |
| fold | 1 .. k, or ens |
| image_id | filename stem, e.g. `12` |
| dice, iou, sensitivity, specificity, precision, ap, auprc_trapz | floats |
| note | empty unless something went wrong |
| git_commit | from config.json |

Plus `reports/<experiment>/summary.csv`: per config, mean and SD across folds of the
per-fold test means (this is the old "mean ± std across folds" table), and the ensemble row.

## Rules

- Never ARIAS through this script. ARIAS has no ground truth and has its own pipeline.
- Do not choose the threshold on the test set. 0.5 unless a threshold was chosen on
  validation folds and saved in config.json.
- Print the number of test images found and assert it matches config.json `n_test`.
- When reporting numbers to James, say which unit the SD is over (folds, seeds, images).
- Next step is usually `stats-compare` on `per_image.csv`.
