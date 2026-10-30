---
name: stats-compare
description: Run pre-specified paired statistical comparisons between segmentation variants (architectures, seeds, losses, preprocessing) from a per-image CSV, with effect sizes, CIs, and multiple-comparison correction. Use when James asks whether a difference is significant, wants a paired t-test, p-values, error bars, seed variance, or statistical language for a poster or paper.
argument-hint: "[path/to/per_image.csv] [baseline] [variants...]"
allowed-tools: Bash(python .claude/skills/stats-compare/scripts/paired_compare.py *)
---

# Paired comparisons, done the same way every time

## 0. Pre-register before looking

Before running, write the comparison into docs/RESULTS.md under
"Pre-registered comparisons" if it is not already there: dataset, metric, baseline,
variants, test, and which one is PRIMARY. Anything added after seeing results is
labelled exploratory. If the entry is missing and results already exist, say so plainly.

## 1. Run the script

```bash
python .claude/skills/stats-compare/scripts/paired_compare.py <per_image.csv> \
    --metric dice --baseline unet --compare res_unet attention_unet se_unet att_res_unet \
    --where dataset=rose1_raw --out reports/<experiment>/stats_dice_rose1_raw.csv
```
Options: `--group-col loss` for loss experiments, `--ens` to compare fold ensembles,
`--variance` for the fold vs seed spread (feedback item 2), `--metric ap` for AUPRC.
Run each dataset/condition separately with `--where`; do not pool raw and preprocessed.

## 2. How to read it (and how to explain it to James)

- **Unit of analysis = test image.** Each image's score is averaged over folds and seeds,
  then variants are compared image by image (paired). n = number of test images (9 for
  ROSE-1). This is the headline test.
- `mean_diff` with `ci95_low/high` (bootstrap over images) is the main result.
  A CI that excludes 0 and a Holm p < 0.05 together support "improved".
- `p_ttest_holm` is the primary p-value (Holm across every comparison in that call).
  `p_wilcoxon_holm` is a robustness check. With n = 9 the smallest possible two-sided
  Wilcoxon p is 0.0039, so it cannot survive heavy correction; say so if relevant.
- `cohens_dz` is the paired effect size. Very large values with tiny diffs mean the
  images agree consistently, not that the effect matters clinically. Report the Dice diff.
- `p_run_ttest_optimistic` pairs fold x seed runs. Folds overlap in training data, so it
  understates uncertainty. Never quote it as the headline.
- Variance mode: if `sd_across_seed_means` is near zero while folds vary, seed effects are
  small. Error bars on figures should use the SD over all runs (folds x seeds), stated.

## 3. Reporting template (no em dashes)

> Residual connections improved test Dice by [DIFF] (95% CI [LO] to [HI]; paired t-test
> across [N] test images, Holm-adjusted p = [P]; [S] seeds x [K] folds per variant).
> Attention gates and SE blocks did not differ from the baseline (adjusted p > [P]).

(Placeholders only. Fill from the script output; never from memory.)

Always include: n and unit, mean difference with CI, the test, the correction, seeds x folds.
If a result is from an exploratory comparison, the sentence says "exploratory".

## 4. Then

Add the table (or its path) to docs/RESULTS.md and run the `rigor-review` skill on the
sentence before it goes on a slide.

## Not for ARIAS group comparisons

ARIAS needs subject-clustered models (two eyes per subject) with covariates:
`feature ~ group + age + sex + quality`, eye nested in subject (mixed model or
subject-mean features). Build that separately when ARIAS labels arrive.
