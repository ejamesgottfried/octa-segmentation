# Roadmap

Presentation feedback, mapped to concrete experiments, in the order to do them.
Each item has a copy-paste prompt for Claude Code. Update the status column as you go.

## Feedback received

| # | Feedback | Answered by |
|---|---|---|
| F1 | Paired t-test across folds: do residual connections significantly help? | E1 |
| F2 | Multiple random seeds (error bars should grow) | E4 |
| F3 | Report a heatmap for the grid search | E2 |
| F4 | Report all parameters that were grid searched | E2 |
| F5 | Nested k-fold for preprocessing parameters | E3, E7 |
| F6 | Cross-dataset generalization: overfit, or ground-truth differences? Train OCTA-500 test ROSE-1 and the reverse; ARIAS later | E5 |
| F7 | AUPRC may be a better metric than Dice | E1, E2 (reporting) |
| F8 | Try different loss functions | E6 |

## Experiment table

| ID | What | Cost | Needs | Status |
|---|---|---|---|---|
| P0 | Repo plumbing (below) | 1 to 2 sessions, no GPU | | partly done |
| A0 | De-identified ARIAS subject table (`scripts/deidentify_arias.py`, run by James) | 10 min | raw sheet | todo |
| E1 | Per-image paired stats + AP on existing ROSE-1 ablation checkpoints | ~30 GPU-min eval | P0 | todo |
| E2 | Grid-search heatmaps, full parameter table, median-kernel sweep | CPU only | P0 | todo |
| E3 | Nested CV for the classical baseline | CPU only, a few hours | P0 | todo |
| E4 | Seeds: unet + res_unet x 3 seeds x 5 folds, ROSE-1 raw; add 3 missing flag combos | ~45 fold-trainings | P0 | todo |
| E5 | Cross-dataset + label-swap matrix | ~6 configs x 5 folds | P0 | todo |
| E6 | Loss functions on res_unet, ROSE-1 raw, 3 seeds | ~5 losses x 15 runs | E4 | todo |
| E7 | U-Net preprocessing sensitivity (milder CLAHE settings) | ~3 configs x 15 runs | E4 | todo |
| E8 | ARIAS: deployed model, projection export, age-regression check, features | later | labels from jlee123 | todo |

Order rationale: P0 makes every later step cheap. E1 to E3 reuse existing checkpoints or
the CPU, so they turn feedback into results fastest. E4 sets the seed protocol that E6
and E7 reuse. E5 is the most scientifically interesting, so start it while E4 runs.

---

## P0. Plumbing (do first, one Claude Code session)

> Read CLAUDE.md and docs/ROADMAP.md. Do Phase 0 in small commits, showing me each diff:
> 1. Create `src/paths.py` with every dataset/results path now hardcoded in src/ and scripts/.
> 2. Add `get_split(name)` to `src/dataset.py` returning (train_imgs, train_masks, test_imgs,
>    test_masks) for: rose1_raw, rose1_prep, octa500_3mm_cap, octa500_3mm_lv, octa500_6mm_lv.
>    Move the ROSE-1 loading out of scripts/rose1_ablation.py. Keep existing functions working.
> 3. Add an optional `seed` argument to `run_kfold` and a `loss_name` argument backed by a
>    `LOSS_REGISTRY` (default "bce_dice" = current CombinedLoss). Defaults must reproduce old behavior.
> 4. Register the 3 missing flag combos in MODEL_REGISTRY: att_se_unet, res_se_unet, att_res_se_unet.
> 5. Build `scripts/evaluate_experiment.py` per the evaluate-run skill. Make it also work on
>    the old layout results/rose1_ablation/<condition>/<variant>/best_model_fold*.pth.
> 6. Add `tests/test_smoke.py` (pytest): build each registry model on a 64x64 tensor, run one
>    training step, run evaluate on a synthetic mask. Must pass on CPU.
> 7. (done 2026-10-19) sbatch files moved to slurm/, logs/ folder, .gitignore, reports/, tests/.
> 8. Set up nbstripout (`pip install nbstripout && nbstripout --install`) and clear outputs
>    from the notebooks in one separate commit, after checking with me which figures to keep.
> Then update CLAUDE.md to drop the "(planned)" tags that are now real.

Also check by hand on Oscar: which CLAHE tile size made `data/preprocessed/`
(`preprocess.py` main uses the default tile 8; the chosen setting is tile 4).
If it was 8, the raw-vs-preprocessed U-Net comparison used non-optimal preprocessing,
which is worth knowing before E7.

## E1. Paired statistics and AUPRC on what you already trained (F1, F7)

> Use the evaluate-run skill to give me a run card that evaluates results/rose1_ablation
> (both conditions, all 5 variants) into reports/rose1_ablation/per_image.csv. After I push
> the CSV, use stats-compare with the primary comparison res_unet vs unet on rose1_raw Dice,
> and the other variants as secondary, Holm-corrected. Also run --metric ap.

Pre-register before running (copy into RESULTS.md): primary = res_unet vs unet, ROSE-1 raw,
Dice, per-image paired t-test, Holm over the 4 variants vs unet.
Note: the September fold-level p-values (n = 5 folds, unadjusted) are superseded by this.

## E2. Grid search reporting (F3, F4, F7)

> Use paper-figure to make heatmaps from the two ROSE-1 grid-search CSVs (CLAHE clip x tile;
> adaptive block x offset), with the best cell outlined, and a table of every parameter and
> value searched, the scorer, and the CV scheme. Then write scripts/median_kernel_sweep.py
> (CPU) since the median kernel was never swept, and add it to the table.

## E3. Nested CV for the classical baseline (F5)

The classical baseline is the model whose numbers depend on tuned parameters, and it runs on
CPU, so do a proper nested CV: outer 5 folds over the 30 training images, inner CV grid search
per outer fold, score the outer fold with the inner winner. Report the outer-fold mean and
how often each parameter combination won. Final test score uses params tuned on all 30.

> Write scripts/classical_nested_cv.py using classical_baseline.cv_grid_search for the inner
> loop. Same grids as before. Output per-outer-fold winners and scores to reports/.

## E4. Seeds and the full flag grid (F2, ablation gap)

> Use new-experiment to make scripts/rose1_seeds.py: dataset rose1_raw; models unet, res_unet,
> att_se_unet, res_se_unet, att_res_se_unet; seeds 0, 1, 2. Then a run card.

Then evaluate-run, then `stats-compare --variance` (do error bars grow?) and the paired test
including seeds. Optionally add attention_unet, se_unet, att_res_unet seeds 1 and 2 so all
8 combos have 3 seeds.

## E5. Cross-dataset and label-swap (F6)

The key design point: OCTA-500 3 mm images have BOTH GT_LargeVessel (manual) and
GT_Capillary (model-generated) on the same ILM_OPL projection, and ROSE-1 SVC is the same
device family and the same 3 mm field. So:

- **Label swap, zero domain shift:** train on OCTA-500 3 mm LV, test on OCTA-500 3 mm Cap
  and vice versa. Any drop here is purely a labeling-convention effect.
- **Domain shift, matched labels as far as possible:** train ROSE-1 SVC, test OCTA-500 3 mm
  against LV and against Cap; train OCTA-500 3 mm LV, test ROSE-1.
- **Diagnostic:** compare predicted vessel density to each label's density (ROSE-1 ~23%,
  LV ~8%, Cap ~48%). A model that is right about vessels but wrong about labeling convention
  shows a density mismatch, not random errors.
- Overfitting check: in-domain val vs in-domain test gap per training set.

If ROSE-1 to OCTA-500 is poor against Cap but reasonable against LV, and the label-swap drop
is similar in size, the answer to F6 is "ground-truth differences", with evidence.

> Use new-experiment for scripts/cross_dataset.py: train res_unet (3 seeds) on rose1_raw,
> octa500_3mm_lv, octa500_3mm_cap; then extend evaluate_experiment.py with a --test-on option
> so each trained config is scored on every other test set. Output a train x test matrix CSV.

## E6. Loss functions (F8)

Candidates: bce_dice (current), dice, bce, focal, tversky (alpha 0.3, beta 0.7; favors recall,
and sensitivity is currently the weak metric), cldice (topology-aware, suits thin vessels).
Use res_unet, rose1_raw, 3 seeds. Primary metric pre-registered; also report sensitivity
and AP since the motivation is thin-vessel recall.

## E7. U-Net preprocessing sensitivity (F5, U-Net side)

Full nested CV with U-Net training is expensive. Cheaper and still answers the critique:
retrain U-Net on 2 or 3 milder preprocessing settings (e.g. clip 1.0 tile 4; clip 2.0 tile 8)
with the E4 seed protocol. If none beat raw, "preprocessing does not help the U-Net" holds
regardless of which parameters were chosen. State this reasoning in the methods.

## E8. ARIAS (later)

Deployed model = fold ensemble of the OCTA-500 6 mm large-vessel res_unet (answers
"which checkpoint"). Then: batch projection export with a de-identified manifest, vessel
density and fractal dimension features, age-regression sanity check, then pre-specified
group comparison with eyes clustered in subjects. All ARIAS outputs stay on Oscar.
