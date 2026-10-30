# octa-segmentation

Retinal OCTA vessel segmentation toward Alzheimer's/MCI biomarkers. Lee Lab, Brown.
Owner: James Gottfried (undergrad, beginner programmer, strong on methodology).
Details live in docs/: @docs/DATA.md for datasets, docs/RESULTS.md for findings,
docs/ROADMAP.md for planned work. Read RESULTS.md before writing any claim.

## How we work

- Claude Code runs on James's Mac (VS Code). Code changes go: edit here, commit, `git push`,
  then James runs `git pull` on Oscar and submits with `sbatch`.
- The laptop has no raw datasets, no checkpoints (.pth), and no GPU. It does have
  `data/preprocessed/SVC/` (tracked ROSE-1 preprocessed images) and `results/` training-curve
  PNGs plus `results/rose1_ablation/summary.csv`. Do not train locally. Verify code with
  `python -m pytest tests/ -q` and `--smoke` runs on synthetic data, or by reasoning.
- Items marked (planned) below do not exist yet; ROADMAP Phase 0 creates them.
  If something here disagrees with the actual code, trust the code and tell James.
- Results come back as small CSVs/figures committed under `reports/` from Oscar.
- When a task needs Oscar, end with a "Run card": the exact commands James pastes on Oscar.
- James is learning. Explain non-obvious code in short comments and in your reply.
  Prefer small, reviewable diffs over big rewrites. Ask before restructuring a module.

## Repo layout

- `src/` library code. Notebooks and scripts import from here.
  - `model.py` FlexUNet (+ `MODEL_REGISTRY`, `build_model`). Paper name: TogUNet.
  - `train.py` losses, `train_fold`, `run_kfold`, `run_variant_sweep`.
  - `dataset.py` `OCTADataset`, k-fold helpers, dataset split loaders.
  - `evaluate.py` per-image metrics. `test_model.py` evaluates saved fold checkpoints.
  - `preprocess.py`, `classical_baseline.py`, `plotting.py`.
- `src/paths.py` (planned) single place for dataset/results paths.
- `scripts/` one experiment per file, runnable via `python scripts/X.py`.
- `slurm/` sbatch files, one per script. Logs go to `logs/` (tracked empty folder, contents ignored).
- `tests/` CPU smoke tests on synthetic data (`python -m pytest tests/ -q`).
- `notebooks/{datasets,preprocessing,training,evaluating,figures}/` analysis only, thin.
  (An older untracked `notebooks/evalutating/` with a typo may still exist on Oscar.)
- `results/` checkpoints + raw outputs. Checkpoints exist only on Oscar.
- `reports/` tracked small CSV summaries and final figures (this is what gets cited).
- `private/` gitignored. Only `private/arias_subjects_deid.csv` may be read (see below).
- `docs/notes.md` James's raw dataset notes; `docs/DATA.md` is the curated version.

## Conventions (follow these)

- Every new script: argparse, `--smoke`, `--task-id` for SLURM arrays, resumable
  (skip configs whose checkpoints exist), writes `config.json` with git commit hash.
  Use the `new-experiment` skill.
- Never hardcode a dataset path in a script. Add a loader in `src/dataset.py`.
- Never change the test sets: ROSE-1 uses its official 30/9 split; OCTA-500 uses
  `train_test_split(..., random_state=42)`. Folds use `KFold(random_state=42)`.
- Checkpoints save `{model_name, model_kwargs, state_dict}`; keep that format.
  Old checkpoints need `remap_legacy_state_dict`.
- Model output is logits. Apply sigmoid at metric time; threshold 0.5.
- Per-image metrics go in a long-format CSV (schema in the `evaluate-run` skill).
- Figures: blue `#0072B2` / orange `#E69F00`, save SVG + 300 dpi PNG. Use `paper-figure` skill.
- Pandas: read results CSVs with `keep_default_na=False` (the string "None" is a real value).
- `data/preprocessed/` is tracked in git and Oscar trains from it. Never `git rm` it:
  a pull on Oscar would delete those files there.
- No em dashes in any prose, docs, comments, or figure text James will publish.
- Do not name any file `test.py` (shadows a stdlib module).

## Oscar facts

- Repo: `/users/egottfri/code/octa-segmentation`. Conda env: `octa-seg`.
- sbatch header must `source /users/egottfri/miniconda3/etc/profile.d/conda.sh` then
  `conda activate octa-seg`. Use `set -eo pipefail`, not `set -u` (breaks conda activate).
- GPU: `-p gpu --gres=gpu:1`. SLURM does not create `logs/`; it must exist.
- Jupyter: put `%load_ext autoreload` + `%autoreload 2` at the top of notebooks so a
  `git pull` does not require a kernel restart (restart anyway if classes act strange).

## Scientific guardrails (non-negotiable)

- GT_Capillary (OCTA-500) is MODEL-GENERATED, not hand-traced. Dice against it measures
  agreement with another network. Say so whenever capillary numbers appear.
- Absolute Dice is not comparable across datasets (different vessel density floors).
- Say "significant" only after a pre-specified paired test with multiple-comparison
  correction (use `stats-compare`). Fold-level p-values so far are unadjusted.
- Preprocessing params were tuned with the classical scorer, not the U-Net.
- ARIAS: N=46 subjects, two eyes each (cluster by subject), Heidelberg device. Frame as
  pilot/feasibility. Group labels come from jlee123.
- Before reporting any new result or claim, run the `rigor-review` skill.

## ARIAS privacy (hard rule)

ARIAS is private human-subjects data, and this GitHub repo is PUBLIC.
Some raw ARIAS filenames contain surnames.
- Never open, list, or print raw ARIAS files, filenames, DOBs, or the raw subject sheet.
- The one ARIAS file you may read is `private/arias_subjects_deid.csv` (made by
  `scripts/deidentify_arias.py`; columns: subject, group, age, sex, eyes, quality, exclude).
  Use it for counts, joins, and analysis design. Never copy its rows into any tracked file,
  commit message, docs, or reports/. Aggregates (group counts, mean age) are fine.
- Scripts that touch ARIAS must output only de-identified IDs and aggregate values,
  and write per-subject outputs under `private/` or Oscar-only `results/`.
- Never `git add -f` anything under `private/`. Never commit ARIAS images.

## Known issues to check before relying on them

- `preprocess.py` `__main__` calls `preprocess_directory` with default `tile_size=8`,
  but the chosen CLAHE setting is tile 4. Confirm which tile size made
  `data/preprocessed/` before trusting raw-vs-preprocessed U-Net comparisons.
- `train.validate` Dice is hard (0.5) but pooled per batch with smoothing; test metrics in
  `evaluate.py` are per image. Do not compare val Dice to test Dice directly.
- `evaluate.auprc` uses trapezoidal `auc`, which can overestimate; prefer
  `sklearn.metrics.average_precision_score` for new reporting.
- `run_kfold` does not seed model init yet (needed for multi-seed runs).
- `test_model.evaluate_on_test` and `classical_baseline.evaluate_on_test` share a name.

## Skills in this repo

`new-experiment` write a script + sbatch. `oscar-run` run card, monitoring, failed jobs.
`evaluate-run` checkpoints to per-image CSV + summary. `stats-compare` paired tests.
`paper-figure` publication figures. `rigor-review` methods audit before claims.
