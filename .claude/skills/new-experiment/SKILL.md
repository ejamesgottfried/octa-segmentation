---
name: new-experiment
description: Create a new training or evaluation experiment script plus its SLURM sbatch file for Oscar. Use when James asks for a new experiment, sweep, ablation, seed run, loss comparison, cross-dataset run, or "a script I can run on Oscar".
argument-hint: "[experiment-name] [what it tests]"
---

# New experiment: script + sbatch

Goal: a script that is boring to run. One SLURM array task = one config, resumable,
smoke-testable on the laptop, and self-describing (config.json with git commit).

## Steps

1. **Pin down the question first.** Write 2 to 4 lines for James before any code:
   - Which presentation-feedback item (docs/ROADMAP.md) this answers.
   - The grid: datasets x models x seeds x (loss / preprocessing / ...). Count the configs.
   - The primary comparison and metric, decided now (goes into docs/RESULTS.md
     "Pre-registered comparisons" before results exist).
   - Rough GPU cost: configs x folds x minutes per fold (use the run log in RESULTS.md).
   If the grid is over ~60 configs, propose a smaller first pass.

2. **Copy the templates**, do not hand-roll:
   - `templates/experiment_template.py` to `scripts/<name>.py`
   - `templates/job_template.sbatch` to `slurm/<name>.sbatch`
   Fill every `EDIT` marker. Keep the CLI flags exactly as in the template.

3. **Data comes from `src/dataset.py`**, by name, via `get_split(name)`. If the dataset
   name is new, add a loader there (no paths in the script). If `get_split` does not
   exist yet, stop and do ROADMAP Phase 0 first.

4. **Anything new in the model or loss goes in `src/`** behind a registry name
   (`MODEL_REGISTRY`, `LOSS_REGISTRY`) with defaults that keep old behavior.

5. **Verify on the laptop:**
   ```bash
   python scripts/<name>.py --list-tasks      # check count matches --array range
   python scripts/<name>.py --smoke           # synthetic data, CPU, must finish < 1 min
   ```
   Fix until both pass. Do not claim the script works without running `--smoke`.

6. **Set the array range** in the sbatch to `0-(N-1)` from `--list-tasks`, with a `%8`
   throttle. Set `-t` from the run log plus about 30%. Unknown runtime: 2 hours, and the
   run card starts with a single test task.

7. **Finish with a run card** (use the `oscar-run` skill format) and add a row to
   the experiment table in docs/ROADMAP.md with status `ready`.

## Output layout the template produces

```
results/<experiment>/<config_id>/
    config.json                 # grid values, args, git commit, host, start time
    best_model_fold{k}.pth      # {model_name, model_kwargs, state_dict}
    training_curve_fold{k}.png
    fold_val_dice.json          # written last; its presence means "config finished"
```

## Do not

- Do not evaluate on the test set inside the training script. That is `evaluate-run`.
- Do not pick thresholds, epochs, or hyperparameters by looking at test metrics.
- Do not change folds or test splits (see CLAUDE.md conventions).
