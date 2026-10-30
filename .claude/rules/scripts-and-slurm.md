---
paths:
  - "scripts/**/*.py"
  - "slurm/**"
  - "**/*.sbatch"
---

# Rules for experiment scripts and SLURM files

- Start new scripts from `.claude/skills/new-experiment/templates/`. Do not hand-roll.
- A script must run end to end with `python scripts/X.py --smoke` on a laptop CPU in
  under a minute, using synthetic data. Run that before saying a script is done.
- Each SLURM array task trains exactly one config. `--list-tasks` prints the task table
  so the `--array=0-N` range in the sbatch file can be checked against it.
- Results go to `results/<experiment>/<config-id>/`. Never overwrite a finished config;
  skip it (resumable) unless `--force` is passed.
- sbatch files: `set -eo pipefail`, logs to `logs/%x-%A_%a.out`, `mkdir -p logs` note in
  the header, conda activation exactly as in CLAUDE.md, `cd` to the repo root,
  print `git rev-parse --short HEAD` and `hostname` at start.
- Time limits: set from measured runtimes (see `docs/RESULTS.md` run log), plus ~30%.
  A first run of a new script uses a short test job (`--array=0`) before the full array.
- Do not add `--mail-user` with a placeholder address. Use James's address or omit it.
