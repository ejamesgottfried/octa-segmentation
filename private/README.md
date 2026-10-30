# private/

Everything in this folder except this README is gitignored. The repo is public.

- `arias_subjects_deid.csv`: made by `python scripts/deidentify_arias.py <raw sheet>`.
  Claude Code may read this one file. Fill in `exclude` and `exclude_reason` by hand.
- Keep the raw ARIAS subject sheet OUTSIDE the repo (for example `~/leelab/`).
- Before any `git push`, `git status` should never list a file under `private/`.
