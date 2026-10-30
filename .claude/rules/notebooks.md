---
paths:
  - "notebooks/**"
---

# Rules for notebooks

- Notebooks are for looking at results, not for computing them. If a cell does more
  than load, filter, and plot, that logic belongs in `src/` or `scripts/`.
- First cell, always:
  ```python
  %load_ext autoreload
  %autoreload 2
  import sys; sys.path.insert(0, "../../src")  # from notebooks/<subdir>/
  ```
- Read results with `pd.read_csv(path, keep_default_na=False)`.
- Notebook outputs (embedded images) are most of the repo's size and can leak data.
  Prefer clearing outputs before committing (Kernel > Restart and Clear Outputs, or
  `nbstripout` once it is set up per docs/ROADMAP.md P0). Never commit outputs that show
  ARIAS subject details.
- When editing a notebook, prefer giving James copy-paste-ready cells in your reply over
  rewriting the .ipynb JSON, unless he asks you to edit the file.
