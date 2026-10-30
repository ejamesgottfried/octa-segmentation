---
paths:
  - "src/**/*.py"
---

# Rules for library code in src/

- Notebooks and scripts on Oscar import these functions. Do not rename or remove a public
  function or change a positional argument without listing every caller you found
  (`grep -rn "function_name" scripts notebooks src`) and updating them in the same change.
- New behavior goes behind a keyword argument whose default keeps the old behavior.
- Each public function gets a short docstring: what it does, args, returns, and any
  shape conventions (images are (B, 1, H, W) float in [0, 1]; masks binarized with `> 127`).
- Keep the plain-English comment style already in the files. James reads these to learn.
- Registries: new models go in `MODEL_REGISTRY` (model.py). New losses go in a parallel
  `LOSS_REGISTRY` (train.py). Scripts select by name string, never by importing classes.
- No hardcoded paths. Paths come from `src/paths.py` or function arguments.
- Randomness: anything stochastic takes a `seed` argument and seeds python, numpy, torch.
- Any change that could alter numbers (metrics, transforms, splits, thresholds, losses)
  must be called out in your reply so James knows old results may not reproduce.
