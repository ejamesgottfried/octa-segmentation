---
name: paper-figure
description: Make publication-quality figures (posters, slides, papers) from results CSVs in James's house style. Use for any plot, heatmap, grid-search figure, comparison chart, prediction overlay, PR curve, or cross-dataset matrix, and when James says a figure looks off.
---

# Figures in house style

Figures are made by a script, `scripts/figures/<fig_name>.py`, that reads only files in
`reports/` and writes to `reports/figures/`. That way any figure can be remade on the
laptop from committed CSVs without Oscar. Prediction overlays are the exception (they need
checkpoints and images), so they run on Oscar and only the PNG/SVG is committed.

## Style (put once in `src/plotting.py` as `set_paper_style()` and reuse)

```python
import matplotlib as mpl
BLUE, ORANGE, GREY = "#0072B2", "#E69F00", "#7F7F7F"   # colorblind safe
def set_paper_style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none",          # text stays editable in Illustrator/PowerPoint
        "savefig.bbox": "tight", "savefig.dpi": 300,
    })
def save(fig, stem):                     # always both formats
    fig.savefig(f"reports/figures/{stem}.svg"); fig.savefig(f"reports/figures/{stem}.png")
```
Sizes: single column 3.5 in wide, double column 7.2 in, poster panels can be larger but
keep font sizes proportional. Baseline or "ours" gets BLUE, the comparison gets ORANGE,
everything else GREY. Use a colorblind-safe sequential map (`cividis` or `viridis`) for heatmaps.

## Honest-plot rules

- Show the individual points (images or runs) on top of any mean, not bare bars.
  For paired comparisons, draw per-image lines connecting variants.
- Say in the caption what the error bars are (SD over folds x seeds, or 95% CI) and n.
- If the y-axis does not start at 0 (normal for Dice), make the range obvious and keep it
  identical across panels that are compared.
- Never put absolute Dice from different datasets on one axis as if comparable.
  Plot differences or per-dataset panels.
- Capillary panels say "GT_Capillary (model-generated labels)" in the title or caption.
- No em dashes anywhere in titles, labels, legends, or captions.

## Figure recipes for the current feedback

| Figure | Input | Notes |
|---|---|---|
| Grid-search heatmap (feedback 3) | reports/grid_search/*.csv | One heatmap per stage, value printed in each cell, best cell outlined, axis = parameter values actually searched |
| Searched-parameter table (feedback 4) | same | Table listing every parameter, every value tried, scorer used, CV scheme |
| Paired variant plot (feedback 1) | per_image.csv | x = variant, y = Dice, grey line per image, mean diamond, annotate diff + CI from stats CSV |
| Seed spread (feedback 2) | per_image.csv | points per run, colored by seed, error bar = SD over all runs |
| Cross-dataset matrix (feedback 6) | cross-dataset summary | rows = train set, columns = test set, cell = Dice; diagonal outlined; separate panel for label-swap experiment |
| PR curves (feedback 7) | saved probs or curve CSV | one curve per variant, AP in legend |
| Prediction overlay | Oscar only | TP white, FN ORANGE, FP BLUE on dark background; same crop for every variant |

## Checklist before handing over

Open the PNG and look at it (Read the image). Check: text not clipped, legend not covering
data, colors match the house palette, caption text drafted with n and error-bar meaning.
