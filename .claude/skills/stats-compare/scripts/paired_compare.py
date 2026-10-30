"""Paired comparisons of segmentation variants from a long-format per-image CSV.

Primary analysis (unit = test image):
    For each variant, average the metric over folds (and seeds) per test image, then
    compare variants image by image. Same images, same folds, so the comparison is paired.
    Reports mean difference, bootstrap 95% CI, Cohen's dz, paired t and Wilcoxon p-values,
    and Holm-adjusted p-values across all comparisons in this call.

Secondary analysis (unit = training run, i.e. fold x seed):
    Pairs per-run test means. Folds share training data, so these p-values are
    optimistic (Nadeau and Bengio 2003). Report as a check, never as the headline.

Variance mode (--variance): splits spread into across-fold and across-seed parts,
which answers "do error bars grow with multiple seeds?".

Example:
    python paired_compare.py reports/rose1_ablation/per_image.csv \
        --metric dice --baseline unet --compare res_unet attention_unet se_unet att_res_unet \
        --where dataset=rose1_raw --out reports/rose1_ablation/stats_dice_raw.csv
"""

import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats


def holm(pvals):
    """Holm-Bonferroni step-down adjustment. NaNs pass through untouched."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.where(ok)[0][np.argsort(p[ok])]
    m, running = len(idx), 0.0
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def bootstrap_ci(diffs, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def load(path, metric, where, group_col, use_ens):
    df = pd.read_csv(path, keep_default_na=False, dtype=str)
    for cond in where:
        key, val = cond.split("=", 1)
        if key not in df.columns:
            sys.exit(f"--where column '{key}' not in CSV columns {list(df.columns)}")
        df = df[df[key] == val]
    if "fold" in df.columns:
        df = df[df["fold"] == "ens"] if use_ens else df[df["fold"] != "ens"]
    for col in (metric, group_col, "image_id"):
        if col not in df.columns:
            sys.exit(f"column '{col}' missing from CSV")
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    n_nan = int(df[metric].isna().sum())
    if n_nan:
        print(f"WARNING: {n_nan} rows have no {metric} value (see 'note' column); excluded.")
    df = df.dropna(subset=[metric])
    if "seed" not in df.columns:
        df["seed"] = "0"
    if "fold" not in df.columns:
        df["fold"] = "0"
    return df


def per_image(df, metric, group_col, variant):
    sub = df[df[group_col] == variant]
    if sub.empty:
        sys.exit(f"no rows for {group_col}={variant}")
    return sub.groupby("image_id")[metric].mean()


def per_run(df, metric, group_col, variant):
    sub = df[df[group_col] == variant]
    return sub.groupby(["seed", "fold"])[metric].mean()


def compare(df, metric, group_col, baseline, others, n_boot):
    rows = []
    base_img = per_image(df, metric, group_col, baseline)
    base_run = per_run(df, metric, group_col, baseline)
    for other in others:
        oth_img = per_image(df, metric, group_col, other)
        common = base_img.index.intersection(oth_img.index)
        if len(common) < len(base_img) or len(common) < len(oth_img):
            print(f"WARNING: {other} vs {baseline}: only {len(common)} images in common")
        b, o = base_img.loc[common].to_numpy(), oth_img.loc[common].to_numpy()
        d = o - b
        sd = d.std(ddof=1) if len(d) > 1 else np.nan
        lo, hi = bootstrap_ci(d, n_boot) if len(d) > 1 else (np.nan, np.nan)
        p_t = stats.ttest_rel(o, b).pvalue if len(d) > 1 and sd > 0 else np.nan
        p_w = (stats.wilcoxon(d).pvalue
               if len(d) > 1 and np.any(d != 0) else np.nan)

        oth_run = per_run(df, metric, group_col, other)
        rc = base_run.index.intersection(oth_run.index)
        rd = oth_run.loc[rc].to_numpy() - base_run.loc[rc].to_numpy()
        p_run = (stats.ttest_rel(oth_run.loc[rc], base_run.loc[rc]).pvalue
                 if len(rc) > 1 and rd.std() > 0 else np.nan)

        rows.append({
            "metric": metric, "baseline": baseline, "variant": other,
            "n_images": len(d), "mean_baseline": b.mean(), "mean_variant": o.mean(),
            "mean_diff": d.mean(), "ci95_low": lo, "ci95_high": hi,
            "cohens_dz": d.mean() / sd if sd and sd > 0 else np.nan,
            "images_improved": int((d > 0).sum()),
            "p_ttest": p_t, "p_wilcoxon": p_w,
            "n_runs": len(rc), "p_run_ttest_optimistic": p_run,
        })
    out = pd.DataFrame(rows)
    out["p_ttest_holm"] = holm(out["p_ttest"])
    out["p_wilcoxon_holm"] = holm(out["p_wilcoxon"])
    return out


def variance_table(df, metric, group_col):
    rows = []
    for variant, sub in df.groupby(group_col):
        run_means = sub.groupby(["seed", "fold"])[metric].mean()
        seeds = run_means.index.get_level_values("seed").unique()
        within_seed_fold_sd = np.nanmean([
            run_means.xs(s, level="seed").std(ddof=1) if len(run_means.xs(s, level="seed")) > 1 else np.nan
            for s in seeds])
        seed_means = run_means.groupby(level="seed").mean()
        rows.append({
            group_col: variant, "n_seeds": len(seeds),
            "n_runs": len(run_means), "mean": run_means.mean(),
            "sd_all_runs": run_means.std(ddof=1),
            "sd_across_folds_within_seed": within_seed_fold_sd,
            "sd_across_seed_means": seed_means.std(ddof=1) if len(seeds) > 1 else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv")
    ap.add_argument("--metric", default="dice")
    ap.add_argument("--group-col", default="model", help="column holding variant names")
    ap.add_argument("--baseline")
    ap.add_argument("--compare", nargs="+")
    ap.add_argument("--where", nargs="*", default=[], help="filters like dataset=rose1_raw")
    ap.add_argument("--ens", action="store_true", help="use fold-ensemble rows instead of folds")
    ap.add_argument("--variance", action="store_true", help="variance decomposition only")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", help="write the result table to this CSV")
    args = ap.parse_args()

    df = load(args.csv, args.metric, args.where, args.group_col, args.ens)
    if args.variance:
        res = variance_table(df, args.metric, args.group_col)
    else:
        if not args.baseline or not args.compare:
            sys.exit("--baseline and --compare are required unless --variance")
        res = compare(df, args.metric, args.group_col, args.baseline, args.compare, args.n_boot)

    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.float_format", "{:.4g}".format):
        print(res.to_string(index=False))
    if args.out:
        res.to_csv(args.out, index=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
