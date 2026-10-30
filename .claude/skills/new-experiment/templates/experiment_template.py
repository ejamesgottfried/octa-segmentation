"""EDIT: one sentence on what this experiment tests.

Answers presentation feedback item EDIT (see docs/ROADMAP.md).
Every combination in GRID is one config; one SLURM array task trains one config
with k-fold CV. Test-set evaluation is a separate step (scripts/evaluate_experiment.py).

Usage (from the repo root):
    python scripts/EDIT_NAME.py --list-tasks   # print task table, check sbatch --array range
    python scripts/EDIT_NAME.py --smoke        # laptop CPU, synthetic data, under a minute
    python scripts/EDIT_NAME.py --task-id 3    # train one config (this is what SLURM runs)
    python scripts/EDIT_NAME.py                # train every config, one after another
"""

import argparse
import itertools
import json
import os
import random
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch

# ---------------------------------------------------------------- EDIT these
EXPERIMENT = "EDIT_NAME"          # folder name under results/
GRID = {                          # every combination becomes one task
    "dataset": ["rose1_raw"],     # names understood by dataset.get_split()
    "model": ["unet", "res_unet"],
    "seed": [0, 1, 2],
}
N_SPLITS = 5
TRAIN_KWARGS = dict(num_epochs=100, patience=10, batch_size=4, lr=1e-4)
MODEL_KWARGS = {"in_channels": 1, "num_classes": 1}
# ---------------------------------------------------------------------------


def all_configs():
    """Every grid combination, in a fixed order so task ids are stable."""
    keys = list(GRID)
    return [dict(zip(keys, values)) for values in itertools.product(*GRID.values())]


def config_id(cfg):
    """Folder-safe name for a config, e.g. 'dataset-rose1_raw__model-unet__seed-0'."""
    return "__".join(f"{k}-{v}" for k, v in cfg.items())


def set_seed(seed):
    """Seed every RNG that touches training: model init, shuffling, augmentation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def make_synthetic_split(root, n_train=6, n_test=2, size=64):
    """Tiny fake vessel images so --smoke can exercise the full pipeline on a CPU."""
    import cv2
    rng = np.random.default_rng(0)
    paths = {k: [] for k in ("train_imgs", "train_masks", "test_imgs", "test_masks")}
    for split, n in (("train", n_train), ("test", n_test)):
        for i in range(n):
            mask = np.zeros((size, size), np.uint8)
            for _ in range(4):  # a few random lines stand in for vessels
                p1, p2 = rng.integers(0, size, 2), rng.integers(0, size, 2)
                cv2.line(mask, tuple(map(int, p1)), tuple(map(int, p2)), 255, 2)
            img = np.clip(mask * 0.6 + rng.normal(60, 20, mask.shape), 0, 255).astype(np.uint8)
            ip, mp = root / f"{split}_img_{i}.png", root / f"{split}_mask_{i}.png"
            cv2.imwrite(str(ip), img)
            cv2.imwrite(str(mp), mask)
            paths[f"{split}_imgs"].append(ip)
            paths[f"{split}_masks"].append(mp)
    return paths["train_imgs"], paths["train_masks"], paths["test_imgs"], paths["test_masks"]


def is_done(out_dir):
    return (out_dir / "fold_val_dice.json").exists()


def run_config(cfg, results_root, device, smoke=False, smoke_dir=None, force=False):
    from train import run_kfold

    out_dir = results_root / config_id(cfg)
    if is_done(out_dir) and not force:
        print(f"[skip] {config_id(cfg)} already finished", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    if smoke:
        data = make_synthetic_split(smoke_dir)
        model_kwargs = {**MODEL_KWARGS, "features": (8, 16, 32, 64)}  # tiny model
        train_kwargs = dict(num_epochs=1, patience=1, batch_size=2, lr=1e-3)
        n_splits = 2
    else:
        from dataset import get_split  # added in ROADMAP Phase 0
        data = get_split(cfg["dataset"])
        model_kwargs, train_kwargs, n_splits = MODEL_KWARGS, TRAIN_KWARGS, N_SPLITS

    train_imgs, train_masks, test_imgs, test_masks = data
    assert len(train_imgs) == len(train_masks) > 0, "empty or mismatched training lists"

    (out_dir / "config.json").write_text(json.dumps({
        "experiment": EXPERIMENT, "config": cfg, "config_id": config_id(cfg),
        "model_kwargs": {k: list(v) if isinstance(v, tuple) else v for k, v in model_kwargs.items()},
        "train_kwargs": train_kwargs, "n_splits": n_splits,
        "n_train": len(train_imgs), "n_test": len(test_imgs),
        "git_commit": git_commit(), "host": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID"),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"), "smoke": smoke,
    }, indent=2))

    set_seed(int(cfg.get("seed", 0)))
    t0 = time.time()
    # EDIT: if GRID has a "loss" key, pass loss_name=cfg["loss"] (needs Phase 0 LOSS_REGISTRY).
    fold_scores = run_kfold(
        train_imgs, train_masks, test_imgs, test_masks,
        model_name=cfg["model"], model_kwargs=model_kwargs,
        n_splits=n_splits, device=device, output_dir=str(out_dir), **train_kwargs)

    # Written last: its presence is how resumability knows the config finished.
    (out_dir / "fold_val_dice.json").write_text(json.dumps({
        "fold_val_dice": [float(s) for s in fold_scores],
        "minutes": round((time.time() - t0) / 60, 1),
    }, indent=2))
    print(f"[done] {config_id(cfg)} in {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task-id", type=int, help="run one config (SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--list-tasks", action="store_true", help="print task table and exit")
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic run on CPU")
    ap.add_argument("--force", action="store_true", help="retrain finished configs")
    args = ap.parse_args()

    configs = all_configs()
    if args.list_tasks:
        for i, cfg in enumerate(configs):
            print(f"{i:3d}  {config_id(cfg)}")
        print(f"{len(configs)} tasks  ->  #SBATCH --array=0-{len(configs) - 1}%8")
        return

    if args.smoke:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "data").mkdir()
            run_config(configs[0], tmp / "results", "cpu", smoke=True, smoke_dir=tmp / "data")
        print("SMOKE OK", flush=True)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  experiment: {EXPERIMENT}", flush=True)
    results_root = REPO / "results" / EXPERIMENT
    todo = [configs[args.task_id]] if args.task_id is not None else configs
    for cfg in todo:
        run_config(cfg, results_root, device, force=args.force)


if __name__ == "__main__":
    main()
