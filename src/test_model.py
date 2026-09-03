"""Loads saved fold checkpoints (as written by train.run_kfold) and reports
mean +/- std metrics across folds on a held-out test set.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))  # allow `from evaluate import ...` etc. when run directly

from evaluate import evaluate
from model import build_model
from dataset import OCTADataset
from torch.utils.data import DataLoader
import torch
import numpy as np


def evaluate_on_test(test_imgs, test_masks, model_paths, device='cuda'):
    """
    Evaluate saved fold models on the held-out test set.
    Returns averaged metrics across all folds.
    """
    test_ds = OCTADataset(test_imgs, test_masks, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    all_fold_metrics = []

    for fold_idx, model_path in enumerate(model_paths):
        # Load this fold's best model, rebuilt from the checkpoint's own metadata
        checkpoint = torch.load(model_path, map_location=device)
        model = build_model(checkpoint['model_name'], **checkpoint['model_kwargs']).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        fold_metrics = []
        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(device)
                logits = model(images)
                probs = torch.sigmoid(logits)

                # Convert to numpy for your evaluate() function
                pred = probs.cpu().numpy().squeeze()    # (H, W) probabilities
                gt = masks.cpu().numpy().squeeze()      # (H, W) ground truth
                pred_binary = (pred > 0.5).astype(np.uint8)

                # Your evaluate() returns a dict of all metrics.
                # prob_map stays continuous so AUPRC has scores to work with.
                metrics = evaluate(pred_binary, gt, prob_map=pred)
                fold_metrics.append(metrics)

        # Average metrics over the test images for this fold
        avg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        all_fold_metrics.append(avg)
        print(f"Fold {fold_idx+1}: " +
              ", ".join(f"{k}={v:.4f}" for k, v in avg.items()))

    # Average across the folds
    final = {k: np.mean([fm[k] for fm in all_fold_metrics]) for k in all_fold_metrics[0]}
    std = {k: np.std([fm[k] for fm in all_fold_metrics]) for k in all_fold_metrics[0]}

    print("\nTEST SET RESULTS (mean ± std across folds)")
    for k in final:
        print(f"{k}: {final[k]:.4f} ± {std[k]:.4f}")

    return final, std