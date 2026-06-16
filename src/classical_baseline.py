import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
import itertools
from sklearn.model_selection import KFold
from evaluate import dice, evaluate


def global_threshold(img):
    """Otsu's global thresholding."""
    _, out = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return out


def adaptive_threshold(img, block_size=71, offset_c=-30):
    """Adaptive Gaussian threshold. Parameters tunable for grid search."""
    return cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv.THRESH_BINARY, block_size, offset_c)


def morphological_post_process(img, kernel_size=3):
    """Morphological closing with elliptical kernel."""
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=1)


def apply_preprocessing(img, clip=2.0, tile=4, median_kernel=3):
    """CLAHE + median filter."""
    clahe = cv.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    out = clahe.apply(img)
    out = cv.medianBlur(out, median_kernel)
    return out


def segment_adaptive(img, block_size, offset_c, clip=None, tile=None, use_morph=True):
    """
    Full adaptive pipeline with optional preprocessing.
    If clip/tile given, applies CLAHE+median first.
    """
    proc = img
    if clip is not None:
        proc = apply_preprocessing(proc, clip, tile)
    pred = adaptive_threshold(proc, block_size, offset_c)
    if use_morph:
        pred = morphological_post_process(pred)
    return pred


def cv_grid_search(img_paths, mask_paths, block_sizes, c_values,
                   clip=None, tile=None, n_splits=5, random_state=42):
    """
    Cross-validated grid search over adaptive threshold params on training data.
    Returns (best_block, best_C, results_list).
    """
    img_paths, mask_paths = list(img_paths), list(mask_paths)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results = []
    for block, C in itertools.product(block_sizes, c_values):
        fold_scores = []
        for _, val_idx in kf.split(img_paths):
            vs = []
            for i in val_idx:
                img = cv.imread(str(img_paths[i]), cv.IMREAD_GRAYSCALE)
                gt = cv.imread(str(mask_paths[i]), cv.IMREAD_GRAYSCALE)
                pred = segment_adaptive(img, block, C, clip, tile)
                vs.append(dice(pred, gt))
            fold_scores.append(np.mean(vs))
        results.append({'block_size': block, 'C': C,
                        'dice': np.mean(fold_scores), 'std': np.std(fold_scores)})

    best = max(results, key=lambda r: r['dice'])
    return best['block_size'], best['C'], results


def evaluate_on_test(img_paths, mask_paths, block_size, offset_c,
                     clip=None, tile=None, use_morph=True):
    """Apply locked params to a test set, return (mean_metrics, std_metrics)."""
    metrics = []
    for img_path, mask_path in zip(img_paths, mask_paths):
        img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
        gt = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
        pred = segment_adaptive(img, block_size, offset_c, clip, tile, use_morph)
        metrics.append(evaluate(pred, gt))
    mean = {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}
    std = {k: np.std([m[k] for m in metrics]) for k in metrics[0]}
    return mean, std