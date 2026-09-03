"""Image preprocessing: CLAHE + median filtering, run once and cached to
disk (see preprocess_directory / the __main__ block below) so training can
just read the already-preprocessed images instead of redoing this per epoch.
"""

import numpy as np
import cv2
from pathlib import Path


def apply_clahe(img, clip_limit=2.0, tile_size=4):
    """Contrast Limited Adaptive Histogram Equalization: boosts local
    contrast (per tile) rather than globally, making subtle vessel
    structure more visible without over-amplifying noise."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(img)


def apply_median_filter(img, kernel_size=3):
    """Median blur: denoises while preserving edges, unlike a Gaussian blur."""
    return cv2.medianBlur(img, kernel_size)


def preprocess_image(img, clip_limit=2.0, tile_size=8, kernel_size=3):
    """CLAHE followed by median filtering."""
    img = apply_clahe(img, clip_limit, tile_size)
    img = apply_median_filter(img, kernel_size)
    return img

def preprocess_directory(input_dir, output_dir, clip_limit=2.0, tile_size=8, kernel_size=3):
    """
    Preprocess all images in a directory and save to output directory.
    Applies CLAHE and median filtering to each image.
    
    Args:
        input_dir: path to directory containing raw images
        output_dir: path to save preprocessed images
        clip_limit: CLAHE clip limit
        tile_size: CLAHE tile size
        kernel_size: median filter kernel size
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle both .tif and .png
    img_paths = sorted(input_dir.glob("*.tif"))
    if len(img_paths) == 0:
        img_paths = sorted(input_dir.glob("*.png"))
    if len(img_paths) == 0:
        img_paths = sorted(input_dir.glob("*.bmp"))
    
    print(f"Processing {len(img_paths)} images from {input_dir.name}...")
    
    for img_path in img_paths:
        # Load image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"failed to load {img_path.name}, skipping")
            continue
        
        # Preprocess
        processed = preprocess_image(img, clip_limit, tile_size, kernel_size)
        
        # Save with same filename
        save_path = output_dir / img_path.name
        cv2.imwrite(str(save_path), processed)
    
    print(f"Done! Saved to {output_dir}")


if __name__ == "__main__":
    # EDIT ME: point this at your own ROSE-1 copy.
    ROSE1_BASE = Path("/files22_lrsresearch/ENG_Lee-Lab_Shared/group/data/public/rose_dataset/ROSE-1")

    # Preprocess SVC train/test images; output feeds src/train.py's 'rose1'
    # branch and scripts/rose1_ablation.py's "preprocessed" condition.
    for split in ["train", "test"]:
        preprocess_directory(
            input_dir=ROSE1_BASE / f"SVC/{split}/img",
            output_dir=Path(f"/users/egottfri/code/octa-segmentation/data/preprocessed/SVC/{split}/img")
        )