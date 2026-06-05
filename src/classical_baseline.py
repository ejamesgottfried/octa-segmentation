import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
from evaluate import dice, iou, sensitivity, specificity, precision, evaluate

# Classical baseline methods for OCTA vessel segmentation
# Methods: global thresholding, adaptive thresholding, morphological operations
# These serve as a performance baseline to compare against deep learning approaches

def global_threshold(img):
    """
    Apply Otsu's global thresholding to segment vessels.
    Otsu's finds optimal threshold using image intensity histogram, maximizing between-class variance.
    """
    # Otsu's automatically finds best threshold value
    thresh_value, imgThresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    return imgThresh

def adaptive_threshold(img):
    """
    Apply adaptive thresholding. Later, I will sweep values to find best block_size and offset_c
    Using Gaussian Thresholding rather than mean, which weights closer pixels heavier.
    """
    
    max_val = 255

    block_size = 71 # change?

    offset_c = -30 # change?

    imgThresh = cv.adaptiveThreshold(img, max_val, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block_size, offset_c)

    return imgThresh

def morphological_post_process(img):
    """
    Apply morphological operations to threshold outputs
    Use closing (dilation -> erosion) for now, but test others later
    """
    # test other kernel sizes later. ellipse works best for octa images?
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))

    closed = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations = 1)

    return closed





def visualize_results(img, gt, pred, title=""):
    """Display original image, ground truth, and prediction side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(gt, cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    
    axes[2].imshow(pred, cmap='gray')
    axes[2].set_title(f"Prediction: {title}")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


def run_classical_baselines(img_dir, gt_dir):
    """
    Run all classical baseline methods on a directory of images.
    
    Args:
        img_dir: path to directory containing OCTA images
        gt_dir: path to directory containing ground truth masks
    
    Returns:
        dictionary of results for each method
    """
    img_dir = Path(img_dir)
    gt_dir = Path(gt_dir)
    
    # Get all image paths
    # Try tif first, fall back to png
    img_paths = sorted(img_dir.glob("*.tif"))
    if len(img_paths) == 0:
        img_paths = sorted(img_dir.glob("*.png"))
    
    # Store results for each method
    results = {
        'global': [],
        'global_morph': [],
        'adaptive': [],
        'adaptive_morph': [],
    }
    
    for img_path in img_paths:
        # Load image and ground truth
        img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
        gt = cv.imread(str(gt_dir / img_path.name), cv.IMREAD_GRAYSCALE)
        
        # Run each method
        global_pred = global_threshold(img)
        adaptive_pred = adaptive_threshold(img)
        
        # Run with and without morphology
        global_morph_pred = morphological_post_process(global_pred)
        adaptive_morph_pred = morphological_post_process(adaptive_pred)
        
        # Evaluate each method
        results['global'].append(evaluate(global_pred, gt))
        results['global_morph'].append(evaluate(global_morph_pred, gt))
        results['adaptive'].append(evaluate(adaptive_pred, gt))
        results['adaptive_morph'].append(evaluate(adaptive_morph_pred, gt))
    
    # Average results across all images
    summary = {}
    for method, method_results in results.items():
        summary[method] = {
            metric: np.mean([r[metric] for r in method_results])
            for metric in method_results[0].keys()
        }
    
    return summary


# call on the rose-1 dataset
if __name__ == "__main__":

    ROSE1_BASE = Path("/files22_lrsresearch/ENG_Lee-Lab_Shared/group/data/public/rose_dataset/ROSE-1")

    results = run_classical_baselines(
        img_dir=ROSE1_BASE / "SVC/train/img",
        gt_dir=ROSE1_BASE / "SVC/train/gt"
    )

    for method, metrics in results.items():
        print(f"\n{method}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")