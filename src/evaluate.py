import numpy as np
from sklearn.metrics import auc, precision_recall_curve

# Take in both ground truth and prediction mask as inputs
# Flatten/binarize inputs before calculations


# Calculate Dice and IoU, good metrics for imblanced classes
# Dice is more commonly used I believe
def dice(pred, gt):
    # Binarize inputs (ensure 0s and 1s)
    pred = (pred > 0).astype(np.float32)
    gt = (gt > 0).astype(np.float32)

    if pred.sum() == 0 and gt.sum() == 0:
        raise ValueError("Both prediction and ground truth are empty!")
    
    # Calculate intersection (bitwise and)
    intersection = np.logical_and(pred, gt).sum()

    # Calculate Dice
    dice = (2 * intersection) / (pred.sum() + gt.sum())
    
    return dice

def iou(pred, gt):
    # Binarize inputs (ensure 0s and 1s)
    pred = (pred > 0).astype(np.float32)
    gt = (gt > 0).astype(np.float32)

    if pred.sum() == 0 and gt.sum() == 0:
        raise ValueError("Both prediction and ground truth are empty!")

    # Calculate intersection (bitwise and)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    iou = intersection / union

    return iou


# Calculate True Positive (Recall/Sensitivity) "How often we find vessels"
def sensitivity(pred, gt):
    # Binarize inputs (ensure 0s and 1s)
    pred = (pred > 0).astype(np.float32)
    gt = (gt > 0).astype(np.float32)

    # Calculate true positives and false negatives
    # True positive when pred and gt are both 1
    tp = np.sum((pred == 1) & (gt == 1))
    # False negative when pred = 0, gt = 1
    fn = np.sum((pred == 0) & (gt == 1))

    if (tp + fn) == 0:
        raise ValueError("No vessel pixels in ground truth!")
    
    tp_rate = tp / (tp + fn)

    return tp_rate

# Find specificity "How often we avoid false positives"
def specificity(pred, gt):
    # Binarize inputs (ensure 0s and 1s)
    pred = (pred > 0).astype(np.float32)
    gt = (gt > 0).astype(np.float32)

    # Calculate true positives and false negatives
    # True negative when pred and gt are both 0
    tn = np.sum((pred == 0) & (gt == 0))
    # False positive when pred = 1, gt = 0
    fp = np.sum((pred == 1) & (gt == 0))

    if (tn + fp) == 0:
        raise ValueError("No background pixels in gt, very weird!")
    
    tn_rate = tn / (tn + fp)

    return tn_rate


# Calculate Precision
def precision(pred, gt):
    # Binarize inputs (ensure 0s and 1s)
    pred = (pred > 0).astype(np.float32)
    gt = (gt > 0).astype(np.float32)

    # Calculate true positives and false positives
    # True positive when pred and gt are both 1
    tp = np.sum((pred == 1) & (gt == 1))
    # false positive when pred = 1, gt = 0
    fp = np.sum((pred == 1) & (gt == 0))

    if (tp + fp) == 0:
        raise ValueError("Model didn't predict any vessels!")
    
    precision = tp / (tp + fp)

    return precision

# Calculate Area under PR curve
def auprc(prob_map, gt):

    gt = (gt > 0).astype(np.float32)
    
    # Flatten to 1D arrays
    gt_flat = gt.flatten()
    prob_flat = prob_map.flatten()
    
    # Compute precision-recall curve across all thresholds
    p, r, _ = precision_recall_curve(gt_flat, prob_flat)
    
    # Compute area under curve
    return auc(r, p)

# Return all metrics as dictionary
def evaluate(pred, gt, prob_map=None):
    """
    Compute all segmentation metrics.
    
    Args:
        pred: binary prediction mask
        gt: binary ground truth mask
        prob_map: probability map (optional, for AUPRC)
    
    Returns:
        dictionary of all metrics
    """

    metrics = {
        'dice': dice(pred, gt),
        'iou': iou(pred, gt),
        'sensitivity': sensitivity(pred, gt),
        'specificity': specificity(pred, gt),
        'precision': precision(pred, gt),
    }
    
    if prob_map is not None:
        p, r, _ = precision_recall_curve(gt.flatten(), prob_map.flatten())
        metrics['auprc'] = auc(r, p)
    
    return metrics