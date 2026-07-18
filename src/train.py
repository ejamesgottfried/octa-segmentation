import torch
import torch.nn as nn
import torch.nn.functional as F

from model import build_model
from dataset import get_kfold_splits, create_datasets
from plotting import plot_training_curves
from torch.utils.data import DataLoader
import numpy as np

# Combine Dice with BCE

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Convert raw logits to probabilities with sigmoid
        probs = torch.sigmoid(logits)
        
        # Flatten both to 1D so we can compute sums easily
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # Compute intersection and the dice coefficient
        # intersection = sum of (probs * targets)
        # dice = (2*intersection + smooth) / (probs.sum() + targets.sum() + smooth)

        intersection = (probs * targets).sum()

        # Calculate Dice
        dice = (2 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)

        # 4. Return 1 - dice
        return 1 - dice
    
class CombinedLoss(nn.Module):
    """
    Combined BCE + Dice loss for vessel segmentation.
    BCE gives stable per-pixel gradients,
    Dice handles the class imbalance (thin vessels).
    """
    def __init__(self, smooth=1.0, bce_weight=0.5):
        super().__init__()
        self.dice = DiceLoss(smooth=smooth)
        # built in BCE
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        dice_loss = self.dice(logits, targets)
        bce_loss = self.bce(logits, targets)
        # Weighted sum of the two
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss
    

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """
    Run one epoch of training. Returns average loss over the epoch.
    """
    model.train()  # set model to training mode (enables dropout, batchnorm updates)
    running_loss = 0.0

    for images, masks in loader:
        # 1. Move data to GPU (or CPU)
        images = images.to(device)
        masks = masks.to(device)

        # 2. Forward pass
        logits = model(images)

        # 3. Compute loss
        loss = loss_fn(logits, masks)

        # 4. Backward pass: clear old gradients, then compute new ones
        optimizer.zero_grad()
        loss.backward()

        # 5. Update weights
        optimizer.step()

        # 6. Track loss (.item() pulls the number out of the tensor)
        running_loss += loss.item()

    # Average loss across all batches
    return running_loss / len(loader)

def compute_dice(preds, targets, smooth=1.0):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return ((2 * intersection + smooth) / 
            (preds.sum() + targets.sum() + smooth)).item()

def validate(model, loader, loss_fn, device):
    """
    Run validation. Returns average loss and average Dice score.
    No weight updates happen here.
    """
    model.eval()  # set to evaluation mode (batchnorm/dropout behave differently)
    running_loss = 0.0
    running_dice = 0.0

    with torch.no_grad():  # disable gradient tracking: saves memory, faster
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass only
            logits = model(images)
            loss = loss_fn(logits, masks)
            running_loss += loss.item()

            # Compute Dice metric on binarized predictions
            preds = (torch.sigmoid(logits) > 0.5).float()
            dice_score = compute_dice(preds, masks)
            running_dice += dice_score

    avg_loss = running_loss / len(loader)
    avg_dice = running_dice / len(loader)
    return avg_loss, avg_dice

def train_fold(model, train_loader, val_loader, loss_fn, optimizer, device,
               model_name, model_kwargs=None,
               num_epochs=100, patience=10, save_path="best_model.pth"):
    """
    Train a model on one fold with early stopping.
    Saves the best model (by validation Dice) to save_path.
    Returns the best validation Dice achieved.
    """
    model_kwargs = model_kwargs or {}
    best_val_dice = 0.0
    epochs_without_improvement = 0

    # Track for plotting
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': []
    }

    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)

        # Validate
        val_loss, val_dice = validate(model, val_loader, loss_fn, device)

        # record epoch
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)

        # Log progress
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"train_loss: {train_loss:.4f} | "
              f"val_loss: {val_loss:.4f} | "
              f"val_dice: {val_dice:.4f}")

        # Save best model & check early stopping
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_without_improvement = 0
            torch.save({
                'model_name': model_name,
                'model_kwargs': model_kwargs,
                'state_dict': model.state_dict(),
            }, save_path)  # save best weights + how to rebuild the model
            print(f"New best! Saved (val_dice: {val_dice:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return best_val_dice,history


def run_kfold(train_imgs, train_masks, test_imgs, test_masks,
              model_name='unet', model_kwargs=None,
              n_splits=5, num_epochs=100, patience=10, batch_size=4,
              lr=1e-4, device='cuda', output_dir='results'):
    """
    Run k-fold cross-validation training.
    Trains a fresh model on each fold, saves best weights and curves,
    returns the list of best validation Dice scores.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    model_kwargs = model_kwargs or {'in_channels': 1, 'num_classes': 1}

    # Get the fold splits (test set held out, 5 folds on training data)
    splits = get_kfold_splits(train_imgs, train_masks, test_imgs, test_masks,
                              n_splits=n_splits)

    fold_scores = []

    for fold_idx, fold in enumerate(splits['folds']):
        print(f"\n{'='*50}")
        print(f"FOLD {fold_idx + 1}/{n_splits}")
        print(f"{'='*50}")

        # Create datasets for this fold
        train_ds, val_ds, test_ds = create_datasets(fold, splits['test'])

        # Wrap in DataLoaders
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Fresh model & optimizer for each fold
        model = build_model(model_name, **model_kwargs).to(device)
        loss_fn = CombinedLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Train this fold
        save_path = f"{output_dir}/best_model_fold{fold_idx+1}.pth"
        best_dice, history = train_fold(
            model, train_loader, val_loader, loss_fn, optimizer, device,
            model_name=model_name, model_kwargs=model_kwargs,
            num_epochs=num_epochs, patience=patience, save_path=save_path
        )

        # Save the training curve for this fold
        curve_path = f"{output_dir}/training_curve_fold{fold_idx+1}.png"
        plot_training_curves(history, save_path=curve_path)

        fold_scores.append(best_dice)
        print(f"Fold {fold_idx+1} best Dice: {best_dice:.4f}")

    # Summary across all folds
    mean_dice = np.mean(fold_scores)
    std_dice = np.std(fold_scores)
    print(f"\n{'='*50}")
    print(f"K-FOLD RESULTS: {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"Individual folds: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"{'='*50}")

    return fold_scores


# to run:
if __name__ == "__main__":
    from pathlib import Path
    from dataset import get_octa500_split

    # use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Which dataset to train on: 'rose1' or 'octa500'
    dataset_name = 'rose1'

    if dataset_name == 'rose1':
        # Data paths (preprocessed SVC on Oscar)
        PREP_BASE = Path("/users/egottfri/code/octa-segmentation/data/preprocessed")
        ROSE1_BASE = Path("/files22_lrsresearch/ENG_Lee-Lab_Shared/group/data/public/rose_dataset/ROSE-1")

        # Preprocessed images, original ground truth masks
        train_imgs = sorted((PREP_BASE / "SVC/train/img").glob("*.tif"))
        train_masks = sorted((ROSE1_BASE / "SVC/train/gt").glob("*.tif"))
        test_imgs = sorted((PREP_BASE / "SVC/test/img").glob("*.tif"))
        test_masks = sorted((ROSE1_BASE / "SVC/test/gt").glob("*.tif"))
    elif dataset_name == 'octa500':
        train_imgs, train_masks, test_imgs, test_masks = get_octa500_split()
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Options: 'rose1', 'octa500'")

    print(f"Train images: {len(train_imgs)}, Train masks: {len(train_masks)}")
    print(f"Test images: {len(test_imgs)}, Test masks: {len(test_masks)}")

    # Run k-fold training
    model_name = 'unet'
    fold_scores = run_kfold(
        train_imgs, train_masks, test_imgs, test_masks,
        model_name=model_name,
        model_kwargs={'in_channels': 1, 'num_classes': 1},
        n_splits=5,
        num_epochs=100,
        patience=10,
        batch_size=4,
        lr=1e-4,
        device=device,
        output_dir=f'results/{model_name}_{dataset_name}'
    )