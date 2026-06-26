import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import Dataset
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split, KFold

class OCTADataset(Dataset):
    def __init__(self, img_paths, mask_paths, augment=False):
        # store image and mask paths
        # store augmentation flag
        # define augmentation transforms

        self.img_path = img_paths
        self.mask_path = mask_paths
        self.augment = augment


        # validation transform, just normalize
        self.val_transform = A.Compose([
            A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0), 
            ToTensorV2()
        ])

        # training transform, random augmentations
        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, idx):
        img = cv2.imread(str(self.img_path[idx]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.mask_path[idx]), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise ValueError(f"Failed to load image or mask at index {idx}")

        # Binarize mask to 0/1 (cv2 loads it as 0/255)
        mask = (mask > 127).astype(np.float32)

        if self.augment:
            transformed = self.train_transform(image=img, mask=mask)
        else:
            transformed = self.val_transform(image=img, mask=mask)

        image = transformed['image']
        mask = transformed['mask']

        # Ensure mask is float with a channel dimension: (1, H, W)
        mask = mask.unsqueeze(0).float()

        return image, mask
    

    
def get_kfold_splits(train_img_paths, train_mask_paths,
                     test_img_paths, test_mask_paths,
                     n_splits=5, random_state=42):
    """
    Apply k-fold cross validation to a pre-split dataset.
    Uses ROSE-1's existing train/test split.

    For ROSE-1 SVC: 30 train images → 5 folds of 24 train / 6 val.
                    9 test images held out (passed through unchanged).
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds = []
    for train_idx, val_idx in kfold.split(train_img_paths):
        train_imgs = [train_img_paths[i] for i in train_idx]
        train_masks = [train_mask_paths[i] for i in train_idx]
        val_imgs = [train_img_paths[i] for i in val_idx]
        val_masks = [train_mask_paths[i] for i in val_idx]
        folds.append((train_imgs, train_masks, val_imgs, val_masks))

    return {
        'test': (test_img_paths, test_mask_paths),
        'folds': folds
    }


def create_datasets(fold, test_split):
    """
    Create train, val, and test datasets from a fold.
    
    Args:
        fold: tuple of (train_imgs, train_masks, val_imgs, val_masks)
        test_split: tuple of (test_imgs, test_masks)
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    train_imgs, train_masks, val_imgs, val_masks = fold
    test_imgs, test_masks = test_split
    
    train_dataset = OCTADataset(train_imgs, train_masks, augment=True)
    val_dataset = OCTADataset(val_imgs, val_masks, augment=False)
    test_dataset = OCTADataset(test_imgs, test_masks, augment=False)
    
    return train_dataset, val_dataset, test_dataset


