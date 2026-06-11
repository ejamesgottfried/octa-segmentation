import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import Dataset
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split, KFold

class ROSEDataset(Dataset):
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
        return len(self.img_paths)
    
    def __getitem__(self, idx):

        # load and convert image to numpy array
        img = cv2.imread(str(self.img_path[idx]), cv2.IMREAD_GRAYSCALE) 
        mask = cv2.imread(str(self.mask_path[idx]), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise ValueError(f"Failed to load image or mask at index {idx}")


        if self.augment:
            transformed = self.train_transform(image = img, mask = mask)
        else: 
            transformed = self.val_transform(image = img, mask = mask)

        return transformed['image'], transformed['mask']
    

    
def get_kfold_splits(img_paths, mask_paths, n_splits=5, test_size=0.2, random_state=42):
    """
    Split dataset into test set and k-fold train/val splits.

    for rose-1: 9 image test set
    24 train, 6 val for each fold
    
    Args:
        img_paths: list of image paths
        mask_paths: list of mask paths
        n_splits: number of folds (default 5)
        test_size: fraction for test set (default 0.2)
        random_state: random seed for reproducibility
    
    Returns:
        dictionary with test set and k fold splits
    """
    # First split off test set
    train_val_imgs, test_imgs, train_val_masks, test_masks = train_test_split(
        img_paths, mask_paths,
        test_size=test_size,
        random_state=random_state
    )
    
    # Apply k-fold to remaining training/validation data
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    folds = []
    for train_idx, val_idx in kfold.split(train_val_imgs):
        # Get paths for this fold
        train_imgs = [train_val_imgs[i] for i in train_idx]
        train_masks = [train_val_masks[i] for i in train_idx]
        val_imgs = [train_val_imgs[i] for i in val_idx]
        val_masks = [train_val_masks[i] for i in val_idx]
        
        folds.append((train_imgs, train_masks, val_imgs, val_masks))
    
    return {
        'test': (test_imgs, test_masks),
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
    
    train_dataset = ROSEDataset(train_imgs, train_masks, augment=True)
    val_dataset = ROSEDataset(val_imgs, val_masks, augment=False)
    test_dataset = ROSEDataset(test_imgs, test_masks, augment=False)
    
    return train_dataset, val_dataset, test_dataset


