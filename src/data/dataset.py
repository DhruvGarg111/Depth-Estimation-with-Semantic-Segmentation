"""
Dataset classes for multi-modal depth estimation.

Supports loading RGB images, depth maps, sparse depth, and semantic segmentation
from the NYU Depth V2 dataset format.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from .transforms import SynchronizedTransform, get_train_transforms, get_val_transforms


class RGBDSegDataset(Dataset):
    """
    Multi-modal dataset for depth estimation with semantic segmentation.
    
    Loads aligned RGB images, ground truth depth maps, sparse depth maps,
    and semantic segmentation maps for training depth estimation models.
    
    Args:
        rgb_dir: Path to directory containing RGB images
        depth_dir: Path to directory containing ground truth depth maps
        sparse_dir: Path to directory containing sparse depth maps
        semseg_dir: Path to directory containing semantic segmentation maps
        file_list: List of filenames to include in the dataset
        transform: Transform to apply to samples (default: None)
        train: Whether this is a training dataset (default: True)
    
    Example:
        >>> dataset = RGBDSegDataset(
        ...     rgb_dir='data/images',
        ...     depth_dir='data/depths',
        ...     sparse_dir='data/sparse',
        ...     semseg_dir='data/semseg',
        ...     file_list=['0001.png', '0002.png'],
        ...     train=True
        ... )
        >>> sample = dataset[0]
        >>> print(sample['rgb'].shape)  # [3, 256, 256]
    """
    
    def __init__(
        self,
        rgb_dir: str,
        depth_dir: str,
        sparse_dir: str,
        semseg_dir: str,
        file_list: List[str],
        transform: Optional[Callable] = None,
        train: bool = True
    ):
        self.rgb_dir = Path(rgb_dir)
        self.depth_dir = Path(depth_dir)
        self.sparse_dir = Path(sparse_dir)
        self.semseg_dir = Path(semseg_dir)
        self.file_list = file_list
        self.train = train
        
        # Use provided transform or default
        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_train_transforms() if train else get_val_transforms()
        
        # Validate directories exist
        for dir_path, name in [
            (self.rgb_dir, 'RGB'),
            (self.depth_dir, 'Depth'),
            (self.sparse_dir, 'Sparse'),
            (self.semseg_dir, 'Semantic')
        ]:
            if not dir_path.exists():
                raise ValueError(f"{name} directory not found: {dir_path}")
        
        # Verify files exist
        self._validate_files()
    
    def _validate_files(self) -> None:
        """Verify all files in file_list exist in all directories."""
        missing = []
        for fname in self.file_list[:5]:  # Check first 5 files
            for dir_path in [self.rgb_dir, self.depth_dir, self.sparse_dir, self.semseg_dir]:
                if not (dir_path / fname).exists():
                    missing.append(str(dir_path / fname))
        
        if missing:
            raise FileNotFoundError(f"Missing files: {missing[:3]}...")
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
        
        Returns:
            Dictionary containing:
                - 'rgb': RGB image tensor [3, H, W]
                - 'depth': Ground truth depth tensor [1, H, W]
                - 'sparse': Sparse depth tensor [1, H, W]
                - 'semseg': Semantic segmentation tensor [1, H, W]
                - 'filename': Original filename
        """
        fname = self.file_list[idx]
        
        # Load images
        rgb = Image.open(self.rgb_dir / fname).convert('RGB')
        depth = Image.open(self.depth_dir / fname).convert('L')
        sparse = Image.open(self.sparse_dir / fname).convert('L')
        semseg = Image.open(self.semseg_dir / fname).convert('L')
        
        # Apply transforms
        sample = self.transform({
            'rgb': rgb,
            'depth': depth,
            'sparse': sparse,
            'semseg': semseg
        })
        
        sample['filename'] = fname
        return sample


def create_dataloaders(
    rgb_dir: str,
    depth_dir: str,
    sparse_dir: str,
    semseg_dir: str,
    batch_size: int = 4,
    train_split: float = 0.9,
    num_workers: int = 4,
    pin_memory: bool = True,
    image_size: Tuple[int, int] = (256, 256)
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        rgb_dir: Path to RGB images directory
        depth_dir: Path to depth maps directory
        sparse_dir: Path to sparse depth directory
        semseg_dir: Path to semantic segmentation directory
        batch_size: Batch size for training and validation
        train_split: Fraction of data to use for training (default: 0.9)
        num_workers: Number of data loading workers (default: 4)
        pin_memory: Whether to pin memory for faster GPU transfer (default: True)
        image_size: Target image size (default: (256, 256))
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Get all filenames
    all_filenames = sorted(os.listdir(rgb_dir))
    split_idx = int(train_split * len(all_filenames))
    
    train_files = all_filenames[:split_idx]
    val_files = all_filenames[split_idx:]
    
    print(f"Dataset split: {len(train_files)} training, {len(val_files)} validation")
    
    # Create transforms
    train_transform = get_train_transforms(size=image_size)
    val_transform = get_val_transforms(size=image_size)
    
    # Create datasets
    train_dataset = RGBDSegDataset(
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        sparse_dir=sparse_dir,
        semseg_dir=semseg_dir,
        file_list=train_files,
        transform=train_transform,
        train=True
    )
    
    val_dataset = RGBDSegDataset(
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        sparse_dir=sparse_dir,
        semseg_dir=semseg_dir,
        file_list=val_files,
        transform=val_transform,
        train=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader
