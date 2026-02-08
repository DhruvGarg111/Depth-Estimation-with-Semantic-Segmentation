"""
Data augmentation and transformation utilities for depth estimation.

This module provides transform pipelines for training and validation,
including geometric and photometric augmentations.
"""

import random
from typing import Dict, Tuple, Any
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF


class SynchronizedTransform:
    """
    Apply synchronized geometric transforms to all input modalities.
    
    Ensures that RGB, depth, sparse depth, and semantic maps are transformed
    identically to maintain spatial correspondence.
    
    Args:
        size: Target size (height, width) for resizing
        train: Whether to apply training augmentations (default: True)
        horizontal_flip_prob: Probability of horizontal flip (default: 0.5)
        vertical_flip_prob: Probability of vertical flip (default: 0.0)
        color_jitter: Whether to apply color jitter to RGB (default: True)
        brightness: Brightness jitter range (default: 0.2)
        contrast: Contrast jitter range (default: 0.2)
        saturation: Saturation jitter range (default: 0.1)
    """
    
    def __init__(
        self,
        size: Tuple[int, int] = (256, 256),
        train: bool = True,
        horizontal_flip_prob: float = 0.5,
        vertical_flip_prob: float = 0.0,
        color_jitter: bool = True,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.1
    ):
        self.size = size
        self.train = train
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.color_jitter = color_jitter
        
        self.resize = transforms.Resize(size)
        self.to_tensor = transforms.ToTensor()
        self.norm_rgb = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        
        if color_jitter:
            self.color_transform = transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation
            )
        else:
            self.color_transform = None
    
    def __call__(self, sample: Dict[str, Image.Image]) -> Dict[str, torch.Tensor]:
        """
        Apply transforms to a sample dictionary.
        
        Args:
            sample: Dictionary containing PIL Images with keys:
                   'rgb', 'depth', 'sparse', 'semseg'
        
        Returns:
            Dictionary of transformed tensors
        """
        rgb = sample['rgb']
        depth = sample['depth']
        sparse = sample['sparse']
        semseg = sample['semseg']
        
        # Resize all images
        rgb = self.resize(rgb)
        depth = self.resize(depth)
        sparse = self.resize(sparse)
        semseg = self.resize(semseg)
        
        if self.train:
            # Synchronized horizontal flip
            if random.random() < self.horizontal_flip_prob:
                rgb = TF.hflip(rgb)
                depth = TF.hflip(depth)
                sparse = TF.hflip(sparse)
                semseg = TF.hflip(semseg)
            
            # Synchronized vertical flip
            if random.random() < self.vertical_flip_prob:
                rgb = TF.vflip(rgb)
                depth = TF.vflip(depth)
                sparse = TF.vflip(sparse)
                semseg = TF.vflip(semseg)
            
            # Color jitter (RGB only)
            if self.color_transform is not None:
                rgb = self.color_transform(rgb)
        
        # Convert to tensors
        rgb = self.norm_rgb(self.to_tensor(rgb))
        depth = self.to_tensor(depth)
        sparse = self.to_tensor(sparse)
        semseg = self.to_tensor(semseg)
        
        return {
            'rgb': rgb,
            'depth': depth,
            'sparse': sparse,
            'semseg': semseg
        }


def get_train_transforms(
    size: Tuple[int, int] = (256, 256),
    horizontal_flip_prob: float = 0.5,
    color_jitter: bool = True
) -> SynchronizedTransform:
    """
    Get training transforms with augmentation.
    
    Args:
        size: Target image size
        horizontal_flip_prob: Probability of horizontal flip
        color_jitter: Whether to apply color jitter
    
    Returns:
        SynchronizedTransform configured for training
    """
    return SynchronizedTransform(
        size=size,
        train=True,
        horizontal_flip_prob=horizontal_flip_prob,
        color_jitter=color_jitter
    )


def get_val_transforms(size: Tuple[int, int] = (256, 256)) -> SynchronizedTransform:
    """
    Get validation/test transforms (no augmentation).
    
    Args:
        size: Target image size
    
    Returns:
        SynchronizedTransform configured for validation
    """
    return SynchronizedTransform(
        size=size,
        train=False,
        horizontal_flip_prob=0.0,
        color_jitter=False
    )


def denormalize(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
) -> torch.Tensor:
    """
    Denormalize a tensor for visualization.
    
    Args:
        tensor: Normalized tensor [C, H, W]
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Denormalized tensor
    """
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor.clamp(0, 1)
