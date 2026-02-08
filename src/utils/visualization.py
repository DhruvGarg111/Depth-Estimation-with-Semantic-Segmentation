"""
Visualization utilities for depth estimation results.

This module provides functions for creating publication-quality visualizations
of depth estimation results, including side-by-side comparisons and depth colormaps.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, List, Tuple, Union
from pathlib import Path
from PIL import Image
from torchvision.transforms import functional as TF


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
        Denormalized tensor clamped to [0, 1]
    """
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor.clamp(0, 1)


def depth_to_colormap(
    depth: torch.Tensor,
    colormap: str = 'plasma',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    invalid_mask: Optional[torch.Tensor] = None
) -> np.ndarray:
    """
    Convert depth tensor to colormap image.
    
    Args:
        depth: Depth tensor [1, H, W] or [H, W]
        colormap: Matplotlib colormap name (default: 'plasma')
        vmin: Minimum depth value for normalization
        vmax: Maximum depth value for normalization
        invalid_mask: Mask for invalid pixels (will be shown as black)
    
    Returns:
        RGB numpy array [H, W, 3] with values in [0, 1]
    """
    if depth.dim() == 3:
        depth = depth.squeeze(0)
    
    depth_np = depth.cpu().numpy()
    
    if vmin is None:
        vmin = np.percentile(depth_np[depth_np > 0], 5) if np.any(depth_np > 0) else 0
    if vmax is None:
        vmax = np.percentile(depth_np, 95)
    
    # Normalize to [0, 1]
    depth_normalized = (depth_np - vmin) / (vmax - vmin + 1e-8)
    depth_normalized = np.clip(depth_normalized, 0, 1)
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    depth_colored = cmap(depth_normalized)[:, :, :3]
    
    # Apply invalid mask
    if invalid_mask is not None:
        invalid_np = invalid_mask.cpu().numpy()
        if invalid_np.ndim == 3:
            invalid_np = invalid_np.squeeze(0)
        depth_colored[invalid_np] = 0
    
    return depth_colored


def create_comparison_figure(
    rgb: torch.Tensor,
    gt_depth: torch.Tensor,
    pred_depth: torch.Tensor,
    sparse_depth: Optional[torch.Tensor] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 4),
    colormap: str = 'plasma',
    save_path: Optional[str] = None
) -> Figure:
    """
    Create a side-by-side comparison figure.
    
    Args:
        rgb: RGB image tensor [3, H, W]
        gt_depth: Ground truth depth [1, H, W]
        pred_depth: Predicted depth [1, H, W]
        sparse_depth: Optional sparse depth input [1, H, W]
        title: Figure title
        figsize: Figure size
        colormap: Depth colormap
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    num_cols = 4 if sparse_depth is not None else 3
    fig, axes = plt.subplots(1, num_cols, figsize=figsize)
    
    # RGB Image
    rgb_denorm = denormalize(rgb)
    rgb_np = rgb_denorm.permute(1, 2, 0).cpu().numpy()
    axes[0].imshow(rgb_np)
    axes[0].set_title('RGB Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    col_idx = 1
    
    # Sparse Depth (if provided)
    if sparse_depth is not None:
        sparse_colored = depth_to_colormap(sparse_depth, colormap)
        axes[col_idx].imshow(sparse_colored)
        axes[col_idx].set_title('Sparse Depth Input', fontsize=12, fontweight='bold')
        axes[col_idx].axis('off')
        col_idx += 1
    
    # Get common depth range for consistent coloring
    gt_np = gt_depth.squeeze().cpu().numpy()
    pred_np = pred_depth.squeeze().cpu().numpy()
    vmin = min(gt_np.min(), pred_np.min())
    vmax = max(gt_np.max(), pred_np.max())
    
    # Ground Truth Depth
    gt_colored = depth_to_colormap(gt_depth, colormap, vmin, vmax)
    axes[col_idx].imshow(gt_colored)
    axes[col_idx].set_title('Ground Truth Depth', fontsize=12, fontweight='bold')
    axes[col_idx].axis('off')
    col_idx += 1
    
    # Predicted Depth
    pred_colored = depth_to_colormap(pred_depth, colormap, vmin, vmax)
    axes[col_idx].imshow(pred_colored)
    axes[col_idx].set_title('Predicted Depth', fontsize=12, fontweight='bold')
    axes[col_idx].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        print(f"Saved figure to {save_path}")
    
    return fig


def visualize_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 4,
    save_dir: Optional[str] = None,
    colormap: str = 'plasma'
) -> List[Figure]:
    """
    Visualize model predictions on samples from a dataloader.
    
    Args:
        model: Trained model
        dataloader: DataLoader with samples
        device: Device to run inference on
        num_samples: Number of samples to visualize
        save_dir: Directory to save figures (optional)
        colormap: Depth colormap
    
    Returns:
        List of matplotlib figures
    """
    model.eval()
    figures = []
    
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_samples:
                break
            
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            sparse = batch['sparse'].to(device)
            semseg = batch['semseg'].to(device)
            
            # Create input tensor [B, 6, H, W]
            # Use semantic segmentation channel twice to match expected input
            input_tensor = torch.cat([rgb, sparse, semseg, semseg], dim=1)
            
            # Get predictions
            predictions = model(input_tensor)
            pred_depth = predictions[0] if isinstance(predictions, (list, tuple)) else predictions
            
            # Create visualization for each sample in batch
            batch_size = rgb.size(0)
            for i in range(min(batch_size, num_samples - len(figures))):
                save_path = None
                if save_dir:
                    save_path = str(Path(save_dir) / f"prediction_{batch_idx}_{i}.png")
                
                fig = create_comparison_figure(
                    rgb=rgb[i],
                    gt_depth=depth[i],
                    pred_depth=pred_depth[i],
                    sparse_depth=sparse[i],
                    title=f"Sample {batch_idx * batch_size + i}",
                    colormap=colormap,
                    save_path=save_path
                )
                figures.append(fig)
                
                if len(figures) >= num_samples:
                    break
    
    return figures


def save_prediction_grid(
    predictions: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    save_path: str,
    cols: int = 4,
    colormap: str = 'plasma',
    figsize_per_image: Tuple[float, float] = (4, 3)
) -> None:
    """
    Save a grid of predictions to a single image.
    
    Args:
        predictions: List of (rgb, gt_depth, pred_depth) tuples
        save_path: Path to save the grid image
        cols: Number of columns in the grid
        colormap: Depth colormap
        figsize_per_image: Size per image in the grid
    """
    n = len(predictions)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(
        rows * 3, cols,  # 3 rows per sample (RGB, GT, Pred)
        figsize=(figsize_per_image[0] * cols, figsize_per_image[1] * rows * 3)
    )
    
    if rows * cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(-1, max(rows * 3, cols))
    
    for idx, (rgb, gt_depth, pred_depth) in enumerate(predictions):
        col = idx % cols
        row_base = (idx // cols) * 3
        
        # RGB
        rgb_denorm = denormalize(rgb)
        rgb_np = rgb_denorm.permute(1, 2, 0).cpu().numpy()
        axes[row_base, col].imshow(rgb_np)
        axes[row_base, col].set_title('RGB')
        axes[row_base, col].axis('off')
        
        # GT Depth
        gt_colored = depth_to_colormap(gt_depth, colormap)
        axes[row_base + 1, col].imshow(gt_colored)
        axes[row_base + 1, col].set_title('Ground Truth')
        axes[row_base + 1, col].axis('off')
        
        # Predicted Depth
        pred_colored = depth_to_colormap(pred_depth, colormap)
        axes[row_base + 2, col].imshow(pred_colored)
        axes[row_base + 2, col].set_title('Predicted')
        axes[row_base + 2, col].axis('off')
    
    # Hide empty subplots
    for idx in range(n, rows * cols):
        col = idx % cols
        for row_offset in range(3):
            row = (idx // cols) * 3 + row_offset
            if row < len(axes) and col < len(axes[0]):
                axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved prediction grid to {save_path}")


def create_error_map(
    gt_depth: torch.Tensor,
    pred_depth: torch.Tensor,
    error_type: str = 'absolute',
    colormap: str = 'hot'
) -> np.ndarray:
    """
    Create an error visualization map.
    
    Args:
        gt_depth: Ground truth depth [1, H, W] or [H, W]
        pred_depth: Predicted depth [1, H, W] or [H, W]
        error_type: 'absolute' or 'relative'
        colormap: Matplotlib colormap for error visualization
    
    Returns:
        RGB numpy array [H, W, 3]
    """
    if gt_depth.dim() == 3:
        gt_depth = gt_depth.squeeze(0)
    if pred_depth.dim() == 3:
        pred_depth = pred_depth.squeeze(0)
    
    gt_np = gt_depth.cpu().numpy()
    pred_np = pred_depth.cpu().numpy()
    
    if error_type == 'absolute':
        error = np.abs(gt_np - pred_np)
        vmax = np.percentile(error, 95)
    else:  # relative
        error = np.abs(gt_np - pred_np) / (gt_np + 1e-6)
        vmax = 1.0
    
    error_normalized = np.clip(error / vmax, 0, 1)
    
    cmap = plt.get_cmap(colormap)
    error_colored = cmap(error_normalized)[:, :, :3]
    
    return error_colored
