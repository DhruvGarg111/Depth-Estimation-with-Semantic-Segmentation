"""
Depth estimation evaluation metrics.

This module provides standard metrics for evaluating depth estimation quality,
following conventions from the depth estimation literature.

Metrics included:
- Absolute Relative Error (AbsRel)
- Squared Relative Error (SqRel)
- Root Mean Squared Error (RMSE)
- RMSE in log space (RMSElog)
- Threshold accuracies (δ < 1.25, 1.25², 1.25³)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from dataclasses import dataclass, field


@dataclass
class DepthMetrics:
    """
    Container for depth estimation metrics.
    
    Attributes:
        abs_rel: Absolute relative error
        sq_rel: Squared relative error
        rmse: Root mean squared error
        rmse_log: RMSE in log space
        delta_1: Accuracy under threshold 1.25
        delta_2: Accuracy under threshold 1.25²
        delta_3: Accuracy under threshold 1.25³
        count: Number of samples aggregated
    """
    abs_rel: float = 0.0
    sq_rel: float = 0.0
    rmse: float = 0.0
    rmse_log: float = 0.0
    delta_1: float = 0.0
    delta_2: float = 0.0
    delta_3: float = 0.0
    count: int = 0
    
    def update(self, other: 'DepthMetrics') -> None:
        """Aggregate metrics from another DepthMetrics instance."""
        total = self.count + other.count
        if total == 0:
            return
        
        # Weighted average
        w1 = self.count / total if total > 0 else 0
        w2 = other.count / total if total > 0 else 0
        
        self.abs_rel = w1 * self.abs_rel + w2 * other.abs_rel
        self.sq_rel = w1 * self.sq_rel + w2 * other.sq_rel
        self.rmse = w1 * self.rmse + w2 * other.rmse
        self.rmse_log = w1 * self.rmse_log + w2 * other.rmse_log
        self.delta_1 = w1 * self.delta_1 + w2 * other.delta_1
        self.delta_2 = w1 * self.delta_2 + w2 * other.delta_2
        self.delta_3 = w1 * self.delta_3 + w2 * other.delta_3
        self.count = total
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            'abs_rel': self.abs_rel,
            'sq_rel': self.sq_rel,
            'rmse': self.rmse,
            'rmse_log': self.rmse_log,
            'delta_1': self.delta_1,
            'delta_2': self.delta_2,
            'delta_3': self.delta_3
        }
    
    def __str__(self) -> str:
        """Pretty print metrics."""
        return (
            f"Depth Metrics:\n"
            f"  AbsRel: {self.abs_rel:.4f}\n"
            f"  SqRel:  {self.sq_rel:.4f}\n"
            f"  RMSE:   {self.rmse:.4f}\n"
            f"  RMSElog:{self.rmse_log:.4f}\n"
            f"  δ<1.25: {self.delta_1:.4f}\n"
            f"  δ<1.25²:{self.delta_2:.4f}\n"
            f"  δ<1.25³:{self.delta_3:.4f}"
        )


def compute_depth_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    min_depth: float = 1e-3,
    max_depth: float = 80.0
) -> DepthMetrics:
    """
    Compute standard depth estimation metrics.
    
    Args:
        pred: Predicted depth tensor [B, 1, H, W] or [B, H, W]
        target: Ground truth depth tensor [B, 1, H, W] or [B, H, W]
        mask: Optional valid pixel mask (True for valid pixels)
        min_depth: Minimum valid depth value (default: 1e-3)
        max_depth: Maximum valid depth value (default: 80.0)
    
    Returns:
        DepthMetrics containing all computed metrics
    
    Example:
        >>> pred = model(input)
        >>> metrics = compute_depth_metrics(pred, ground_truth)
        >>> print(f"RMSE: {metrics.rmse:.4f}")
    """
    # Ensure 4D tensors
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    
    # Remove channel dimension for computation
    pred = pred.squeeze(1)
    target = target.squeeze(1)
    
    # Create validity mask
    if mask is None:
        mask = (target > min_depth) & (target < max_depth)
    
    # Apply mask
    pred = pred[mask]
    target = target[mask]
    
    if pred.numel() == 0:
        return DepthMetrics()
    
    # Clamp predictions to valid range
    pred = pred.clamp(min_depth, max_depth)
    
    # Compute metrics
    thresh = torch.max(pred / target, target / pred)
    
    abs_rel = torch.mean(torch.abs(pred - target) / target).item()
    sq_rel = torch.mean(((pred - target) ** 2) / target).item()
    rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()
    
    # RMSE in log space (avoid log(0))
    pred_log = torch.log(pred.clamp(min=1e-6))
    target_log = torch.log(target.clamp(min=1e-6))
    rmse_log = torch.sqrt(torch.mean((pred_log - target_log) ** 2)).item()
    
    # Threshold accuracies
    delta_1 = (thresh < 1.25).float().mean().item()
    delta_2 = (thresh < 1.25 ** 2).float().mean().item()
    delta_3 = (thresh < 1.25 ** 3).float().mean().item()
    
    return DepthMetrics(
        abs_rel=abs_rel,
        sq_rel=sq_rel,
        rmse=rmse,
        rmse_log=rmse_log,
        delta_1=delta_1,
        delta_2=delta_2,
        delta_3=delta_3,
        count=pred.numel()
    )


class MetricsTracker:
    """
    Track and aggregate metrics over multiple batches.
    
    Example:
        >>> tracker = MetricsTracker()
        >>> for batch in dataloader:
        ...     pred = model(batch)
        ...     tracker.update(pred, batch['depth'])
        >>> print(tracker.get_metrics())
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self.metrics = DepthMetrics()
    
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> None:
        """Update metrics with a new batch."""
        batch_metrics = compute_depth_metrics(pred, target, mask)
        self.metrics.update(batch_metrics)
    
    def get_metrics(self) -> DepthMetrics:
        """Get aggregated metrics."""
        return self.metrics
    
    def get_dict(self) -> Dict[str, float]:
        """Get metrics as dictionary."""
        return self.metrics.to_dict()
