"""
Loss functions for depth estimation training.

This module provides various loss functions optimized for depth estimation,
including multi-scale losses, edge-aware losses, and combined loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class MultiScaleLoss(nn.Module):
    """
    Multi-scale loss for coarse-to-fine depth supervision.
    
    Computes weighted loss across multiple prediction scales, with higher
    weights typically assigned to finer scales.
    
    Args:
        weights: List of weights for each scale (finest to coarsest)
        base_loss: Base loss function ('l1' or 'l2')
        normalize: Whether to normalize depth values before loss computation
    
    Example:
        >>> criterion = MultiScaleLoss(weights=[1.0, 0.7, 0.5, 0.3, 0.2])
        >>> predictions = model(input)  # List of 5 predictions
        >>> loss = criterion(predictions, target)
    """
    
    def __init__(
        self,
        weights: Optional[List[float]] = None,
        base_loss: str = 'l1',
        normalize: bool = False
    ):
        super().__init__()
        self.weights = weights or [1.0, 0.7, 0.5, 0.3, 0.2]
        self.normalize = normalize
        
        if base_loss == 'l1':
            self.base_criterion = nn.L1Loss(reduction='mean')
        elif base_loss == 'l2':
            self.base_criterion = nn.MSELoss(reduction='mean')
        else:
            raise ValueError(f"Unknown base_loss: {base_loss}")
    
    def forward(
        self,
        predictions: List[torch.Tensor],
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute multi-scale loss.
        
        Args:
            predictions: List of depth predictions at different scales
            target: Ground truth depth tensor [B, 1, H, W]
        
        Returns:
            Weighted sum of losses across scales
        """
        total_loss = 0.0
        total_weight = 0.0
        
        for i, pred in enumerate(predictions):
            weight = self.weights[i] if i < len(self.weights) else self.weights[-1]
            
            # Resize target to match prediction size
            if pred.shape[2:] != target.shape[2:]:
                target_resized = F.interpolate(
                    target,
                    size=pred.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            else:
                target_resized = target
            
            # Normalize if requested
            if self.normalize:
                pred_norm = pred / (pred.mean() + 1e-6)
                target_norm = target_resized / (target_resized.mean() + 1e-6)
                scale_loss = self.base_criterion(pred_norm, target_norm)
            else:
                scale_loss = self.base_criterion(pred, target_resized)
            
            total_loss += weight * scale_loss
            total_weight += weight
        
        return total_loss / total_weight


class BerHuLoss(nn.Module):
    """
    Reverse Huber (BerHu) loss for depth estimation.
    
    BerHu loss behaves like L1 for small errors and L2 for large errors,
    making it robust to outliers while still penalizing large errors.
    
    Args:
        threshold: Threshold ratio for switching between L1 and L2 (default: 0.2)
    
    Reference:
        Eigen et al., "Depth Map Prediction from a Single Image using a 
        Multi-Scale Deep Network", NeurIPS 2014
    """
    
    def __init__(self, threshold: float = 0.2):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute BerHu loss.
        
        Args:
            pred: Predicted depth [B, 1, H, W]
            target: Ground truth depth [B, 1, H, W]
        
        Returns:
            BerHu loss value
        """
        diff = torch.abs(pred - target)
        c = self.threshold * diff.max().item()
        
        # L1 region (small errors)
        l1_mask = diff <= c
        l1_loss = diff[l1_mask]
        
        # L2 region (large errors)
        l2_mask = ~l1_mask
        l2_diff = diff[l2_mask]
        l2_loss = (l2_diff ** 2 + c ** 2) / (2 * c + 1e-6)
        
        # Combine
        total_elements = pred.numel()
        total_loss = (l1_loss.sum() + l2_loss.sum()) / total_elements
        
        return total_loss


class GradientLoss(nn.Module):
    """
    Gradient-based edge-aware loss for depth estimation.
    
    Encourages the model to preserve depth discontinuities at object boundaries
    by computing loss on image gradients.
    
    Args:
        edge_weight: Additional weight for edges (default: 1.0)
    """
    
    def __init__(self, edge_weight: float = 1.0):
        super().__init__()
        self.edge_weight = edge_weight
        
        # Sobel filters for gradient computation
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
    
    def compute_gradient(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute image gradients using Sobel filters."""
        if x.size(1) > 1:
            x = x.mean(dim=1, keepdim=True)
        
        grad_x = F.conv2d(x, self.sobel_x.to(x.device), padding=1)
        grad_y = F.conv2d(x, self.sobel_y.to(x.device), padding=1)
        
        return grad_x, grad_y
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient loss.
        
        Args:
            pred: Predicted depth [B, 1, H, W]
            target: Ground truth depth [B, 1, H, W]
        
        Returns:
            Gradient loss value
        """
        pred_grad_x, pred_grad_y = self.compute_gradient(pred)
        target_grad_x, target_grad_y = self.compute_gradient(target)
        
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        
        return (loss_x + loss_y) * self.edge_weight


class ScaleInvariantLoss(nn.Module):
    """
    Scale-invariant loss for depth estimation.
    
    This loss is invariant to the global scale of depth predictions,
    making it suitable when absolute depth values are less important.
    
    Args:
        lambda_var: Variance regularization weight (default: 0.5)
    
    Reference:
        Eigen et al., "Depth Map Prediction from a Single Image using a 
        Multi-Scale Deep Network", NeurIPS 2014
    """
    
    def __init__(self, lambda_var: float = 0.5):
        super().__init__()
        self.lambda_var = lambda_var
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute scale-invariant loss.
        
        Args:
            pred: Predicted depth [B, 1, H, W]
            target: Ground truth depth [B, 1, H, W]
        
        Returns:
            Scale-invariant loss value
        """
        # Avoid log(0)
        pred = pred.clamp(min=1e-6)
        target = target.clamp(min=1e-6)
        
        # Compute log difference
        log_diff = torch.log(pred) - torch.log(target)
        
        # Flatten spatial dimensions
        log_diff = log_diff.view(log_diff.size(0), -1)
        
        # Scale-invariant term
        n = log_diff.size(1)
        mse = (log_diff ** 2).mean(dim=1)
        var = (log_diff.sum(dim=1) ** 2) / (n ** 2)
        
        loss = (mse - self.lambda_var * var).mean()
        
        return loss


class CombinedDepthLoss(nn.Module):
    """
    Combined loss function for depth estimation.
    
    Combines multiple loss functions with configurable weights for
    comprehensive depth supervision.
    
    Args:
        l1_weight: Weight for L1 loss (default: 1.0)
        gradient_weight: Weight for gradient loss (default: 0.5)
        berhu_weight: Weight for BerHu loss (default: 0.0)
        scale_invariant_weight: Weight for scale-invariant loss (default: 0.0)
        multi_scale: Whether to use multi-scale supervision (default: True)
        scale_weights: Weights for different scales (default: [1.0, 0.7, 0.5, 0.3, 0.2])
    
    Example:
        >>> criterion = CombinedDepthLoss(l1_weight=1.0, gradient_weight=0.5)
        >>> loss = criterion(predictions, target)
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        gradient_weight: float = 0.5,
        berhu_weight: float = 0.0,
        scale_invariant_weight: float = 0.0,
        multi_scale: bool = True,
        scale_weights: Optional[List[float]] = None
    ):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.gradient_weight = gradient_weight
        self.berhu_weight = berhu_weight
        self.scale_invariant_weight = scale_invariant_weight
        self.multi_scale = multi_scale
        self.scale_weights = scale_weights or [1.0, 0.7, 0.5, 0.3, 0.2]
        
        self.l1_loss = nn.L1Loss()
        self.gradient_loss = GradientLoss() if gradient_weight > 0 else None
        self.berhu_loss = BerHuLoss() if berhu_weight > 0 else None
        self.si_loss = ScaleInvariantLoss() if scale_invariant_weight > 0 else None
    
    def _compute_loss_single_scale(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute combined loss for a single scale."""
        total_loss = 0.0
        
        if self.l1_weight > 0:
            total_loss += self.l1_weight * self.l1_loss(pred, target)
        
        if self.gradient_loss is not None:
            total_loss += self.gradient_weight * self.gradient_loss(pred, target)
        
        if self.berhu_loss is not None:
            total_loss += self.berhu_weight * self.berhu_loss(pred, target)
        
        if self.si_loss is not None:
            total_loss += self.scale_invariant_weight * self.si_loss(pred, target)
        
        return total_loss
    
    def forward(
        self,
        predictions: List[torch.Tensor],
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            predictions: List of predictions (or single prediction tensor)
            target: Ground truth depth [B, 1, H, W]
        
        Returns:
            Combined loss value
        """
        # Handle single prediction
        if isinstance(predictions, torch.Tensor):
            predictions = [predictions]
        
        if not self.multi_scale:
            # Only use finest scale
            return self._compute_loss_single_scale(predictions[0], target)
        
        # Multi-scale loss
        total_loss = 0.0
        total_weight = 0.0
        
        for i, pred in enumerate(predictions):
            weight = self.scale_weights[i] if i < len(self.scale_weights) else self.scale_weights[-1]
            
            # Resize target to match prediction size
            if pred.shape[2:] != target.shape[2:]:
                target_resized = F.interpolate(
                    target,
                    size=pred.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            else:
                target_resized = target
            
            scale_loss = self._compute_loss_single_scale(pred, target_resized)
            total_loss += weight * scale_loss
            total_weight += weight
        
        return total_loss / total_weight
