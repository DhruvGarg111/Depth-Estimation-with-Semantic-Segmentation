"""
Custom neural network layers for DepthNet architecture.

This module contains reusable layer definitions including:
- Convolutional blocks with optional batch normalization
- Transposed convolution (deconvolution) blocks
- Depth prediction heads
- Utility functions for tensor operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_


def conv(in_planes: int, out_planes: int, stride: int = 1, batch_norm: bool = False) -> nn.Sequential:
    """
    Create a convolutional block with optional instance normalization.
    
    Args:
        in_planes: Number of input channels
        out_planes: Number of output channels
        stride: Convolution stride (default: 1)
        batch_norm: Whether to use instance normalization (default: False)
    
    Returns:
        Sequential block containing Conv2d, optional InstanceNorm2d, and ReLU
    """
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm2d(out_planes, eps=1e-3, affine=True),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )


def deconv(in_planes: int, out_planes: int, batch_norm: bool = False) -> nn.Sequential:
    """
    Create a transposed convolution (upsampling) block with optional instance normalization.
    
    Args:
        in_planes: Number of input channels
        out_planes: Number of output channels
        batch_norm: Whether to use instance normalization (default: False)
    
    Returns:
        Sequential block containing ConvTranspose2d, Conv2d, optional InstanceNorm2d, and ReLU
    """
    if batch_norm:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_planes, eps=1e-3, affine=True),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )


def predict_depth(in_planes: int, with_confidence: bool = False) -> nn.Conv2d:
    """
    Create a depth prediction head.
    
    Args:
        in_planes: Number of input channels
        with_confidence: Whether to output confidence map alongside depth (default: False)
    
    Returns:
        Conv2d layer that outputs 1 or 2 channels
    """
    return nn.Conv2d(in_planes, 2 if with_confidence else 1, kernel_size=3, stride=1, padding=1, bias=True)


def post_process_depth(depth: torch.Tensor, activation_function=None, clamp: bool = False) -> torch.Tensor:
    """
    Post-process depth predictions with optional activation and clamping.
    
    Args:
        depth: Raw depth predictions tensor
        activation_function: Optional activation function to apply
        clamp: Whether to clamp values to [10, 60] range (default: False)
    
    Returns:
        Post-processed depth tensor
    """
    if activation_function is not None:
        depth = activation_function(depth)

    if clamp:
        depth = depth.clamp(10, 60)

    return depth


def adaptative_cat(out_conv: torch.Tensor, out_deconv: torch.Tensor, out_depth_up: torch.Tensor) -> torch.Tensor:
    """
    Adaptively concatenate tensors of potentially different spatial sizes.
    
    Crops deconv and depth_up tensors to match conv tensor size before concatenation.
    
    Args:
        out_conv: Encoder feature tensor (reference size)
        out_deconv: Decoder feature tensor
        out_depth_up: Upsampled depth tensor
    
    Returns:
        Concatenated tensor along channel dimension
    """
    out_deconv = out_deconv[:, :, :out_conv.size(2), :out_conv.size(3)]
    out_depth_up = out_depth_up[:, :, :out_conv.size(2), :out_conv.size(3)]
    return torch.cat((out_conv, out_deconv, out_depth_up), 1)


def init_modules(net: nn.Module) -> None:
    """
    Initialize network weights using Xavier initialization.
    
    Args:
        net: Neural network module to initialize
    """
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            xavier_normal_(m.weight)
            if m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            constant_(m.weight, 1)
            constant_(m.bias, 0)
