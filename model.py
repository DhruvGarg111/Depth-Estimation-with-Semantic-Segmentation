"""
DepthNet Model - Standalone Import

This module provides a simple way to import and use the DepthNet model
without needing to import from the src package.

Usage:
    from model import DepthNet, load_pretrained
    
    # Create a new model
    model = DepthNet(batch_norm=True, depth_activation='elu')
    
    # Or load pretrained weights
    model = load_pretrained('depthnet_final.pth')
"""

# Re-export from src.models
from src.models.depthnet import DepthNet, load_pretrained

__all__ = ['DepthNet', 'load_pretrained']
