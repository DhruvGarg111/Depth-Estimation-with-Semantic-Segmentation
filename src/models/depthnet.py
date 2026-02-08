"""
DepthNet: Multi-modal Depth Estimation Network

A U-Net style encoder-decoder architecture for depth estimation that fuses
RGB images, sparse depth maps, and semantic segmentation information.

Architecture:
    - Encoder: 6-level convolutional network with GroupNorm and dropout
    - Decoder: Upsampling with skip connections from encoder
    - Multi-scale outputs: Depth predictions at 5 different resolutions

Input: 6-channel tensor [RGB (3) + Sparse Depth (1) + Semantic (2)]
Output: List of 5 depth maps at different scales (finest to coarsest)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Callable


class DepthNet(nn.Module):
    """
    DepthNet architecture for multi-modal depth estimation.
    
    This network takes multi-modal inputs (RGB + sparse depth + semantic segmentation)
    and produces dense depth maps at multiple scales for coarse-to-fine supervision.
    
    Args:
        batch_norm: Whether to use GroupNorm (8 groups) for normalization (default: False)
        with_confidence: Whether to output confidence maps alongside depth (default: False)
        clamp: Whether to clamp depth values (default: False)
        depth_activation: Activation for depth output - 'elu' or None (default: None)
        dropout: Dropout probability for regularization (default: 0.2)
        in_channels: Number of input channels (default: 6 for RGB+sparse+semantic)
    
    Example:
        >>> model = DepthNet(batch_norm=True, depth_activation='elu')
        >>> rgb = torch.randn(1, 3, 256, 256)
        >>> sparse_depth = torch.randn(1, 1, 256, 256)
        >>> semantic = torch.randn(1, 2, 256, 256)
        >>> x = torch.cat([rgb, sparse_depth, semantic], dim=1)
        >>> predictions = model(x)
        >>> print(predictions[0].shape)  # Finest scale
        torch.Size([1, 1, 256, 256])
    """
    
    def __init__(
        self,
        batch_norm: bool = False,
        with_confidence: bool = False,
        clamp: bool = False,
        depth_activation: Optional[str] = None,
        dropout: float = 0.2,
        in_channels: int = 6
    ):
        super().__init__()
        self.clamp = clamp
        self.with_confidence = with_confidence
        self.dropout = nn.Dropout2d(p=dropout)

        # Choose depth activation function
        if depth_activation == 'elu':
            self.depth_activation: Callable = lambda x: F.elu(x) + 1
        else:
            self.depth_activation = depth_activation or (lambda x: x)

        # Normalization function
        norm = (lambda c: nn.GroupNorm(8, c)) if batch_norm else (lambda c: nn.Identity())

        # ============== Encoder ==============
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            norm(32), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            norm(64), nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            norm(128), nn.ReLU(inplace=True)
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            norm(128), nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            norm(256), nn.ReLU(inplace=True)
        )
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            norm(256), nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2, padding=1, bias=False),
            norm(256), nn.ReLU(inplace=True)
        )
        self.conv5_1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            norm(256), nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            norm(512), nn.ReLU(inplace=True)
        )
        self.conv6_1 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            norm(512), nn.ReLU(inplace=True)
        )

        # ============== Decoder ==============
        def up(in_c: int, out_c: int) -> nn.Sequential:
            """Create upsampling block with bilinear interpolation."""
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                norm(out_c),
                nn.ReLU(inplace=True)
            )

        self.up5 = up(512, 256)
        self.up4 = up(512, 128)  # 256 from up5 + 256 from skip
        self.up3 = up(384, 64)   # 128 from up4 + 256 from skip
        self.up2 = up(192, 32)   # 64 from up3 + 128 from skip
        self.up1 = up(96, 32)    # 32 from up2 + 64 from skip

        # ============== Multi-scale Prediction Heads ==============
        out_channels = 2 if with_confidence else 1
        
        def pred(in_c: int) -> nn.Conv2d:
            """Create prediction head."""
            return nn.Conv2d(in_c, out_channels, 3, padding=1)

        self.pred5 = pred(256 + 256)  # After up5 + skip from conv5
        self.pred4 = pred(128 + 256)  # After up4 + skip from conv4
        self.pred3 = pred(64 + 128)   # After up3 + skip from conv3
        self.pred2 = pred(32 + 64)    # After up2 + skip from conv2
        self.pred1 = pred(32)         # Final prediction

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through DepthNet.
        
        Args:
            x: Input tensor of shape [B, 6, H, W]
               Channels: RGB (3) + Sparse Depth (1) + Semantic (2)
        
        Returns:
            List of 5 depth predictions from finest to coarsest scale:
            - predictions[0]: [B, 1, H, W] - Full resolution
            - predictions[1]: [B, 1, H/2, W/2]
            - predictions[2]: [B, 1, H/4, W/4]
            - predictions[3]: [B, 1, H/8, W/8]
            - predictions[4]: [B, 1, H/16, W/16]
        """
        # ============== Encoder ==============
        d1 = self.conv1(x)                      # [B, 32, H/2, W/2]
        d2 = self.conv2(self.dropout(d1))       # [B, 64, H/4, W/4]
        d3 = self.conv3(self.dropout(d2))       # [B, 128, H/8, W/8]
        d3 = self.conv3_1(d3)                   # [B, 128, H/8, W/8]
        d4 = self.conv4(self.dropout(d3))       # [B, 256, H/16, W/16]
        d4 = self.conv4_1(d4)                   # [B, 256, H/16, W/16]
        d5 = self.conv5(self.dropout(d4))       # [B, 256, H/32, W/32]
        d5 = self.conv5_1(d5)                   # [B, 256, H/32, W/32]
        b = self.conv6(self.dropout(d5))        # [B, 512, H/64, W/64]
        b = self.conv6_1(b)                     # [B, 512, H/64, W/64]

        # ============== Decoder with Skip Connections ==============
        u5 = self.up5(self.dropout(b))          # [B, 256, H/32, W/32]
        u5_cat = torch.cat([u5, d5], dim=1)     # [B, 512, H/32, W/32]
        p5 = self.pred5(u5_cat)

        u4 = self.up4(self.dropout(u5_cat))     # [B, 128, H/16, W/16]
        u4_cat = torch.cat([u4, d4], dim=1)     # [B, 384, H/16, W/16]
        p4 = self.pred4(u4_cat)

        u3 = self.up3(self.dropout(u4_cat))     # [B, 64, H/8, W/8]
        u3_cat = torch.cat([u3, d3], dim=1)     # [B, 192, H/8, W/8]
        p3 = self.pred3(u3_cat)

        u2 = self.up2(self.dropout(u3_cat))     # [B, 32, H/4, W/4]
        u2_cat = torch.cat([u2, d2], dim=1)     # [B, 96, H/4, W/4]
        p2 = self.pred2(u2_cat)

        u1 = self.up1(self.dropout(u2_cat))     # [B, 32, H/2, W/2]
        p1 = self.pred1(u1)                     # [B, 1, H/2, W/2]

        # Upsample to input resolution
        p1 = F.interpolate(p1, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Apply activation and optional clamping
        def post_process(d: torch.Tensor) -> torch.Tensor:
            d = self.depth_activation(d)
            return d.clamp(1e-3, 80) if self.clamp else d

        return [post_process(p1), post_process(p2), post_process(p3), 
                post_process(p4), post_process(p5)]

    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_pretrained(weights_path: str, device: str = 'cuda', **kwargs) -> DepthNet:
    """
    Load a pretrained DepthNet model.
    
    Args:
        weights_path: Path to the .pth weights file
        device: Device to load the model on ('cuda' or 'cpu')
        **kwargs: Additional arguments passed to DepthNet constructor
    
    Returns:
        Loaded DepthNet model in eval mode
    
    Example:
        >>> model = load_pretrained('depthnet_final.pth', device='cuda')
        >>> model.eval()
    """
    model = DepthNet(**kwargs)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model
