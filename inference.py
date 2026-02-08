"""
Inference script for DepthNet depth estimation model.

Usage:
    # Single image inference
    python inference.py --image path/to/image.jpg --weights depthnet_final.pth
    
    # Batch inference on a directory
    python inference.py --input_dir path/to/images --weights depthnet_final.pth --output_dir results
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import DepthNet
from src.utils.visualization import depth_to_colormap, denormalize


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run depth estimation inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input arguments
    parser.add_argument('--image', type=str, default=None,
                        help='Path to single input image')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Directory containing input images')
    parser.add_argument('--sparse_depth', type=str, default=None,
                        help='Path to sparse depth map (optional)')
    parser.add_argument('--semantic', type=str, default=None,
                        help='Path to semantic segmentation map (optional)')
    
    # Model arguments
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights file')
    parser.add_argument('--batch_norm', action='store_true', default=True,
                        help='Model was trained with batch norm')
    parser.add_argument('--depth_activation', type=str, default='elu',
                        help='Depth activation function used in training')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./inference_outputs',
                        help='Directory to save outputs')
    parser.add_argument('--save_colormap', action='store_true', default=True,
                        help='Save colorized depth map')
    parser.add_argument('--save_raw', action='store_true', default=False,
                        help='Save raw depth values as .npy')
    parser.add_argument('--colormap', type=str, default='plasma',
                        help='Matplotlib colormap for visualization')
    
    # Processing arguments
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256],
                        help='Input image size (height, width)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for inference')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for directory processing')
    
    return parser.parse_args()


def load_model(
    weights_path: str,
    device: torch.device,
    batch_norm: bool = True,
    depth_activation: Optional[str] = 'elu'
) -> DepthNet:
    """Load pretrained model."""
    model = DepthNet(
        batch_norm=batch_norm,
        depth_activation=depth_activation if depth_activation != 'none' else None
    )
    
    state_dict = torch.load(weights_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f'Loaded model from {weights_path}')
    return model


def preprocess_image(
    image_path: str,
    size: Tuple[int, int],
    device: torch.device
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Load and preprocess an image for inference.
    
    Returns:
        Preprocessed tensor and original image size
    """
    image = Image.open(image_path).convert('RGB')
    original_size = image.size[::-1]  # (height, width)
    
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor, original_size


def preprocess_depth_map(
    depth_path: str,
    size: Tuple[int, int],
    device: torch.device
) -> torch.Tensor:
    """Load and preprocess a depth/segmentation map."""
    depth = Image.open(depth_path).convert('L')
    
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    
    tensor = transform(depth).unsqueeze(0).to(device)
    return tensor


def create_dummy_inputs(
    batch_size: int,
    size: Tuple[int, int],
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create dummy sparse depth and semantic inputs."""
    sparse = torch.zeros(batch_size, 1, size[0], size[1], device=device)
    semantic = torch.zeros(batch_size, 1, size[0], size[1], device=device)
    return sparse, semantic


@torch.no_grad()
def predict_depth(
    model: DepthNet,
    rgb: torch.Tensor,
    sparse_depth: Optional[torch.Tensor] = None,
    semantic: Optional[torch.Tensor] = None,
    original_size: Optional[Tuple[int, int]] = None
) -> torch.Tensor:
    """
    Run depth prediction.
    
    Args:
        model: DepthNet model
        rgb: RGB tensor [B, 3, H, W]
        sparse_depth: Optional sparse depth [B, 1, H, W]
        semantic: Optional semantic map [B, 1, H, W]
        original_size: Optional size to resize output to
    
    Returns:
        Predicted depth tensor
    """
    device = rgb.device
    batch_size = rgb.size(0)
    size = (rgb.size(2), rgb.size(3))
    
    # Create inputs if not provided
    if sparse_depth is None:
        sparse_depth = torch.zeros(batch_size, 1, size[0], size[1], device=device)
    if semantic is None:
        semantic = torch.zeros(batch_size, 1, size[0], size[1], device=device)
    
    # Create 6-channel input
    input_tensor = torch.cat([rgb, sparse_depth, semantic, semantic], dim=1)
    
    # Forward pass
    predictions = model(input_tensor)
    depth = predictions[0]  # Finest scale
    
    # Resize to original size if specified
    if original_size is not None and depth.shape[2:] != original_size:
        depth = F.interpolate(
            depth,
            size=original_size,
            mode='bilinear',
            align_corners=False
        )
    
    return depth


def save_depth_output(
    depth: torch.Tensor,
    output_path: str,
    colormap: str = 'plasma',
    save_raw: bool = False
) -> None:
    """Save depth prediction to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save colorized depth
    depth_np = depth.squeeze().cpu().numpy()
    depth_colored = depth_to_colormap(depth.squeeze(), colormap)
    
    # Convert to uint8 and save
    depth_colored_uint8 = (depth_colored * 255).astype(np.uint8)
    Image.fromarray(depth_colored_uint8).save(str(output_path))
    print(f'Saved colorized depth to {output_path}')
    
    # Save raw depth values
    if save_raw:
        raw_path = output_path.with_suffix('.npy')
        np.save(str(raw_path), depth_np)
        print(f'Saved raw depth to {raw_path}')


def create_comparison_output(
    rgb_path: str,
    depth: torch.Tensor,
    output_path: str,
    colormap: str = 'plasma'
) -> None:
    """Create side-by-side comparison of input and output."""
    # Load RGB
    rgb = Image.open(rgb_path).convert('RGB')
    rgb_np = np.array(rgb.resize((depth.size(3), depth.size(2))))
    
    # Get colorized depth
    depth_colored = depth_to_colormap(depth.squeeze(), colormap)
    depth_colored_uint8 = (depth_colored * 255).astype(np.uint8)
    
    # Create side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(rgb_np)
    axes[0].set_title('Input RGB', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(depth_colored_uint8)
    axes[1].set_title('Predicted Depth', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved comparison to {output_path}')


def process_single_image(
    model: DepthNet,
    image_path: str,
    args: argparse.Namespace,
    device: torch.device
) -> torch.Tensor:
    """Process a single image."""
    # Load and preprocess RGB
    rgb, original_size = preprocess_image(
        image_path,
        tuple(args.image_size),
        device
    )
    
    # Load optional inputs
    sparse_depth = None
    semantic = None
    
    if args.sparse_depth:
        sparse_depth = preprocess_depth_map(args.sparse_depth, tuple(args.image_size), device)
    if args.semantic:
        semantic = preprocess_depth_map(args.semantic, tuple(args.image_size), device)
    
    # Predict
    depth = predict_depth(model, rgb, sparse_depth, semantic, original_size)
    
    return depth


def process_directory(
    model: DepthNet,
    input_dir: str,
    args: argparse.Namespace,
    device: torch.device
) -> None:
    """Process all images in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_paths = [
        p for p in input_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ]
    
    print(f'Found {len(image_paths)} images in {input_dir}')
    
    for image_path in image_paths:
        print(f'\nProcessing {image_path.name}...')
        
        # Process image
        depth = process_single_image(model, str(image_path), args, device)
        
        # Save outputs
        output_name = image_path.stem + '_depth.png'
        output_path = output_dir / output_name
        
        if args.save_colormap:
            save_depth_output(depth, str(output_path), args.colormap, args.save_raw)
        
        # Save comparison
        comparison_path = output_dir / (image_path.stem + '_comparison.png')
        create_comparison_output(str(image_path), depth, str(comparison_path), args.colormap)


def main():
    """Main inference function."""
    args = parse_args()
    
    # Validate inputs
    if args.image is None and args.input_dir is None:
        raise ValueError('Must specify either --image or --input_dir')
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    model = load_model(
        args.weights,
        device,
        batch_norm=args.batch_norm,
        depth_activation=args.depth_activation
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.image:
        # Single image mode
        print(f'\nProcessing single image: {args.image}')
        depth = process_single_image(model, args.image, args, device)
        
        # Save outputs
        image_name = Path(args.image).stem
        output_path = output_dir / f'{image_name}_depth.png'
        
        if args.save_colormap:
            save_depth_output(depth, str(output_path), args.colormap, args.save_raw)
        
        # Save comparison
        comparison_path = output_dir / f'{image_name}_comparison.png'
        create_comparison_output(args.image, depth, str(comparison_path), args.colormap)
        
    else:
        # Directory mode
        process_directory(model, args.input_dir, args, device)
    
    print(f'\nInference complete! Results saved to {output_dir}')


if __name__ == '__main__':
    main()
