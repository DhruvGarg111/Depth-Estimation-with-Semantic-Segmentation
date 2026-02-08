"""
Training script for DepthNet depth estimation model.

Usage:
    python train.py --config configs/default.yaml
    python train.py --data_dir /path/to/data --epochs 200 --batch_size 4
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import DepthNet
from src.data import create_dataloaders
from src.utils import (
    CombinedDepthLoss,
    MetricsTracker,
    visualize_predictions,
    compute_depth_metrics
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train DepthNet for depth estimation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Root directory containing the dataset')
    parser.add_argument('--rgb_dir', type=str, default=None,
                        help='RGB images directory (default: data_dir/images)')
    parser.add_argument('--depth_dir', type=str, default=None,
                        help='Depth maps directory (default: data_dir/depths)')
    parser.add_argument('--sparse_dir', type=str, default=None,
                        help='Sparse depth directory (default: data_dir/labels)')
    parser.add_argument('--semseg_dir', type=str, default=None,
                        help='Semantic segmentation directory (default: data_dir/instances)')
    
    # Model arguments
    parser.add_argument('--batch_norm', action='store_true', default=True,
                        help='Use GroupNorm in the model')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout probability')
    parser.add_argument('--depth_activation', type=str, default='elu',
                        choices=['elu', 'none'],
                        help='Depth activation function')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping norm (0 to disable)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Use automatic mixed precision training')
    
    # Loss arguments
    parser.add_argument('--l1_weight', type=float, default=1.0,
                        help='Weight for L1 loss')
    parser.add_argument('--gradient_weight', type=float, default=0.5,
                        help='Weight for gradient loss')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Directory to save outputs')
    parser.add_argument('--save_every', type=int, default=50,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_every', type=int, default=10,
                        help='Log metrics every N batches')
    parser.add_argument('--vis_every', type=int, default=10,
                        help='Visualize predictions every N epochs')
    
    # Other arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on')
    
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    scaler: Optional[GradScaler] = None
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Returns:
        Dictionary of training metrics
    """
    model.train()
    metrics_tracker = MetricsTracker()
    running_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}')
    
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        rgb = batch['rgb'].to(device)
        depth = batch['depth'].to(device)
        sparse = batch['sparse'].to(device)
        semseg = batch['semseg'].to(device)
        
        # Create 6-channel input: RGB (3) + Sparse (1) + Semantic (2)
        # Actually USE the semantic segmentation instead of dummy zeros
        input_tensor = torch.cat([rgb, sparse, semseg, semseg], dim=1)
        
        optimizer.zero_grad()
        
        # Forward pass with optional AMP
        if scaler is not None:
            with autocast():
                predictions = model(input_tensor)
                loss = criterion(predictions, depth)
            
            scaler.scale(loss).backward()
            
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(input_tensor)
            loss = criterion(predictions, depth)
            loss.backward()
            
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
        
        # Update metrics
        running_loss += loss.item()
        num_batches += 1
        
        # Compute depth metrics on finest scale
        with torch.no_grad():
            pred_depth = predictions[0] if isinstance(predictions, (list, tuple)) else predictions
            metrics_tracker.update(pred_depth, depth)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss / num_batches:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
    
    avg_loss = running_loss / num_batches
    epoch_metrics = metrics_tracker.get_dict()
    epoch_metrics['loss'] = avg_loss
    
    return epoch_metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate the model.
    
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    metrics_tracker = MetricsTracker()
    running_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(val_loader, desc='Validating'):
        rgb = batch['rgb'].to(device)
        depth = batch['depth'].to(device)
        sparse = batch['sparse'].to(device)
        semseg = batch['semseg'].to(device)
        
        input_tensor = torch.cat([rgb, sparse, semseg, semseg], dim=1)
        
        predictions = model(input_tensor)
        loss = criterion(predictions, depth)
        
        running_loss += loss.item()
        num_batches += 1
        
        pred_depth = predictions[0] if isinstance(predictions, (list, tuple)) else predictions
        metrics_tracker.update(pred_depth, depth)
    
    avg_loss = running_loss / num_batches
    val_metrics = metrics_tracker.get_dict()
    val_metrics['loss'] = avg_loss
    
    return val_metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[object],
    epoch: int,
    metrics: Dict[str, float],
    path: str
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, path)
    print(f'Saved checkpoint to {path}')


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    (output_dir / 'visualizations').mkdir(exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Set up data directories
    data_dir = Path(args.data_dir)
    rgb_dir = args.rgb_dir or str(data_dir / 'images')
    depth_dir = args.depth_dir or str(data_dir / 'depths')
    sparse_dir = args.sparse_dir or str(data_dir / 'labels')
    semseg_dir = args.semseg_dir or str(data_dir / 'instances')
    
    # Create dataloaders
    print('Loading datasets...')
    train_loader, val_loader = create_dataloaders(
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        sparse_dir=sparse_dir,
        semseg_dir=semseg_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=(256, 256)
    )
    
    # Create model
    print('Creating model...')
    model = DepthNet(
        batch_norm=args.batch_norm,
        dropout=args.dropout,
        depth_activation=args.depth_activation if args.depth_activation != 'none' else None
    ).to(device)
    
    print(f'Model parameters: {model.get_num_params():,}')
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Create loss function
    criterion = CombinedDepthLoss(
        l1_weight=args.l1_weight,
        gradient_weight=args.gradient_weight,
        multi_scale=True
    )
    
    # AMP scaler
    scaler = GradScaler() if args.amp and device.type == 'cuda' else None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f'Resuming from {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Training loop
    print(f'\nStarting training for {args.epochs} epochs...\n')
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, args, scaler
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        # Log metrics
        print(f'\nEpoch {epoch + 1}/{args.epochs}:')
        print(f'  Train Loss: {train_metrics["loss"]:.4f} | Val Loss: {val_metrics["loss"]:.4f}')
        print(f'  Val RMSE: {val_metrics["rmse"]:.4f} | Val δ<1.25: {val_metrics["delta_1"]:.4f}')
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                str(output_dir / 'checkpoints' / 'best_model.pth')
            )
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                str(output_dir / 'checkpoints' / f'checkpoint_epoch_{epoch + 1}.pth')
            )
        
        # Visualize predictions
        if (epoch + 1) % args.vis_every == 0:
            visualize_predictions(
                model, val_loader, device,
                num_samples=4,
                save_dir=str(output_dir / 'visualizations' / f'epoch_{epoch + 1}')
            )
    
    # Save final model
    torch.save(model.state_dict(), str(output_dir / 'depthnet_final.pth'))
    print(f'\nTraining complete! Final model saved to {output_dir / "depthnet_final.pth"}')


if __name__ == '__main__':
    main()
