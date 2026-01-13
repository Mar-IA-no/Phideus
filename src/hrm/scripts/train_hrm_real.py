#!/usr/bin/env python3
"""
Real Dataset HRM Training Script
Practical script for training HRM on real harmonic analysis datasets
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import logging
from datetime import datetime

# Import HRM components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from training.train_hrm_hierarchical import PhideusHRM, HRMDataset, HRMTrainer


def setup_logging(log_dir: str):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def validate_dataset(data_path: str, logger: logging.Logger) -> bool:
    """Validate that dataset exists and has correct format"""
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found: {data_path}")
        return False
    
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
            
        # Check if it's a list of dictionaries with required keys
        if not isinstance(data, list) or len(data) == 0:
            logger.error("Dataset must be a non-empty list")
            return False
        
        sample = data[0]
        required_keys = ['ratio_hist_lin', 'ratio_hist_log', 'ratio_hist_entropy']
        
        for key in required_keys:
            if key not in sample:
                logger.error(f"Missing required key: {key}")
                return False
        
        # Check histogram shape
        hist_shape = np.array(sample['ratio_hist_lin']).shape
        if hist_shape != (512,):
            logger.error(f"Invalid histogram shape: {hist_shape}, expected (512,)")
            return False
        
        logger.info(f"✅ Dataset validated: {len(data)} samples")
        return True
        
    except Exception as e:
        logger.error(f"Dataset validation error: {e}")
        return False


def create_model(config: dict, device: str, logger: logging.Logger) -> PhideusHRM:
    """Create and initialize HRM model"""
    try:
        model = PhideusHRM(
            input_dim=config.get('input_dim', (512, 3)),
            l_hidden_dim=config.get('l_hidden_dim', 256),
            h_hidden_dim=config.get('h_hidden_dim', 128),
            N=config.get('N', 4),
            T=config.get('T', 8),
            convergence_type=config.get('convergence_type', 'standard'),
            act_type=config.get('act_type', 'standard')
        ).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"🏗️ Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
        return model
        
    except Exception as e:
        logger.error(f"Model creation failed: {e}")
        raise


def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, 
                   checkpoint_path: str, logger: logging.Logger) -> int:
    """Load model checkpoint and return epoch number"""
    if not os.path.exists(checkpoint_path):
        logger.info("No checkpoint found, starting from scratch")
        return 0
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        
        logger.info(f"✅ Checkpoint loaded from epoch {epoch}")
        return epoch
        
    except Exception as e:
        logger.error(f"Checkpoint loading failed: {e}")
        return 0


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, 
                   epoch: int, loss: float, checkpoint_path: str, 
                   logger: logging.Logger):
    """Save model checkpoint"""
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, checkpoint_path)
        
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        
    except Exception as e:
        logger.error(f"Checkpoint saving failed: {e}")


def plot_training_curves(losses: dict, save_path: str, logger: logging.Logger):
    """Plot and save training curves"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('HRM Training Progress', fontsize=16)
        
        # Total loss
        axes[0, 0].plot(losses['total'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Reconstruction loss
        axes[0, 1].plot(losses['reconstruction'], 'orange')
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # Convergence loss
        axes[1, 0].plot(losses['convergence'], 'green')
        axes[1, 0].set_title('Convergence Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
        
        # ACT loss
        axes[1, 1].plot(losses['act'], 'red')
        axes[1, 1].set_title('ACT Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📈 Training curves saved: {save_path}")
        
    except Exception as e:
        logger.error(f"Plot saving failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Train HRM on real dataset')
    parser.add_argument('--data-path', required=True, help='Path to JSON dataset')
    parser.add_argument('--output-dir', default='./hrm_training_output', 
                       help='Output directory for models and logs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', default='auto', 
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--resume', action='store_true', 
                       help='Resume from checkpoint if available')
    parser.add_argument('--validation-split', type=float, default=0.2,
                       help='Validation split ratio')
    
    # Model configuration
    parser.add_argument('--l-hidden-dim', type=int, default=256, 
                       help='L-Module hidden dimension')
    parser.add_argument('--h-hidden-dim', type=int, default=128,
                       help='H-Module hidden dimension')
    parser.add_argument('--N', type=int, default=4, 
                       help='Number of high-level cycles')
    parser.add_argument('--T', type=int, default=8, 
                       help='Number of low-level steps per cycle')
    parser.add_argument('--convergence-type', default='standard',
                       choices=['standard', 'adaptive', 'residual'],
                       help='Type of hierarchical convergence')
    parser.add_argument('--act-type', default='standard',
                       choices=['standard', 'enhanced'],
                       help='Type of ACT mechanism')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(str(output_dir / 'logs'))
    logger.info("🚀 Starting HRM training on real dataset")
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"💻 Using device: {device}")
    
    # Validate dataset
    if not validate_dataset(args.data_path, logger):
        logger.error("Dataset validation failed. Exiting.")
        return 1
    
    # Create datasets
    logger.info("📚 Loading datasets...")
    train_dataset = HRMDataset(args.data_path, mode='train', 
                              validation_split=args.validation_split)
    val_dataset = HRMDataset(args.data_path, mode='validation', 
                            validation_split=args.validation_split)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    
    logger.info(f"📊 Training samples: {len(train_dataset)}")
    logger.info(f"📊 Validation samples: {len(val_dataset)}")
    
    # Create model
    model_config = {
        'input_dim': (512, 3),
        'l_hidden_dim': args.l_hidden_dim,
        'h_hidden_dim': args.h_hidden_dim,
        'N': args.N,
        'T': args.T,
        'convergence_type': args.convergence_type,
        'act_type': args.act_type
    }
    
    model = create_model(model_config, device, logger)
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=10, verbose=True
    )
    
    # Create trainer
    trainer_config = {
        'device': device,
        'mixed_precision': device == 'cuda',
        'convergence_weight': 0.1,
        'act_weight': 0.05,
        'deep_supervision_layers': [2, 4],
        'gradient_clip_norm': 1.0
    }
    
    trainer = HRMTrainer(model, optimizer, **trainer_config)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        checkpoint_path = output_dir / 'checkpoints' / 'latest.pth'
        start_epoch = load_checkpoint(model, optimizer, str(checkpoint_path), logger)
    
    # Training loop
    logger.info(f"🎯 Starting training from epoch {start_epoch} to {args.epochs}")
    
    # Track losses for plotting
    loss_history = {
        'total': [],
        'reconstruction': [],
        'convergence': [],
        'act': []
    }
    
    best_val_loss = float('inf')
    
    try:
        for epoch in range(start_epoch, args.epochs):
            logger.info(f"🔄 Epoch {epoch + 1}/{args.epochs}")
            
            # Training
            train_losses = trainer.train_epoch(train_loader)
            
            # Validation
            val_losses = trainer.validate(val_loader)
            
            # Scheduler step
            scheduler.step(val_losses['total'])
            
            # Log progress
            logger.info(f"📊 Train Loss: {train_losses['total']:.6f} | "
                       f"Val Loss: {val_losses['total']:.6f}")
            
            # Track losses
            for key in loss_history:
                loss_history[key].append(val_losses[key])
            
            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                best_model_path = output_dir / 'models' / 'best_hrm_model.pth'
                best_model_path.parent.mkdir(parents=True, exist_ok=True)
                save_checkpoint(model, optimizer, epoch + 1, val_losses['total'],
                              str(best_model_path), logger)
                logger.info(f"🏆 New best model saved with loss: {best_val_loss:.6f}")
            
            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_dir = output_dir / 'checkpoints'
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                save_checkpoint(model, optimizer, epoch + 1, val_losses['total'],
                              str(checkpoint_dir / f'epoch_{epoch + 1}.pth'), logger)
            
            # Always save latest
            latest_path = output_dir / 'checkpoints' / 'latest.pth'
            latest_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, optimizer, epoch + 1, val_losses['total'],
                          str(latest_path), logger)
        
        # Final model save
        final_model_path = output_dir / 'models' / 'final_hrm_model.pth'
        final_model_path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(model, optimizer, args.epochs, val_losses['total'],
                      str(final_model_path), logger)
        
        # Plot training curves
        plot_path = output_dir / 'plots' / 'training_curves.png'
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_training_curves(loss_history, str(plot_path), logger)
        
        logger.info(f"✅ Training completed successfully!")
        logger.info(f"📁 Models saved in: {output_dir / 'models'}")
        logger.info(f"📈 Training curves: {plot_path}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted by user")
        # Save current state
        interrupt_path = output_dir / 'checkpoints' / 'interrupted.pth'
        interrupt_path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(model, optimizer, epoch, val_losses['total'],
                      str(interrupt_path), logger)
        return 1
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    exit(main())