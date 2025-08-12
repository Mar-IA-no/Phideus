#!/usr/bin/env python3
"""
Train VAE Phideus - Pipeline de entrenamiento para VAE + CNN 1D

Entrenamiento optimizado para RTX 3090:
- FP16 mixed precision
- Adam8bit optimizer
- Gradient accumulation
- Checkpointing automático
- Validación y visualización

Uso:
    python train_vae_phideus.py --data train_data.json --epochs 100
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import matplotlib.pyplot as plt

# Imports locales
from vae_phideus_v1 import PhideusVAE, PhideusDataset, vae_loss, create_model_and_dataset

# Suprimir warnings de deprecated
warnings.filterwarnings("ignore", category=UserWarning)


class VAETrainer:
    """Trainer para VAE Phideus con todas las optimizaciones."""
    
    def __init__(self, model: PhideusVAE, device: str, 
                 learning_rate: float = 1e-3, beta_schedule: str = 'constant'):
        self.model = model
        self.device = device
        self.beta_schedule = beta_schedule
        
        # Optimizer con 8-bit Adam si disponible
        try:
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.Adam8bit(model.parameters(), lr=learning_rate)
            print("✅ Using bitsandbytes Adam8bit optimizer")
        except ImportError:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            print("⚠️ Using standard Adam optimizer (install bitsandbytes for 8bit)")
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.beta_values = []
        
    def get_beta(self, epoch: int, total_epochs: int) -> float:
        """β scheduling para β-VAE."""
        if self.beta_schedule == 'constant':
            return 1.0
        elif self.beta_schedule == 'linear':
            return min(1.0, epoch / (total_epochs * 0.5))  # Ramp up first half
        elif self.beta_schedule == 'cyclical':
            cycle = 10
            return 0.5 * (1 + np.cos(2 * np.pi * (epoch % cycle) / cycle))
        else:
            return 1.0
    
    def train_epoch(self, dataloader: DataLoader, epoch: int, total_epochs: int,
                   accumulation_steps: int = 1) -> Dict[str, float]:
        """Entrenamiento de una época."""
        self.model.train()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0
        
        beta = self.get_beta(epoch, total_epochs)
        self.beta_values.append(beta)
        
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(self.device)
            
            # Forward pass with mixed precision
            with autocast():
                output = self.model(batch)
                loss_dict = vae_loss(output['reconstruction'], batch,
                                   output['mu'], output['logvar'], beta=beta)
                
                # Gradient accumulation
                loss = loss_dict['total_loss'] / accumulation_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            # Tracking
            total_loss += loss_dict['total_loss'].item()
            total_recon_loss += loss_dict['recon_loss'].item()
            total_kl_loss += loss_dict['kl_loss'].item()
            num_batches += 1
            
            # Progress logging
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx:3d}/{len(dataloader)} | "
                      f"Loss: {loss_dict['total_loss'].item():.4f} | "
                      f"Recon: {loss_dict['recon_loss'].item():.4f} | "
                      f"KL: {loss_dict['kl_loss'].item():.4f} | "
                      f"β: {beta:.3f}")
        
        # Learning rate scheduling
        self.scheduler.step()
        
        avg_losses = {
            'total_loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'kl_loss': total_kl_loss / num_batches,
            'beta': beta,
            'lr': self.scheduler.get_last_lr()[0]
        }
        
        self.train_losses.append(avg_losses)
        return avg_losses
    
    def validate(self, dataloader: DataLoader, epoch: int, total_epochs: int) -> Dict[str, float]:
        """Validación del modelo."""
        self.model.eval()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0
        
        beta = self.get_beta(epoch, total_epochs)
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                
                with autocast():
                    output = self.model(batch)
                    loss_dict = vae_loss(output['reconstruction'], batch,
                                       output['mu'], output['logvar'], beta=beta)
                
                total_loss += loss_dict['total_loss'].item()
                total_recon_loss += loss_dict['recon_loss'].item()
                total_kl_loss += loss_dict['kl_loss'].item()
                num_batches += 1
        
        avg_losses = {
            'total_loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'kl_loss': total_kl_loss / num_batches,
            'beta': beta
        }
        
        self.val_losses.append(avg_losses)
        return avg_losses
    
    def save_checkpoint(self, epoch: int, save_path: Path, is_best: bool = False):
        """Guardar checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'beta_values': self.beta_values
        }
        
        # Guardar checkpoint regular
        torch.save(checkpoint, save_path / f'checkpoint_epoch_{epoch}.pth')
        
        # Guardar mejor modelo
        if is_best:
            torch.save(checkpoint, save_path / 'best_model.pth')
        
        # Mantener solo últimos 5 checkpoints
        checkpoints = list(save_path.glob('checkpoint_epoch_*.pth'))
        if len(checkpoints) > 5:
            oldest = min(checkpoints, key=lambda x: x.stat().st_mtime)
            oldest.unlink()
    
    def plot_training_curves(self, save_path: Path):
        """Generar plots de entrenamiento."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss total
        axes[0, 0].plot(epochs, [x['total_loss'] for x in self.train_losses], 'b-', label='Train')
        if self.val_losses:
            axes[0, 0].plot(epochs, [x['total_loss'] for x in self.val_losses], 'r-', label='Val')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Reconstruction loss
        axes[0, 1].plot(epochs, [x['recon_loss'] for x in self.train_losses], 'b-', label='Train')
        if self.val_losses:
            axes[0, 1].plot(epochs, [x['recon_loss'] for x in self.val_losses], 'r-', label='Val')
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # KL loss
        axes[1, 0].plot(epochs, [x['kl_loss'] for x in self.train_losses], 'b-', label='Train')
        if self.val_losses:
            axes[1, 0].plot(epochs, [x['kl_loss'] for x in self.val_losses], 'r-', label='Val')
        axes[1, 0].set_title('KL Divergence')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Beta schedule + LR
        ax1 = axes[1, 1]
        ax1.plot(epochs, self.beta_values, 'g-', label='Beta')
        ax1.set_ylabel('Beta', color='g')
        ax1.tick_params(axis='y', labelcolor='g')
        
        ax2 = ax1.twinx()
        ax2.plot(epochs, [x['lr'] for x in self.train_losses], 'orange', label='LR')
        ax2.set_ylabel('Learning Rate', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax2.set_yscale('log')
        
        axes[1, 1].set_title('Beta Schedule & Learning Rate')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train VAE Phideus')
    parser.add_argument('--data', type=Path, default=Path('test/test-json/test_enriched_512.json'),
                       help='Path to JSON dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--beta-schedule', type=str, default='linear',
                       choices=['constant', 'linear', 'cyclical'])
    parser.add_argument('--use-attention', action='store_true', help='Use linear attention')
    parser.add_argument('--accumulation-steps', type=int, default=1, help='Gradient accumulation')
    parser.add_argument('--save-dir', type=Path, default=Path('vae_checkpoints'),
                       help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Training VAE Phideus on {device}")
    
    # Crear directorio de guardado
    args.save_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar configuración
    config = vars(args)
    with open(args.save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    # Crear modelo y dataset
    model, dataset = create_model_and_dataset(args.data, device, args.use_attention)
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val")
    print(f"🔧 Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Trainer
    trainer = VAETrainer(model, device, args.lr, args.beta_schedule)
    
    # Training loop
    best_val_loss = float('inf')
    
    print("\n🔥 Starting training...")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n📈 Epoch {epoch}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch, args.epochs, 
                                          args.accumulation_steps)
        
        # Validate
        val_metrics = trainer.validate(val_loader, epoch, args.epochs)
        
        # Logging
        print(f"🟢 Train | Loss: {train_metrics['total_loss']:.4f} | "
              f"Recon: {train_metrics['recon_loss']:.4f} | "
              f"KL: {train_metrics['kl_loss']:.4f} | LR: {train_metrics['lr']:.2e}")
        
        print(f"🔵 Val   | Loss: {val_metrics['total_loss']:.4f} | "
              f"Recon: {val_metrics['recon_loss']:.4f} | "
              f"KL: {val_metrics['kl_loss']:.4f}")
        
        # Save checkpoint
        is_best = val_metrics['total_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['total_loss']
            print("🌟 New best model!")
        
        if epoch % 5 == 0 or is_best:
            trainer.save_checkpoint(epoch, args.save_dir, is_best)
        
        # Plot curves every 10 epochs
        if epoch % 10 == 0:
            trainer.plot_training_curves(args.save_dir)
    
    # Final results
    total_time = time.time() - start_time
    print(f"\n✅ Training completed in {total_time/60:.1f} minutes")
    print(f"🏆 Best validation loss: {best_val_loss:.4f}")
    
    # Final plots and save
    trainer.plot_training_curves(args.save_dir)
    trainer.save_checkpoint(args.epochs, args.save_dir, False)
    
    print(f"💾 All results saved to {args.save_dir}")


if __name__ == "__main__":
    main()