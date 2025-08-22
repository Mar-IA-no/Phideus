#!/usr/bin/env python3
"""
VAE Base Training Script - Sin Linear Attention
Train baseline VAE WITHOUT Linear Attention for complete comparison:
1. VAE Base (this script)
2. VAE + Linear Attention 
3. Enhanced HRM
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import dataset from HRM training
from train_hrm_massive import MassiveHRMDataset


class BaseVAE(nn.Module):
    """Baseline VAE WITHOUT Linear Attention - Pure CNN Architecture"""
    
    def __init__(self, input_dim=(512, 3), latent_dim=128, base_channels=64, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Standard CNN encoder - NO Linear Attention
        self.encoder = nn.Sequential(
            # First block
            nn.Conv1d(3, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Second block
            nn.Conv1d(base_channels, base_channels*2, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Third block
            nn.Conv1d(base_channels*2, base_channels*4, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels*4),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Fourth block - simpler than Enhanced VAE
            nn.Conv1d(base_channels*4, base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels*4),
            nn.ReLU(),
            
            # Global pooling
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # Latent space projections
        self.fc_mu = nn.Linear(base_channels*4, latent_dim)
        self.fc_logvar = nn.Linear(base_channels*4, latent_dim)
        
        # Simple decoder - NO complex enhancements
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, base_channels*4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(base_channels*4, base_channels*6),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(base_channels*6, 512 * 3),
            nn.ReLU()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def encode(self, x):
        # x: (batch, 512, 3) -> (batch, 3, 512)
        x = x.transpose(1, 2)
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        decoded = self.decoder(z)
        return decoded.view(-1, 512, 3)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class BaseVAELoss(nn.Module):
    """Standard VAE loss with KL divergence"""
    
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        self.mse = nn.MSELoss(reduction='sum')
    
    def forward(self, recon, original, mu, logvar):
        # Reconstruction loss
        recon_loss = self.mse(recon, original) / original.size(0)
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / original.size(0)
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss


class BaseVAETrainer:
    """Trainer for baseline VAE training"""
    
    def __init__(self, model, optimizer, device='cuda', mixed_precision=True, beta=1.0):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.mixed_precision = mixed_precision
        self.scaler = GradScaler() if mixed_precision else None
        
        # Loss function
        self.criterion = BaseVAELoss(beta=beta)
        
        # Metrics tracking
        self.train_losses = {'total': [], 'recon': [], 'kl': []}
        self.val_losses = {'total': [], 'recon': [], 'kl': []}
    
    def train_epoch(self, train_loader):
        self.model.train()
        epoch_losses = {'total': 0.0, 'recon': 0.0, 'kl': 0.0}
        
        for batch in tqdm(train_loader, desc="Training"):
            data = batch['histogram'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                with autocast(device_type=self.device):
                    recon, mu, logvar = self.model(data)
                    total_loss, recon_loss, kl_loss = self.criterion(recon, data, mu, logvar)
                
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                recon, mu, logvar = self.model(data)
                total_loss, recon_loss, kl_loss = self.criterion(recon, data, mu, logvar)
                total_loss.backward()
                self.optimizer.step()
            
            epoch_losses['total'] += total_loss.item()
            epoch_losses['recon'] += recon_loss.item()
            epoch_losses['kl'] += kl_loss.item()
        
        # Average losses
        for key in epoch_losses:
            avg_loss = epoch_losses[key] / len(train_loader)
            epoch_losses[key] = avg_loss
            self.train_losses[key].append(avg_loss)
        
        return epoch_losses
    
    def validate(self, val_loader):
        self.model.eval()
        val_losses = {'total': 0.0, 'recon': 0.0, 'kl': 0.0}
        
        with torch.no_grad():
            for batch in val_loader:
                data = batch['histogram'].to(self.device)
                
                if self.mixed_precision:
                    with autocast(device_type=self.device):
                        recon, mu, logvar = self.model(data)
                        total_loss, recon_loss, kl_loss = self.criterion(recon, data, mu, logvar)
                else:
                    recon, mu, logvar = self.model(data)
                    total_loss, recon_loss, kl_loss = self.criterion(recon, data, mu, logvar)
                
                val_losses['total'] += total_loss.item()
                val_losses['recon'] += recon_loss.item()
                val_losses['kl'] += kl_loss.item()
        
        # Average losses
        for key in val_losses:
            avg_loss = val_losses[key] / len(val_loader)
            val_losses[key] = avg_loss
            self.val_losses[key].append(avg_loss)
        
        return val_losses


def main():
    """Main training function"""
    logger.info("🚀 Starting Baseline VAE Training (NO Linear Attention)")
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"💻 Using device: {device}")
    
    # Load massive dataset (same as HRM and Enhanced VAE)
    data_path = "data/datasets/massive_synthetic_dataset.json"
    train_dataset = MassiveHRMDataset(data_path, mode='train', validation_split=0.15)
    val_dataset = MassiveHRMDataset(data_path, mode='validation', validation_split=0.15)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Create baseline VAE model - simpler than Enhanced VAE
    model = BaseVAE(
        latent_dim=128,
        base_channels=64,
        dropout=0.1
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"🏗️ Baseline VAE created with {total_params:,} parameters")
    
    # Optimizer with scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=10
    )
    
    # Trainer
    trainer = BaseVAETrainer(model, optimizer, device=device, mixed_precision=True, beta=1.0)
    
    # Training loop
    epochs = 50
    best_val_loss = float('inf')
    
    logger.info(f"🎯 Training for {epochs} epochs on {len(train_dataset)} samples")
    
    for epoch in range(epochs):
        logger.info(f"📊 Epoch {epoch + 1}/{epochs}")
        
        # Train and validate
        train_losses = trainer.train_epoch(train_loader)
        val_losses = trainer.validate(val_loader)
        
        # Scheduler step
        scheduler.step(val_losses['total'])
        
        # Logging
        logger.info(f"Train - Total: {train_losses['total']:.6f}, Recon: {train_losses['recon']:.6f}, KL: {train_losses['kl']:.6f}")
        logger.info(f"Val   - Total: {val_losses['total']:.6f}, Recon: {val_losses['recon']:.6f}, KL: {val_losses['kl']:.6f}")
        
        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            os.makedirs("data/training_outputs/vae/baseline_vae_output/models", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': trainer.train_losses,
                'val_losses': trainer.val_losses,
                'best_val_loss': best_val_loss,
                'config': {
                    'latent_dim': 128,
                    'base_channels': 64,
                    'total_params': total_params,
                    'dataset_size': len(train_dataset) + len(val_dataset),
                    'architecture': 'Baseline VAE (NO Linear Attention)'
                }
            }, "data/training_outputs/vae/baseline_vae_output/models/best_baseline_vae.pth")
            
            logger.info(f"🏆 New best Baseline VAE model saved! Val Loss: {best_val_loss:.6f}")
    
    # Save training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Baseline VAE Training (NO Linear Attention) - 848 Samples', fontsize=16)
    
    # Total loss
    axes[0, 0].plot(trainer.train_losses['total'], label='Train Total', color='blue')
    axes[0, 0].plot(trainer.val_losses['total'], label='Val Total', color='red')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Reconstruction loss
    axes[0, 1].plot(trainer.train_losses['recon'], label='Train Recon', color='green')
    axes[0, 1].plot(trainer.val_losses['recon'], label='Val Recon', color='orange')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # KL loss
    axes[1, 0].plot(trainer.train_losses['kl'], label='Train KL', color='purple')
    axes[1, 0].plot(trainer.val_losses['kl'], label='Val KL', color='brown')
    axes[1, 0].set_title('KL Divergence')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Validation progression
    axes[1, 1].plot(trainer.val_losses['total'], label='Val Total', color='red', linewidth=2)
    axes[1, 1].set_title('Validation Loss Progression')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    os.makedirs("data/training_outputs/vae/baseline_vae_output/plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig("data/training_outputs/vae/baseline_vae_output/plots/baseline_vae_training_curves.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Final results
    logger.info(f"✅ Baseline VAE Training completed!")
    logger.info(f"📊 Final Results:")
    logger.info(f"   Best Val Loss: {best_val_loss:.6f}")
    logger.info(f"   Dataset Size: {len(train_dataset) + len(val_dataset)} samples")
    logger.info(f"   Model Parameters: {total_params:,}")
    logger.info(f"   Architecture: Baseline VAE (NO Linear Attention)")
    logger.info(f"💾 Model saved: data/training_outputs/vae/baseline_vae_output/models/best_baseline_vae.pth")


if __name__ == "__main__":
    main()