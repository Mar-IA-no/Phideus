#!/usr/bin/env python3
"""
Massive HRM Training Script
Train HRM with 848 synthetic samples for fair comparison with VAE
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

class MassiveHRMDataset(Dataset):
    """Dataset for massive HRM training"""
    
    def __init__(self, data_path: str, mode: str = 'train', validation_split: float = 0.15):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Convert to list of samples
        self.samples = []
        for filename, sample_data in self.data.items():
            if 'ratio_hist_enriched' in sample_data:
                hist = np.array(sample_data['ratio_hist_enriched'])
                if hist.shape == (512, 3):
                    self.samples.append({
                        'filename': filename,
                        'histogram': hist
                    })
        
        # Split train/validation
        split_idx = int(len(self.samples) * (1 - validation_split))
        if mode == 'train':
            self.samples = self.samples[:split_idx]
        elif mode == 'validation':
            self.samples = self.samples[split_idx:]
        
        logger.info(f"Dataset {mode}: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        histogram = torch.FloatTensor(sample['histogram'])  # (512, 3)
        return {'histogram': histogram, 'filename': sample['filename']}


class EnhancedHRM(nn.Module):
    """Enhanced HRM for massive dataset training"""
    
    def __init__(self, input_dim=(512, 3), l_hidden_dim=384, h_hidden_dim=192, latent_dim=128, 
                 num_layers=3, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.l_hidden_dim = l_hidden_dim
        self.h_hidden_dim = h_hidden_dim
        self.latent_dim = latent_dim
        
        # Enhanced encoder: Deeper CNN with residual connections
        self.encoder = nn.Sequential(
            # First block
            nn.Conv1d(3, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Second block
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Third block
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Fourth block
            nn.Conv1d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            
            # Global pooling
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(384, l_hidden_dim)
        )
        
        # HRM-inspired hierarchical processing
        # L-Module: Fast multi-scale processing
        self.l_module = nn.GRU(
            l_hidden_dim, l_hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # H-Module: High-level reasoning with attention
        self.h_module = nn.LSTM(
            l_hidden_dim, h_hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            dropout=dropout
        )
        
        # Attention mechanism for H-Module
        self.h_attention = nn.MultiheadAttention(
            h_hidden_dim, num_heads=8, 
            dropout=dropout, batch_first=True
        )
        
        # Hierarchical fusion
        self.fusion = nn.Sequential(
            nn.Linear(l_hidden_dim + h_hidden_dim, l_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(l_hidden_dim, latent_dim)
        )
        
        # Enhanced decoder with skip connections
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, l_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(l_hidden_dim, 768),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(768, 512 * 3),
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
    
    def forward(self, x, cycles=4):
        batch_size = x.shape[0]
        
        # x: (batch, 512, 3) -> (batch, 3, 512) for Conv1d
        x = x.transpose(1, 2)
        
        # Encode
        encoded = self.encoder(x)  # (batch, l_hidden_dim)
        
        # Hierarchical processing with multiple cycles
        l_outputs = []
        l_state = None
        h_state = None
        
        for cycle in range(cycles):
            # L-Module: Fast processing
            l_input = encoded.unsqueeze(1)  # (batch, 1, l_hidden_dim)
            l_output, l_state = self.l_module(l_input, l_state)
            l_output = l_output.squeeze(1)  # (batch, l_hidden_dim)
            l_outputs.append(l_output)
            
            # H-Module: High-level reasoning (every cycle)
            if len(l_outputs) >= 2:
                # Stack recent L outputs for H-Module
                h_input = torch.stack(l_outputs[-2:], dim=1)  # (batch, 2, l_hidden_dim)
                h_output, h_state = self.h_module(h_input, h_state)
                
                # Apply attention to H-Module output
                h_attended, _ = self.h_attention(h_output, h_output, h_output)
                h_final = h_attended[:, -1, :]  # Take last timestep
            else:
                h_final = torch.zeros(batch_size, self.h_hidden_dim, device=x.device)
        
        # Fuse L and H outputs
        l_final = l_outputs[-1]  # Latest L output
        combined = torch.cat([l_final, h_final], dim=1)
        latent = self.fusion(combined)
        
        # Decode
        recon = self.decoder(latent)
        recon = recon.view(batch_size, 512, 3)
        
        return recon, latent


class MassiveHRMTrainer:
    """Trainer for massive HRM training"""
    
    def __init__(self, model, optimizer, device='cuda', mixed_precision=True):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.mixed_precision = mixed_precision
        self.scaler = GradScaler() if mixed_precision else None
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0.0
        
        for batch in tqdm(train_loader, desc="Training"):
            data = batch['histogram'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                with autocast(device_type=self.device):
                    recon, latent = self.model(data)
                    loss = self.criterion(recon, data)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                recon, latent = self.model(data)
                loss = self.criterion(recon, data)
                loss.backward()
                self.optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                data = batch['histogram'].to(self.device)
                
                if self.mixed_precision:
                    with autocast(device_type=self.device):
                        recon, latent = self.model(data)
                        loss = self.criterion(recon, data)
                else:
                    recon, latent = self.model(data)
                    loss = self.criterion(recon, data)
                
                val_loss += loss.item()
        
        avg_loss = val_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss


def main():
    """Main training function"""
    logger.info("🚀 Starting Massive HRM Training (848 samples)")
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"💻 Using device: {device}")
    
    # Load massive dataset
    data_path = "massive_synthetic_dataset.json"
    train_dataset = MassiveHRMDataset(data_path, mode='train', validation_split=0.15)
    val_dataset = MassiveHRMDataset(data_path, mode='validation', validation_split=0.15)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Create enhanced model
    model = EnhancedHRM(
        l_hidden_dim=384,
        h_hidden_dim=192, 
        latent_dim=128,
        num_layers=3,
        dropout=0.1
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"🏗️ Enhanced HRM created with {total_params:,} parameters")
    
    # Optimizer with scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=10
    )
    
    # Trainer
    trainer = MassiveHRMTrainer(model, optimizer, device=device, mixed_precision=True)
    
    # Training loop
    epochs = 50
    best_val_loss = float('inf')
    
    logger.info(f"🎯 Training for {epochs} epochs on {len(train_dataset)} samples")
    
    for epoch in range(epochs):
        logger.info(f"📊 Epoch {epoch + 1}/{epochs}")
        
        # Train and validate
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Logging
        logger.info(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("./massive_hrm_output/models", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': trainer.train_losses,
                'val_losses': trainer.val_losses,
                'best_val_loss': best_val_loss,
                'config': {
                    'l_hidden_dim': 384,
                    'h_hidden_dim': 192,
                    'latent_dim': 128,
                    'total_params': total_params,
                    'dataset_size': len(train_dataset) + len(val_dataset)
                }
            }, "./massive_hrm_output/models/best_massive_hrm.pth")
            
            logger.info(f"🏆 New best model saved! Val Loss: {best_val_loss:.6f}")
    
    # Save training curves
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(trainer.train_losses, label='Train Loss', color='blue')
    plt.plot(trainer.val_losses, label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Enhanced HRM Training - 848 Samples')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(trainer.val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Val Loss')
    plt.title('Validation Loss Progression')
    plt.legend()
    plt.grid(True)
    
    os.makedirs("./massive_hrm_output/plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig("./massive_hrm_output/plots/massive_training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Final results
    logger.info(f"✅ Training completed!")
    logger.info(f"📊 Final Results:")
    logger.info(f"   Best Val Loss: {best_val_loss:.6f}")
    logger.info(f"   Dataset Size: {len(train_dataset) + len(val_dataset)} samples")
    logger.info(f"   Model Parameters: {total_params:,}")
    logger.info(f"💾 Model saved: ./massive_hrm_output/models/best_massive_hrm.pth")


if __name__ == "__main__":
    main()