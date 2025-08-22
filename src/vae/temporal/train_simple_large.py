#!/usr/bin/env python3
"""
Entrenamiento Simple del Temporal VAE con Dataset Masivo
Versión simplificada sin mixed precision para evitar problemas de compatibilidad
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import os
from tqdm import tqdm
from collections import defaultdict

from attention_temporal_vae import RTX3090OptimizedTemporalVAE
from temporal_dataset import create_temporal_dataloaders

class SimpleTemporalVAELoss(nn.Module):
    """Loss simplificado para dataset grande"""
    def __init__(self, beta=1.0, temporal_weight=0.05):
        super().__init__()
        self.beta = beta
        self.temporal_weight = temporal_weight
    
    def forward(self, reconstructed, original, mu, logvar, attention_weights, sequence_lengths=None):
        batch_size = reconstructed.shape[0]
        
        # 1. Reconstruction Loss
        if sequence_lengths is not None:
            target = torch.zeros_like(reconstructed)
            for b in range(batch_size):
                valid_len = sequence_lengths[b]
                target[b] = original[b, :valid_len].mean(dim=0)
        else:
            target = original.mean(dim=1)
        
        recon_loss = F.mse_loss(reconstructed, target, reduction='mean')
        
        # 2. KL Divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        
        # 3. Simple temporal regularization
        temporal_loss = attention_weights.var(dim=-1).mean()
        
        total_loss = recon_loss + self.beta * kl_loss + self.temporal_weight * temporal_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'temporal_loss': temporal_loss
        }

class SimpleTemporalVAETrainer:
    """Trainer simplificado para dataset masivo"""
    def __init__(self, model, train_loader, val_loader, learning_rate=5e-5, device='cuda'):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.save_dir = Path('./checkpoints/simple_large_temporal_vae')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function
        self.criterion = SimpleTemporalVAELoss()
        
        # Optimizer
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-6)
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = defaultdict(list)
        
    def train_epoch(self):
        """Training epoch"""
        self.model.train()
        
        epoch_losses = defaultdict(float)
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch:3d}", leave=False)
        
        for batch_idx, batch_data in enumerate(progress_bar):
            if len(batch_data) == 2:
                sequences, metadata = batch_data
                sequences = sequences.to(self.device)
                lengths = metadata['sequence_length'].to(self.device)
            else:
                sequences, lengths, metadata = batch_data
                sequences = sequences.to(self.device)
                lengths = lengths.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            reconstructed, mu, logvar, attention_weights = self.model(sequences)
            loss_dict = self.criterion(reconstructed, sequences, mu, logvar, attention_weights, lengths)
            
            # Backward pass
            loss_dict['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # Acumular losses
            for key, value in loss_dict.items():
                epoch_losses[key] += value.item()
            
            # Update progress bar
            if batch_idx % 50 == 0:
                progress_bar.set_postfix({
                    'loss': f"{loss_dict['total_loss'].item():.4f}",
                    'recon': f"{loss_dict['recon_loss'].item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
        
        # Promedio de losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return dict(epoch_losses)
    
    def validate_epoch(self):
        """Validación"""
        self.model.eval()
        
        epoch_losses = defaultdict(float)
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc="Validating", leave=False):
                if len(batch_data) == 2:
                    sequences, metadata = batch_data
                    sequences = sequences.to(self.device)
                    lengths = metadata['sequence_length'].to(self.device)
                else:
                    sequences, lengths, metadata = batch_data
                    sequences = sequences.to(self.device)
                    lengths = lengths.to(self.device)
                
                reconstructed, mu, logvar, attention_weights = self.model(sequences)
                loss_dict = self.criterion(reconstructed, sequences, mu, logvar, attention_weights, lengths)
                
                # Acumular losses
                for key, value in loss_dict.items():
                    epoch_losses[key] += value.item()
        
        # Promedio de losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return dict(epoch_losses)
    
    def train(self, num_epochs):
        """Entrenamiento completo"""
        print(f"🚀 Starting Simple Large-Scale Training")
        print(f"📊 Dataset size: {len(self.train_loader.dataset)} sequences")
        print(f"🎯 Target epochs: {num_epochs}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Training
            train_losses = self.train_epoch()
            
            # Validation  
            val_losses = self.validate_epoch()
            
            # Scheduler step
            self.scheduler.step()
            
            # Logging
            for key in train_losses:
                self.training_history[f'train_{key}'].append(train_losses[key])
                self.training_history[f'val_{key}'].append(val_losses[key])
            
            # Console output
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d} | "
                  f"Train: {train_losses['total_loss']:.4f} "
                  f"(R:{train_losses['recon_loss']:.4f} K:{train_losses['kl_loss']:.4f}) | "
                  f"Val: {val_losses['total_loss']:.4f} "
                  f"(R:{val_losses['recon_loss']:.4f} K:{val_losses['kl_loss']:.4f}) | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {elapsed/60:.1f}m")
            
            # Save best model
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                self.save_checkpoint('best_model.pt')
                print(f"    ✨ New best! Val loss: {self.best_val_loss:.4f}")
            
            # Save checkpoints
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
        
        total_time = time.time() - start_time
        print(f"\n✅ Training completed! Total time: {total_time/60:.1f} minutes")
        print(f"🏆 Best validation loss: {self.best_val_loss:.4f}")
        
        # Plot results
        self.plot_training_curves()
    
    def save_checkpoint(self, filename):
        """Guardar checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': dict(self.training_history)
        }
        torch.save(checkpoint, self.save_dir / filename)
    
    def plot_training_curves(self):
        """Generar gráficos de entrenamiento"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        epochs = range(len(self.training_history['train_total_loss']))
        
        # Total Loss
        ax1.plot(epochs, self.training_history['train_total_loss'], 'b-', label='Train')
        ax1.plot(epochs, self.training_history['val_total_loss'], 'r-', label='Val')
        ax1.set_title('Total Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Reconstruction Loss
        ax2.plot(epochs, self.training_history['train_recon_loss'], 'b-', label='Train')
        ax2.plot(epochs, self.training_history['val_recon_loss'], 'r-', label='Val')
        ax2.set_title('Reconstruction Loss')
        ax2.legend()
        ax2.grid(True)
        
        # KL Loss
        ax3.plot(epochs, self.training_history['train_kl_loss'], 'b-', label='Train')
        ax3.plot(epochs, self.training_history['val_kl_loss'], 'r-', label='Val')
        ax3.set_title('KL Divergence Loss')
        ax3.legend()
        ax3.grid(True)
        
        # Learning Rate
        ax4.plot(epochs, [entry for entry in self.training_history.get('learning_rate', [self.optimizer.param_groups[0]['lr']] * len(epochs))])
        ax4.set_title('Learning Rate')
        ax4.set_yscale('log')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()

def main():
    """Main training script"""
    
    print("⏰ ESTIMATED TRAINING TIME: 3-5 hours for 30 epochs")
    print("📊 PROGRESS: Simplified training without mixed precision")
    print("⚡ SPEED: ~8-12 minutes per epoch expected")
    print("🎯 STATUS: Preparing simple large-scale training")
    
    # Configuración 
    config = {
        'batch_size': 4,  # Aumentado un poco más
        'learning_rate': 1e-4,  # LR un poco más alto
        'num_epochs': 30,  # Reducido para test inicial
        'window_size': 1.0,
        'overlap': 0.5,
        'max_sequence_length': 40,  # Reducido para eficiencia
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"🚀 Starting Simple Large Dataset Training")
    print(f"Config: {config}")
    
    # Recopilar WAVs
    wav_files = []
    
    # Sintéticos
    synthetic_dir = Path("/root/Phideus/train/synthetic_dataset_500")
    if synthetic_dir.exists():
        synthetic_files = list(synthetic_dir.glob("*.wav"))
        wav_files.extend(synthetic_files)
        print(f"📁 Synthetic WAVs: {len(synthetic_files)}")
    
    # Reales
    vae_dir = Path("/root/Phideus/train/VAE")
    if vae_dir.exists():
        vae_files = list(vae_dir.glob("*.wav"))
        wav_files.extend(vae_files)
        print(f"📁 Real VAVs: {len(vae_files)}")
    
    print(f"📊 Total WAV files: {len(wav_files)}")
    
    # Data loaders
    train_loader, val_loader, test_loader = create_temporal_dataloaders(
        wav_files,
        batch_size=config['batch_size'],
        window_size=config['window_size'],
        overlap=config['overlap'],
        max_sequence_length=config['max_sequence_length'],
        train_split=0.8,
        val_split=0.2
    )
    
    print(f"✅ Data loaders created:")
    print(f"   Train: {len(train_loader.dataset)} sequences")  
    print(f"   Val: {len(val_loader.dataset)} sequences")
    
    # Modelo
    model = RTX3090OptimizedTemporalVAE()
    
    # Trainer
    trainer = SimpleTemporalVAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config['learning_rate'],
        device=config['device']
    )
    
    # ENTRENAR
    trainer.train(config['num_epochs'])
    
    print("✅ Simple Large-scale Temporal VAE training completed!")

if __name__ == "__main__":
    main()