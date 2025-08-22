#!/usr/bin/env python3
"""
Entrenamiento del Temporal VAE con Dataset Masivo (991 audios)
Optimizado para entrenamiento a gran escala con monitoring detallado
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.cuda.amp as amp

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import os
from tqdm import tqdm
import glob
from collections import defaultdict

from attention_temporal_vae import RTX3090OptimizedTemporalVAE
from temporal_dataset import create_temporal_dataloaders

class LargeScaleTemporalVAELoss(nn.Module):
    """Loss optimizado para dataset grande con regularización"""
    def __init__(self, 
                 beta=1.0,
                 temporal_consistency_weight=0.05,  # Reducido para dataset grande
                 attention_sparsity_weight=0.005):  # Reducido
        super().__init__()
        
        self.beta = beta
        self.temporal_weight = temporal_consistency_weight
        self.sparsity_weight = attention_sparsity_weight
    
    def forward(self, reconstructed, original, mu, logvar, attention_weights, sequence_lengths=None):
        batch_size = reconstructed.shape[0]
        
        # 1. Reconstruction Loss con Huber para robustez
        if sequence_lengths is not None:
            target = torch.zeros_like(reconstructed)
            for b in range(batch_size):
                valid_len = sequence_lengths[b]
                target[b] = original[b, :valid_len].mean(dim=0)
        else:
            target = original.mean(dim=1)
        
        # Huber loss más robusto que MSE para datasets grandes
        recon_loss = F.huber_loss(reconstructed, target, reduction='mean', delta=0.1)
        
        # 2. KL Divergence con β scheduling
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        
        # 3. Temporal consistency simplificada
        temporal_loss = self._compute_simplified_temporal_loss(attention_weights, sequence_lengths)
        
        # 4. Attention sparsity
        sparsity_loss = attention_weights.abs().mean()
        
        # Total loss
        total_loss = (recon_loss + 
                     self.beta * kl_loss + 
                     self.temporal_weight * temporal_loss +
                     self.sparsity_weight * sparsity_loss)
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'temporal_loss': temporal_loss,
            'sparsity_loss': sparsity_loss
        }
    
    def _compute_simplified_temporal_loss(self, attention_weights, sequence_lengths):
        """Temporal loss simplificado para eficiencia"""
        # Solo promedio de attention para evitar cálculos costosos
        return attention_weights.var(dim=-1).mean()

class LargeScaleTemporalVAETrainer:
    """Trainer optimizado para dataset masivo"""
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 learning_rate=5e-5,  # LR más bajo para dataset grande
                 weight_decay=1e-4,
                 device='cuda',
                 save_dir='./checkpoints/large_dataset_temporal_vae',
                 log_interval=50):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        
        # Loss function optimizado
        self.criterion = LargeScaleTemporalVAELoss()
        
        # Optimizer con configuración para dataset grande
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Scheduler cóseno para convergencia suave
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=50,  # 50 epochs
            eta_min=1e-6
        )
        
        # Mixed precision para eficiencia
        self.scaler = amp.GradScaler()
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = defaultdict(list)
        
    def train_epoch(self):
        """Training epoch optimizado"""
        self.model.train()
        
        epoch_losses = defaultdict(float)
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.epoch:3d}",
            leave=False
        )
        
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
            
            # Forward pass con mixed precision
            with amp.autocast():
                reconstructed, mu, logvar, attention_weights = self.model(sequences)
                loss_dict = self.criterion(
                    reconstructed, sequences, mu, logvar, 
                    attention_weights, lengths
                )
            
            # Backward pass
            self.scaler.scale(loss_dict['total_loss']).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Acumular losses
            for key, value in loss_dict.items():
                epoch_losses[key] += value.item()
            
            # Update progress bar
            if batch_idx % self.log_interval == 0:
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
        """Validación optimizada"""
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
                
                with amp.autocast():
                    reconstructed, mu, logvar, attention_weights = self.model(sequences)
                    loss_dict = self.criterion(
                        reconstructed, sequences, mu, logvar,
                        attention_weights, lengths
                    )
                
                # Acumular losses
                for key, value in loss_dict.items():
                    epoch_losses[key] += value.item()
        
        # Promedio de losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return dict(epoch_losses)
    
    def train(self, num_epochs):
        """Entrenamiento completo"""
        print(f"🚀 Starting Large-Scale Temporal VAE Training")
        print(f"📊 Dataset size: {len(self.train_loader.dataset)} sequences")
        print(f"📦 Batch size: {self.train_loader.batch_size}")
        print(f"⚙️  Device: {self.device}")
        print(f"🎯 Target epochs: {num_epochs}")
        print(f"💾 Save directory: {self.save_dir}")
        
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
                  f"({train_losses['recon_loss']:.4f}+{train_losses['kl_loss']:.4f}) | "
                  f"Val: {val_losses['total_loss']:.4f} "
                  f"({val_losses['recon_loss']:.4f}+{val_losses['kl_loss']:.4f}) | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {elapsed/60:.1f}m")
            
            # Save best model
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                self.save_checkpoint('best_model.pt')
                print(f"    🎯 New best model! Val loss: {self.best_val_loss:.4f}")
            
            # Save regular checkpoints
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
        
        total_time = time.time() - start_time
        print(f"\n✅ Training completed!")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"🏆 Best validation loss: {self.best_val_loss:.4f}")
        
        # Save training curves
        self.plot_training_curves()
    
    def save_checkpoint(self, filename):
        """Guardar checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': dict(self.training_history)
        }
        torch.save(checkpoint, self.save_dir / filename)
    
    def plot_training_curves(self):
        """Generar gráficos de entrenamiento"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(len(self.training_history['train_total_loss']))
        
        # Total Loss
        ax1.plot(epochs, self.training_history['train_total_loss'], 'b-', label='Train')
        ax1.plot(epochs, self.training_history['val_total_loss'], 'r-', label='Val')
        ax1.set_title('Total Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Reconstruction Loss
        ax2.plot(epochs, self.training_history['train_recon_loss'], 'b-', label='Train')
        ax2.plot(epochs, self.training_history['val_recon_loss'], 'r-', label='Val')
        ax2.set_title('Reconstruction Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        # KL Loss
        ax3.plot(epochs, self.training_history['train_kl_loss'], 'b-', label='Train')
        ax3.plot(epochs, self.training_history['val_kl_loss'], 'r-', label='Val')
        ax3.set_title('KL Divergence Loss')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True)
        
        # Temporal Loss
        ax4.plot(epochs, self.training_history['train_temporal_loss'], 'b-', label='Train')
        ax4.plot(epochs, self.training_history['val_temporal_loss'], 'r-', label='Val')
        ax4.set_title('Temporal Consistency Loss')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves_large_dataset.png', dpi=150, bbox_inches='tight')
        plt.close()

def main():
    """Main training script para dataset masivo"""
    
    # 🕒 ESTIMATED TIME: 4-6 hours para 50 epochs con 991 audios
    print("⏰ ESTIMATED TRAINING TIME: 4-6 hours for 50 epochs")
    print("📊 PROGRESS: Will provide updates every 50 batches")
    print("⚡ SPEED: ~15-20 minutes per epoch expected")
    print("🎯 STATUS: Preparing large-scale dataset training")
    
    # Configuración optimizada para dataset grande
    config = {
        'batch_size': 3,  # Aumentado ligeramente
        'learning_rate': 5e-5,  # Más conservador
        'num_epochs': 50,  # Más epochs para dataset grande
        'window_size': 1.0,
        'overlap': 0.5,
        'max_sequence_length': 50,  # Reducido para manejar más samples
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"🚀 Starting Large Dataset Training")
    print(f"Config: {config}")
    
    # Recopilar todos los WAVs disponibles
    wav_files = []
    
    # Sintéticos (848 archivos)
    synthetic_dir = Path("/root/Phideus/train/synthetic_dataset_500")
    if synthetic_dir.exists():
        synthetic_files = list(synthetic_dir.glob("*.wav"))
        wav_files.extend(synthetic_files)
        print(f"📁 Synthetic WAVs: {len(synthetic_files)}")
    
    # Reales del VAE (78 archivos)
    vae_dir = Path("/root/Phideus/train/VAE")
    if vae_dir.exists():
        vae_files = list(vae_dir.glob("*.wav"))
        wav_files.extend(vae_files)
        print(f"📁 Real VAE WAVs: {len(vae_files)}")
    
    # Test WAVs (60 archivos)
    for test_dir in [Path("/root/Phideus/test_wavs"), Path("/root/Phideus/test/test_wavs")]:
        if test_dir.exists():
            test_files = list(test_dir.glob("*.wav"))
            wav_files.extend(test_files)
            print(f"📁 Test WAVs ({test_dir.name}): {len(test_files)}")
    
    total_wavs = len(wav_files)
    print(f"📊 Total WAV files: {total_wavs}")
    
    if total_wavs < 500:
        print("⚠️  Warning: Dataset smaller than expected")
    
    # Crear data loaders
    print("⚙️  Creating temporal data loaders...")
    train_loader, val_loader, test_loader = create_temporal_dataloaders(
        wav_files,
        batch_size=config['batch_size'],
        window_size=config['window_size'],
        overlap=config['overlap'],
        max_sequence_length=config['max_sequence_length'],
        train_split=0.8,  # 80% para train
        val_split=0.15    # 15% para val, 5% para test
    )
    
    print(f"✅ Data loaders created:")
    print(f"   Train sequences: {len(train_loader.dataset)}")
    print(f"   Val sequences: {len(val_loader.dataset)}")
    print(f"   Test sequences: {len(test_loader.dataset)}")
    
    # Crear modelo optimizado
    model = RTX3090OptimizedTemporalVAE()
    
    # Crear trainer
    trainer = LargeScaleTemporalVAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config['learning_rate'],
        device=config['device']
    )
    
    # ENTRENAR
    trainer.train(config['num_epochs'])
    
    print("✅ Large-scale Temporal VAE training completed!")

if __name__ == "__main__":
    main()