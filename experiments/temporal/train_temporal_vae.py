#!/usr/bin/env python3
"""
Training Pipeline para Attention-Based Temporal VAE
Entrenamiento optimizado para RTX 3090 con mixed precision y memory management
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torch.cuda.amp as amp

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import os
from tqdm import tqdm
import wandb

from attention_temporal_vae import RTX3090OptimizedTemporalVAE
from temporal_dataset import create_temporal_dataloaders

class TemporalVAELoss(nn.Module):
    """
    Loss function para Temporal VAE
    Combina VAE loss standard con regularización temporal
    """
    def __init__(self, 
                 beta=1.0, 
                 temporal_consistency_weight=0.1,
                 attention_sparsity_weight=0.01):
        super().__init__()
        
        self.beta = beta
        self.temporal_weight = temporal_consistency_weight
        self.sparsity_weight = attention_sparsity_weight
    
    def forward(self, reconstructed, original, mu, logvar, attention_weights, sequence_lengths=None):
        """
        Compute total loss
        
        Args:
            reconstructed: (batch, 512, 3) - histograma reconstruido
            original: (batch, seq_len, 512, 3) - secuencia original
            mu, logvar: (batch, latent_dim) - parámetros VAE
            attention_weights: (batch, num_heads, seq_len, seq_len)
            sequence_lengths: (batch,) - longitudes reales de secuencias
        """
        batch_size = reconstructed.shape[0]
        
        # 1. Reconstruction Loss
        # Comparar con promedio temporal de la secuencia original
        if sequence_lengths is not None:
            # Usar solo la parte válida de cada secuencia
            target = torch.zeros_like(reconstructed)
            for b in range(batch_size):
                valid_len = sequence_lengths[b]
                target[b] = original[b, :valid_len].mean(dim=0)
        else:
            target = original.mean(dim=1)  # Promedio temporal
        
        recon_loss = F.mse_loss(reconstructed, target, reduction='mean')
        
        # 2. KL Divergence (VAE standard)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        
        # 3. Temporal Consistency Loss
        temporal_loss = self._compute_temporal_consistency(
            attention_weights, sequence_lengths
        )
        
        # 4. Attention Sparsity Loss (opcional)
        sparsity_loss = self._compute_attention_sparsity(
            attention_weights, sequence_lengths
        )
        
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
    
    def _compute_temporal_consistency(self, attention_weights, sequence_lengths):
        """
        Regularización para attention patterns más coherentes temporalmente
        """
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # Promedio across heads
        avg_attention = attention_weights.mean(dim=1)  # (batch, seq_len, seq_len)
        
        consistency_loss = 0.0
        valid_samples = 0
        
        for b in range(batch_size):
            if sequence_lengths is not None:
                valid_len = sequence_lengths[b]
                att_matrix = avg_attention[b, :valid_len, :valid_len]
            else:
                att_matrix = avg_attention[b]
                valid_len = seq_len
            
            if valid_len > 2:
                # Penalizar attention patterns muy dispersos
                # Favorecer conexiones entre frames temporalmente cercanos
                temporal_distances = torch.abs(
                    torch.arange(valid_len, device=att_matrix.device).unsqueeze(0) -
                    torch.arange(valid_len, device=att_matrix.device).unsqueeze(1)
                ).float()
                
                # Weight inverso a la distancia temporal
                temporal_weights = 1.0 / (1.0 + temporal_distances)
                
                # Loss: attention debería correlacionar con proximidad temporal
                consistency = F.mse_loss(att_matrix, temporal_weights)
                consistency_loss += consistency
                valid_samples += 1
        
        return consistency_loss / max(valid_samples, 1)
    
    def _compute_attention_sparsity(self, attention_weights, sequence_lengths):
        """
        Regularización para attention patterns más sparse/interpretables
        """
        # L1 regularization en attention weights para promover sparsity
        return attention_weights.abs().mean()

class TemporalVAETrainer:
    """
    Trainer completo para Attention-Based Temporal VAE
    Optimizado para RTX 3090 con mixed precision
    """
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 learning_rate=1e-4,
                 weight_decay=1e-5,
                 device='cuda',
                 mixed_precision=True,
                 gradient_clipping=1.0,
                 save_dir='./checkpoints/temporal_vae',
                 log_wandb=False):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.mixed_precision = mixed_precision
        self.gradient_clipping = gradient_clipping
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function
        self.criterion = TemporalVAELoss()
        
        # Optimizer y scheduler
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10,
            min_lr=1e-6
        )
        
        # Mixed precision
        if mixed_precision:
            self.scaler = amp.GradScaler()
        
        # Logging
        self.log_wandb = log_wandb
        if log_wandb:
            wandb.init(project="phideus-temporal-vae")
            wandb.config.update({
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'mixed_precision': mixed_precision
            })
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self):
        """Entrenamiento de una época"""
        self.model.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'recon_loss': 0.0,
            'kl_loss': 0.0,
            'temporal_loss': 0.0,
            'sparsity_loss': 0.0
        }
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, (sequences, metadata) in enumerate(progress_bar):
            sequences = sequences.to(self.device)
            # Extraer lengths del metadata (ya viene como tensor del DataLoader)
            lengths = metadata['sequence_length'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                with amp.autocast():
                    # Forward pass
                    reconstructed, mu, logvar, attention_weights = self.model(sequences)
                    
                    # Compute loss
                    loss_dict = self.criterion(
                        reconstructed, sequences, mu, logvar, 
                        attention_weights, lengths
                    )
                
                # Backward pass con mixed precision
                self.scaler.scale(loss_dict['total_loss']).backward()
                
                # Gradient clipping
                if self.gradient_clipping > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clipping
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
            else:
                # Forward pass sin mixed precision
                reconstructed, mu, logvar, attention_weights = self.model(sequences)
                
                loss_dict = self.criterion(
                    reconstructed, sequences, mu, logvar,
                    attention_weights, lengths
                )
                
                # Backward pass
                loss_dict['total_loss'].backward()
                
                # Gradient clipping
                if self.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clipping
                    )
                
                self.optimizer.step()
            
            # Acumular losses
            for key, value in loss_dict.items():
                epoch_losses[key] += value.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.4f}",
                'recon': f"{loss_dict['recon_loss'].item():.4f}",
                'kl': f"{loss_dict['kl_loss'].item():.4f}"
            })
            
            # Log a wandb si está habilitado
            if self.log_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train_batch_loss': loss_dict['total_loss'].item(),
                    'train_batch_recon': loss_dict['recon_loss'].item(),
                    'train_batch_kl': loss_dict['kl_loss'].item(),
                    'epoch': self.epoch,
                    'batch': batch_idx
                })
        
        # Promedio de losses
        for key in epoch_losses:
            epoch_losses[key] /= len(self.train_loader)
        
        return epoch_losses
    
    def validate_epoch(self):
        """Validación de una época"""
        self.model.eval()
        
        epoch_losses = {
            'total_loss': 0.0,
            'recon_loss': 0.0,
            'kl_loss': 0.0,
            'temporal_loss': 0.0,
            'sparsity_loss': 0.0
        }
        
        with torch.no_grad():
            for sequences, metadata in tqdm(self.val_loader, desc="Validation"):
                sequences = sequences.to(self.device)
                # Extraer lengths del metadata (ya viene como tensor del DataLoader)
                lengths = metadata['sequence_length'].to(self.device)
                
                if self.mixed_precision:
                    with amp.autocast():
                        reconstructed, mu, logvar, attention_weights = self.model(sequences)
                        loss_dict = self.criterion(
                            reconstructed, sequences, mu, logvar,
                            attention_weights, lengths
                        )
                else:
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
            epoch_losses[key] /= len(self.val_loader)
        
        return epoch_losses
    
    def train(self, num_epochs):
        """Entrenamiento completo"""
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.mixed_precision}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Training
            train_losses = self.train_epoch()
            
            # Validation
            val_losses = self.validate_epoch()
            
            # Learning rate scheduling
            self.scheduler.step(val_losses['total_loss'])
            
            # Logging
            self.train_losses.append(train_losses)
            self.val_losses.append(val_losses)
            
            print(f"Epoch {epoch}:")
            print(f"  Train Loss: {train_losses['total_loss']:.4f} "
                  f"(Recon: {train_losses['recon_loss']:.4f}, "
                  f"KL: {train_losses['kl_loss']:.4f})")
            print(f"  Val Loss: {val_losses['total_loss']:.4f} "
                  f"(Recon: {val_losses['recon_loss']:.4f}, "
                  f"KL: {val_losses['kl_loss']:.4f})")
            
            if self.log_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_losses['total_loss'],
                    'val_loss': val_losses['total_loss'],
                    'train_recon': train_losses['recon_loss'],
                    'val_recon': val_losses['recon_loss'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save best model
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                self.save_checkpoint('best_model.pt')
                print(f"  New best model saved! Val loss: {self.best_val_loss:.4f}")
            
            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
        
        print("Training completed!")
        
    def save_checkpoint(self, filename):
        """Guardar checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        if self.mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, self.save_dir / filename)
    
    def load_checkpoint(self, filename):
        """Cargar checkpoint"""
        checkpoint = torch.load(self.save_dir / filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        if self.mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

def main():
    """Main training script"""
    
    # Configuración
    config = {
        'batch_size': 2,  # Reducido para RTX 3090
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'window_size': 1.0,
        'overlap': 0.5,
        'max_sequence_length': 60,  # Optimizado RTX 3090
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("Initializing Temporal VAE Training...")
    print(f"Config: {config}")
    
    # Buscar archivos WAV (usar datos existentes)
    possible_dirs = [
        Path("../../../test_wavs"),
        Path("../../../wavs_sinteticos_v3.0"), 
        Path("./test_data")
    ]
    
    wav_files = []
    for wav_dir in possible_dirs:
        if wav_dir.exists():
            files = list(wav_dir.glob("*.wav"))
            wav_files.extend(files)
            print(f"Found {len(files)} WAV files in {wav_dir}")
    
    if not wav_files:
        print("⚠️  No WAV files found for training")
        print("Please ensure you have audio files in one of these directories:")
        for d in possible_dirs:
            print(f"  - {d}")
        return
    
    print(f"Total WAV files: {len(wav_files)}")
    
    # Crear data loaders
    train_loader, val_loader, test_loader = create_temporal_dataloaders(
        wav_files,
        batch_size=config['batch_size'],
        window_size=config['window_size'],
        overlap=config['overlap'],
        max_sequence_length=config['max_sequence_length']
    )
    
    # Crear modelo optimizado para RTX 3090
    model = RTX3090OptimizedTemporalVAE()
    
    # Crear trainer
    trainer = TemporalVAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config['learning_rate'],
        device=config['device'],
        mixed_precision=True,
        log_wandb=False  # Cambiar a True si quieres usar wandb
    )
    
    # Entrenar
    trainer.train(config['num_epochs'])
    
    print("✅ Training completed successfully!")

if __name__ == "__main__":
    main()