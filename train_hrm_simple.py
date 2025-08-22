#!/usr/bin/env python3
"""
Simple HRM Training Script
Direct execution without relative imports
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

class HRMDataset(Dataset):
    """Simple dataset for HRM training"""
    
    def __init__(self, data_path: str, mode: str = 'train', validation_split: float = 0.2):
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


class SimpleHRM(nn.Module):
    """Simplified HRM for testing"""
    
    def __init__(self, input_dim=(512, 3), hidden_dim=256, latent_dim=128):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Simple encoder: CNN 1D
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Global pooling
            nn.Flatten(),
            nn.Linear(256, hidden_dim)
        )
        
        # HRM-inspired components
        self.h_module = nn.LSTM(hidden_dim, hidden_dim//2, batch_first=True)
        self.l_module = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 512 * 3),
            nn.ReLU()
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # x: (batch, 512, 3) -> (batch, 3, 512) for Conv1d
        x = x.transpose(1, 2)
        
        # Encode
        encoded = self.encoder(x)  # (batch, hidden_dim)
        
        # HRM-inspired processing
        # L-Module (fast processing)
        l_input = encoded.unsqueeze(1)  # (batch, 1, hidden_dim)
        l_output, _ = self.l_module(l_input)
        l_output = l_output.squeeze(1)  # (batch, hidden_dim)
        
        # H-Module (high-level reasoning)
        h_input = l_output.unsqueeze(1)  # (batch, 1, hidden_dim)
        h_output, _ = self.h_module(h_input)
        h_output = h_output.squeeze(1)  # (batch, hidden_dim//2)
        
        # Combine L and H outputs
        combined = torch.cat([l_output, h_output], dim=1)  # (batch, hidden_dim + hidden_dim//2)
        
        # Project to latent space
        latent = self.output_proj(combined[:, :self.hidden_dim])  # (batch, latent_dim)
        
        # Decode for reconstruction
        recon = self.decoder(latent)  # (batch, 512*3)
        recon = recon.view(batch_size, 512, 3)  # (batch, 512, 3)
        
        return recon, latent


def train_simple_hrm():
    """Train simplified HRM model"""
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load dataset
    data_path = "./models/datasets/train_vae_enriched_512.json"
    train_dataset = HRMDataset(data_path, mode='train', validation_split=0.2)
    val_dataset = HRMDataset(data_path, mode='validation', validation_split=0.2)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Create model
    model = SimpleHRM(hidden_dim=256, latent_dim=128).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {total_params:,} parameters")
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # Training loop
    epochs = 20
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training"):
            data = batch['histogram'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            recon, latent = model(data)
            loss = criterion(recon, data)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                data = batch['histogram'].to(device)
                recon, latent = model(data)
                loss = criterion(recon, data)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Save model
    os.makedirs("./hrm_training_output/models", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': {
            'hidden_dim': 256,
            'latent_dim': 128,
            'total_params': total_params
        }
    }, "./hrm_training_output/models/simple_hrm_model.pth")
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Simple HRM Training Curves')
    plt.legend()
    plt.grid(True)
    
    os.makedirs("./hrm_training_output/plots", exist_ok=True)
    plt.savefig("./hrm_training_output/plots/training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Final results
    logger.info(f"Training completed!")
    logger.info(f"Final Train Loss: {train_losses[-1]:.6f}")
    logger.info(f"Final Val Loss: {val_losses[-1]:.6f}")
    logger.info(f"Model saved to: ./hrm_training_output/models/simple_hrm_model.pth")
    
    return model, train_losses, val_losses


if __name__ == "__main__":
    train_simple_hrm()