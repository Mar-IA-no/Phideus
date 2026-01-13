#!/usr/bin/env python3
"""
Comparación Tri-Arquitectura con Analizador 5.0
================================================

Entrena y compara las tres arquitecturas con datos del Analizador 5.0:
1. Baseline VAE (promedio temporal)
2. Enhanced VAE (promedio temporal)
3. Enhanced HRM (secuencias temporales completas)

Este script replica las pruebas realizadas con 4.1 pero usando el formato temporal.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import logging
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

# Dataset 5.0
from src.datasets.temporal_dataset_5 import TemporalDataset5

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================
# ARQUITECTURAS
# ============================================

class BaselineVAE5(nn.Module):
    """VAE Baseline para datos 5.0 (promedio temporal -> [B, 3])"""

    def __init__(self, n_bins=256, latent_dim=128, base_channels=64):
        super().__init__()
        self.n_bins = n_bins
        self.latent_dim = latent_dim

        # Encoder CNN simple
        self.encoder = nn.Sequential(
            nn.Conv1d(3, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.Conv1d(base_channels, base_channels*2, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels*2),
            nn.ReLU(),
            nn.Conv1d(base_channels*2, base_channels*4, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels*4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(base_channels*4, latent_dim)
        self.fc_logvar = nn.Linear(base_channels*4, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, n_bins * 3),
            nn.Sigmoid()
        )

    def encode(self, x):
        # x: [batch, B, 3] -> [batch, 3, B]
        x = x.transpose(1, 2)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z).view(-1, self.n_bins, 3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class EnhancedVAE5(nn.Module):
    """VAE Mejorado para datos 5.0 (promedio temporal -> [B, 3])"""

    def __init__(self, n_bins=256, latent_dim=128, base_channels=64, dropout=0.1):
        super().__init__()
        self.n_bins = n_bins
        self.latent_dim = latent_dim

        # Encoder más profundo
        self.encoder = nn.Sequential(
            nn.Conv1d(3, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(base_channels, base_channels*2, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(base_channels*2, base_channels*4, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels*4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(base_channels*4, base_channels*6, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels*6),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(base_channels*6, latent_dim)
        self.fc_logvar = nn.Linear(base_channels*6, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, base_channels*6),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(base_channels*6, base_channels*8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(base_channels*8, n_bins * 3),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = x.transpose(1, 2)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z).view(-1, self.n_bins, 3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class EnhancedHRM5(nn.Module):
    """
    HRM Mejorado para datos 5.0 (secuencias temporales [T, B, 3])
    Aprovecha la dimensión temporal con módulos GRU/LSTM + Attention.
    """

    def __init__(self, n_bins=256, max_frames=100, latent_dim=128,
                 l_hidden=384, h_hidden=192, n_heads=8, dropout=0.1):
        super().__init__()
        self.n_bins = n_bins
        self.max_frames = max_frames
        self.latent_dim = latent_dim
        self.input_dim = n_bins * 3

        # CNN Encoder (procesa cada frame)
        self.frame_encoder = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        # L-Module: Procesamiento rápido (GRU)
        self.l_module = nn.GRU(
            input_size=384,
            hidden_size=l_hidden,
            num_layers=3,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

        # H-Module: Razonamiento de alto nivel (LSTM + Attention)
        self.h_module = nn.LSTM(
            input_size=l_hidden,
            hidden_size=h_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

        # Multi-head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=h_hidden,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Fusión jerárquica
        self.fusion = nn.Sequential(
            nn.Linear(h_hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 384),
            nn.ReLU(),
            nn.Linear(384, 768),
            nn.ReLU(),
            nn.Linear(768, n_bins * 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: [batch, T, B, 3] - Secuencia temporal de histogramas
        Returns:
            recon: [batch, B, 3] - Reconstrucción (promedio temporal)
        """
        batch_size, T, B, C = x.shape

        # Encode cada frame: [batch*T, 384]
        x_flat = x.view(batch_size * T, B, C)
        x_flat = x_flat.transpose(1, 2)  # [batch*T, 3, B]
        frame_features = self.frame_encoder(x_flat)  # [batch*T, 384]
        frame_features = frame_features.view(batch_size, T, -1)  # [batch, T, 384]

        # L-Module: Procesamiento rápido
        l_out, _ = self.l_module(frame_features)  # [batch, T, l_hidden]

        # H-Module: Razonamiento temporal
        h_out, _ = self.h_module(l_out)  # [batch, T, h_hidden]

        # Attention: Capturar relaciones globales
        attn_out, _ = self.attention(h_out, h_out, h_out)  # [batch, T, h_hidden]

        # Agregación: Usar último estado + promedio
        final_state = attn_out[:, -1, :]  # [batch, h_hidden]
        avg_state = attn_out.mean(dim=1)  # [batch, h_hidden]
        combined = final_state + avg_state  # Skip connection

        # Fusión y latente
        latent = self.fusion(combined)  # [batch, latent_dim]

        # Decoder
        recon = self.decoder(latent).view(batch_size, B, C)

        return recon


# ============================================
# FUNCIONES DE PÉRDIDA
# ============================================

def vae_loss(recon, target, mu, logvar, beta=1.0):
    """Pérdida VAE: Reconstrucción + KL Divergence"""
    recon_loss = nn.functional.mse_loss(recon, target, reduction='sum') / target.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.size(0)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def hrm_loss(recon, target):
    """Pérdida HRM: MSE directo (para comparación justa, usa el promedio temporal como target)"""
    return nn.functional.mse_loss(recon, target, reduction='sum') / target.size(0)


# ============================================
# ENTRENAMIENTO
# ============================================

def train_model(model, train_loader, val_loader, epochs, device, model_name,
                is_vae=True, lr=1e-3, use_amp=True):
    """Entrena un modelo y retorna métricas."""

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    scaler = GradScaler() if use_amp else None

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_state = None

    logger.info(f"\n{'='*60}")
    logger.info(f"Entrenando: {model_name}")
    logger.info(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"{'='*60}")

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()

            if use_amp:
                with autocast('cuda'):
                    if is_vae:
                        # VAE usa datos estáticos [batch, B, 3]
                        recon, mu, logvar = model(batch)
                        loss, _, _ = vae_loss(recon, batch, mu, logvar)
                    else:
                        # HRM usa secuencias [batch, T, B, 3]
                        # Target: promedio temporal de los frames de entrada
                        target = batch.mean(dim=1)  # [batch, B, 3]
                        recon = model(batch)
                        loss = hrm_loss(recon, target)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                if is_vae:
                    recon, mu, logvar = model(batch)
                    loss, _, _ = vae_loss(recon, batch, mu, logvar)
                else:
                    target = batch.mean(dim=1)
                    recon = model(batch)
                    loss = hrm_loss(recon, target)

                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                if is_vae:
                    recon, mu, logvar = model(batch)
                    loss, _, _ = vae_loss(recon, batch, mu, logvar)
                else:
                    target = batch.mean(dim=1)
                    recon = model(batch)
                    loss = hrm_loss(recon, target)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, Best: {best_val_loss:.4f}")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'best_state': best_state,
        'total_params': sum(p.numel() for p in model.parameters())
    }


# ============================================
# COMPARACIÓN PRINCIPAL
# ============================================

def run_comparison(json_path: str, output_dir: str, epochs: int = 50,
                   batch_size: int = 32, max_frames: int = 100):
    """Ejecuta la comparación tri-arquitectura."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Cargar datasets
    logger.info("\n=== Cargando Datasets ===")

    # Dataset para VAE (promedio temporal)
    dataset_vae = TemporalDataset5(json_path, strategy='average')
    n_bins = dataset_vae.n_bins

    # Dataset para HRM (secuencias)
    dataset_hrm = TemporalDataset5(json_path, strategy='sequence', max_frames=max_frames)

    # Splits (mismo para ambos)
    n_total = len(dataset_vae)
    n_train = int(0.85 * n_total)
    n_val = n_total - n_train

    # Para VAE
    train_vae, val_vae = random_split(dataset_vae, [n_train, n_val],
                                       generator=torch.Generator().manual_seed(42))
    train_loader_vae = DataLoader(train_vae, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader_vae = DataLoader(val_vae, batch_size=batch_size, shuffle=False, num_workers=0)

    # Para HRM (mismo split)
    train_hrm, val_hrm = random_split(dataset_hrm, [n_train, n_val],
                                       generator=torch.Generator().manual_seed(42))
    train_loader_hrm = DataLoader(train_hrm, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader_hrm = DataLoader(val_hrm, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info(f"Dataset: {n_total} archivos ({n_train} train / {n_val} val)")
    logger.info(f"Bins: {n_bins}, Max frames: {max_frames}")

    results = {}

    # 1. Baseline VAE
    logger.info("\n" + "="*60)
    logger.info("1/3 - BASELINE VAE")
    logger.info("="*60)
    model_base = BaselineVAE5(n_bins=n_bins).to(device)
    results['baseline_vae'] = train_model(
        model_base, train_loader_vae, val_loader_vae,
        epochs, device, "Baseline VAE 5.0", is_vae=True
    )

    # 2. Enhanced VAE
    logger.info("\n" + "="*60)
    logger.info("2/3 - ENHANCED VAE")
    logger.info("="*60)
    model_enhanced = EnhancedVAE5(n_bins=n_bins).to(device)
    results['enhanced_vae'] = train_model(
        model_enhanced, train_loader_vae, val_loader_vae,
        epochs, device, "Enhanced VAE 5.0", is_vae=True
    )

    # 3. Enhanced HRM
    logger.info("\n" + "="*60)
    logger.info("3/3 - ENHANCED HRM")
    logger.info("="*60)
    model_hrm = EnhancedHRM5(n_bins=n_bins, max_frames=max_frames).to(device)
    results['enhanced_hrm'] = train_model(
        model_hrm, train_loader_hrm, val_loader_hrm,
        epochs, device, "Enhanced HRM 5.0", is_vae=False
    )

    # Generar reporte
    generate_report(results, output_dir, n_bins, max_frames, n_total, epochs)

    return results


def generate_report(results, output_dir, n_bins, max_frames, n_samples, epochs):
    """Genera reporte y visualizaciones."""

    # Visualización
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Curvas de entrenamiento
    ax = axes[0, 0]
    for name, res in results.items():
        ax.plot(res['val_losses'], label=f"{name} (val)")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Loss Curves')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 2. Comparación final
    ax = axes[0, 1]
    names = list(results.keys())
    losses = [results[n]['best_val_loss'] for n in names]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax.bar(names, losses, color=colors)
    ax.set_ylabel('Best Validation Loss')
    ax.set_title('Final Performance Comparison')
    ax.set_yscale('log')
    for bar, loss in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width()/2, loss * 1.1, f'{loss:.2f}',
                ha='center', va='bottom', fontsize=9)

    # 3. Parámetros
    ax = axes[1, 0]
    params = [results[n]['total_params'] / 1e6 for n in names]
    bars = ax.bar(names, params, color=colors)
    ax.set_ylabel('Parameters (M)')
    ax.set_title('Model Parameters')
    for bar, p in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width()/2, p + 0.1, f'{p:.2f}M',
                ha='center', va='bottom', fontsize=9)

    # 4. Eficiencia
    ax = axes[1, 1]
    efficiency = [results[n]['best_val_loss'] / (results[n]['total_params'] / 1e6) for n in names]
    bars = ax.bar(names, efficiency, color=colors)
    ax.set_ylabel('Loss / Million Params')
    ax.set_title('Parameter Efficiency')
    ax.set_yscale('log')

    plt.suptitle(f'Comparación Tri-Arquitectura - Analizador 5.0\n({n_samples} muestras, {n_bins} bins, {epochs} epochs)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_5.0.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Reporte markdown
    hrm_loss = results['enhanced_hrm']['best_val_loss']
    report = f"""# Comparación Tri-Arquitectura - Analizador 5.0

## Configuración
- **Muestras**: {n_samples}
- **Bins**: {n_bins} (escala lineal)
- **Max frames**: {max_frames}
- **Epochs**: {epochs}
- **Fecha**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Resultados

| Arquitectura | Val Loss | Parámetros | Eficiencia | vs HRM |
|--------------|----------|------------|------------|--------|
| Baseline VAE | {results['baseline_vae']['best_val_loss']:.4f} | {results['baseline_vae']['total_params']/1e6:.2f}M | {results['baseline_vae']['best_val_loss']/(results['baseline_vae']['total_params']/1e6):.2f} | {((results['baseline_vae']['best_val_loss'] - hrm_loss) / hrm_loss * 100):+.1f}% |
| Enhanced VAE | {results['enhanced_vae']['best_val_loss']:.4f} | {results['enhanced_vae']['total_params']/1e6:.2f}M | {results['enhanced_vae']['best_val_loss']/(results['enhanced_vae']['total_params']/1e6):.2f} | {((results['enhanced_vae']['best_val_loss'] - hrm_loss) / hrm_loss * 100):+.1f}% |
| Enhanced HRM | {results['enhanced_hrm']['best_val_loss']:.4f} | {results['enhanced_hrm']['total_params']/1e6:.2f}M | {results['enhanced_hrm']['best_val_loss']/(results['enhanced_hrm']['total_params']/1e6):.4f} | Baseline |

## Mejora VAE Enhanced vs Baseline
{((results['baseline_vae']['best_val_loss'] - results['enhanced_vae']['best_val_loss']) / results['baseline_vae']['best_val_loss'] * 100):.3f}%

## Ganador
**{'Enhanced HRM' if hrm_loss == min(losses) else 'VAE'}** con val loss de {min(losses):.4f}
"""

    with open(output_dir / 'report_5.0.md', 'w') as f:
        f.write(report)

    logger.info(f"\nReporte guardado en: {output_dir}")
    logger.info(f"  - comparison_5.0.png")
    logger.info(f"  - report_5.0.md")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Comparación tri-arquitectura con Analizador 5.0')
    parser.add_argument('--json', type=str, required=True, help='JSON del dataset 5.0')
    parser.add_argument('--output', type=str, default='data/training_outputs/comparisons/comparison_5.0',
                        help='Directorio de salida')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs de entrenamiento')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--max-frames', type=int, default=100, help='Max frames para HRM')

    args = parser.parse_args()

    run_comparison(
        json_path=args.json,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_frames=args.max_frames
    )
