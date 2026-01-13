#!/usr/bin/env python3
"""
Experimentos Rigurosos - Analizador 5.0
========================================

Implementa los 4 experimentos del diseño científico:

E1: HRM-temporal   - HRM con secuencias completas [T, B, 3]
E2: VAE-temporal   - VAE con LSTM para secuencias [T, B, 3]
E3: HRM-estático   - HRM con datos promediados [B, 3] (control)
E4: VAE-estático   - VAE con datos promediados [B, 3] (control)

Esto permite evaluar:
- Efecto de la temporalidad (E1 vs E3, E2 vs E4)
- Superioridad arquitectónica (E1 vs E2, E3 vs E4)
- Interacción arquitectura × temporalidad
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

from src.datasets.temporal_dataset_5 import TemporalDataset5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================
# ARQUITECTURAS
# ============================================

class HRMTemporal(nn.Module):
    """
    E1: HRM con procesamiento temporal completo.
    Procesa secuencias [batch, T, B, 3] usando GRU + LSTM + Attention.
    """

    def __init__(self, n_bins=256, max_frames=100, l_hidden=256, h_hidden=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_bins = n_bins
        self.max_frames = max_frames

        # Encoder por frame
        self.frame_encoder = nn.Sequential(
            nn.Linear(n_bins * 3, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # L-Module: GRU rápido
        self.l_module = nn.GRU(
            input_size=256,
            hidden_size=l_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

        # H-Module: LSTM para razonamiento
        self.h_module = nn.LSTM(
            input_size=l_hidden,
            hidden_size=h_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=h_hidden,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(h_hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, n_bins * 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        """x: [batch, T, B, 3] -> [batch, B, 3]"""
        batch_size, T, B, C = x.shape

        # Flatten frames: [batch, T, B*3]
        x_flat = x.view(batch_size, T, -1)

        # Encode cada frame
        frame_features = self.frame_encoder(x_flat)  # [batch, T, 256]

        # L-Module
        l_out, _ = self.l_module(frame_features)  # [batch, T, l_hidden]

        # H-Module
        h_out, _ = self.h_module(l_out)  # [batch, T, h_hidden]

        # Attention
        attn_out, _ = self.attention(h_out, h_out, h_out)

        # Agregación: último + promedio
        final = attn_out[:, -1, :] + attn_out.mean(dim=1)

        # Decode
        recon = self.decoder(final).view(batch_size, B, C)

        return recon


class VAETemporal(nn.Module):
    """
    E2: VAE con procesamiento temporal (LSTM).
    Procesa secuencias [batch, T, B, 3] con LSTM encoder.
    """

    def __init__(self, n_bins=256, max_frames=100, latent_dim=64, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.n_bins = n_bins
        self.latent_dim = latent_dim

        # Encoder temporal
        self.frame_encoder = nn.Sequential(
            nn.Linear(n_bins * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, n_bins * 3),
            nn.Sigmoid()
        )

    def encode(self, x):
        """x: [batch, T, B, 3]"""
        batch_size, T, B, C = x.shape
        x_flat = x.view(batch_size, T, -1)

        # Encode frames
        frame_features = self.frame_encoder(x_flat)

        # LSTM
        _, (h_n, _) = self.lstm_encoder(frame_features)
        h_final = h_n[-1]  # Último hidden state

        return self.fc_mu(h_final), self.fc_logvar(h_final)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        batch_size = x.size(0)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z).view(batch_size, self.n_bins, 3)
        return recon, mu, logvar


class HRMStatic(nn.Module):
    """
    E3: HRM con datos estáticos (control).
    Procesa datos promediados [batch, B, 3].
    Usa la misma arquitectura jerárquica pero sin temporalidad.
    """

    def __init__(self, n_bins=256, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.n_bins = n_bins

        # Encoder jerárquico (simula L-Module + H-Module sin temporalidad)
        self.encoder = nn.Sequential(
            nn.Linear(n_bins * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, n_bins * 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        """x: [batch, B, 3] -> [batch, B, 3]"""
        batch_size, B, C = x.shape
        x_flat = x.view(batch_size, -1)
        encoded = self.encoder(x_flat)
        recon = self.decoder(encoded).view(batch_size, B, C)
        return recon


class VAEStatic(nn.Module):
    """
    E4: VAE con datos estáticos (control).
    Procesa datos promediados [batch, B, 3].
    """

    def __init__(self, n_bins=256, latent_dim=64, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.n_bins = n_bins
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(n_bins * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, n_bins * 3),
            nn.Sigmoid()
        )

    def encode(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        h = self.encoder(x_flat)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size = x.size(0)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z).view(batch_size, self.n_bins, 3)
        return recon, mu, logvar


# ============================================
# FUNCIONES DE ENTRENAMIENTO
# ============================================

def train_epoch(model, loader, optimizer, device, is_vae=False, use_amp=True):
    """Entrena una época."""
    model.train()
    total_loss = 0.0
    scaler = GradScaler() if use_amp else None

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Target: para temporales, promedio de frames
        if len(batch.shape) == 4:  # [batch, T, B, 3]
            target = batch.mean(dim=1)  # [batch, B, 3]
        else:
            target = batch

        if use_amp:
            with autocast('cuda'):
                if is_vae:
                    recon, mu, logvar = model(batch)
                    recon_loss = nn.functional.mse_loss(recon, target, reduction='sum') / batch.size(0)
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch.size(0)
                    loss = recon_loss + kl_loss
                else:
                    recon = model(batch)
                    loss = nn.functional.mse_loss(recon, target, reduction='sum') / batch.size(0)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            if is_vae:
                recon, mu, logvar = model(batch)
                recon_loss = nn.functional.mse_loss(recon, target, reduction='sum') / batch.size(0)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch.size(0)
                loss = recon_loss + kl_loss
            else:
                recon = model(batch)
                loss = nn.functional.mse_loss(recon, target, reduction='sum') / batch.size(0)

            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device, is_vae=False):
    """Valida el modelo."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            if len(batch.shape) == 4:
                target = batch.mean(dim=1)
            else:
                target = batch

            if is_vae:
                recon, mu, logvar = model(batch)
                recon_loss = nn.functional.mse_loss(recon, target, reduction='sum') / batch.size(0)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch.size(0)
                loss = recon_loss + kl_loss
            else:
                recon = model(batch)
                loss = nn.functional.mse_loss(recon, target, reduction='sum') / batch.size(0)

            total_loss += loss.item()

    return total_loss / len(loader)


def run_experiment(name, model, train_loader, val_loader, device, epochs=50, lr=1e-3, is_vae=False):
    """Ejecuta un experimento completo."""
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENTO: {name}")
    logger.info(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"{'='*60}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, is_vae)
        val_loss = validate(model, val_loader, device, is_vae)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, Best: {best_val_loss:.4f}")

    return {
        'name': name,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'total_params': sum(p.numel() for p in model.parameters())
    }


# ============================================
# EJECUCIÓN PRINCIPAL
# ============================================

def main(data_path: str, output_dir: str, epochs: int = 50, batch_size: int = 32, max_frames: int = 100):
    """Ejecuta los 4 experimentos."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Cargar datasets
    logger.info("\n=== Cargando Datasets ===")

    # Temporal (para E1, E2)
    ds_temporal = TemporalDataset5(data_path, strategy='sequence', max_frames=max_frames)
    n_bins = ds_temporal.n_bins

    # Estático (para E3, E4)
    ds_static = TemporalDataset5(data_path, strategy='average')

    # Splits
    n_total = len(ds_temporal)
    n_train = int(0.85 * n_total)
    n_val = n_total - n_train

    gen = torch.Generator().manual_seed(42)

    train_temp, val_temp = random_split(ds_temporal, [n_train, n_val], generator=gen)
    train_stat, val_stat = random_split(ds_static, [n_train, n_val], generator=gen)

    loader_train_temp = DataLoader(train_temp, batch_size=batch_size, shuffle=True)
    loader_val_temp = DataLoader(val_temp, batch_size=batch_size)
    loader_train_stat = DataLoader(train_stat, batch_size=batch_size, shuffle=True)
    loader_val_stat = DataLoader(val_stat, batch_size=batch_size)

    logger.info(f"Dataset: {n_total} archivos ({n_train} train / {n_val} val)")
    logger.info(f"Bins: {n_bins}, Max frames: {max_frames}")

    results = {}

    # E1: HRM Temporal
    model_e1 = HRMTemporal(n_bins=n_bins, max_frames=max_frames).to(device)
    results['E1_HRM_temporal'] = run_experiment(
        "E1: HRM Temporal", model_e1, loader_train_temp, loader_val_temp,
        device, epochs, is_vae=False
    )

    # E2: VAE Temporal
    model_e2 = VAETemporal(n_bins=n_bins, max_frames=max_frames).to(device)
    results['E2_VAE_temporal'] = run_experiment(
        "E2: VAE Temporal", model_e2, loader_train_temp, loader_val_temp,
        device, epochs, is_vae=True
    )

    # E3: HRM Estático
    model_e3 = HRMStatic(n_bins=n_bins).to(device)
    results['E3_HRM_static'] = run_experiment(
        "E3: HRM Estático", model_e3, loader_train_stat, loader_val_stat,
        device, epochs, is_vae=False
    )

    # E4: VAE Estático
    model_e4 = VAEStatic(n_bins=n_bins).to(device)
    results['E4_VAE_static'] = run_experiment(
        "E4: VAE Estático", model_e4, loader_train_stat, loader_val_stat,
        device, epochs, is_vae=True
    )

    # Generar reporte
    generate_report(results, output_dir, n_bins, max_frames, n_total, epochs)

    return results


def generate_report(results, output_dir, n_bins, max_frames, n_samples, epochs):
    """Genera visualizaciones y reporte."""

    # Visualización
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {'E1_HRM_temporal': '#2ecc71', 'E2_VAE_temporal': '#3498db',
              'E3_HRM_static': '#e74c3c', 'E4_VAE_static': '#9b59b6'}

    # 1. Curvas de validación
    ax = axes[0, 0]
    for name, res in results.items():
        ax.plot(res['val_losses'], label=name.replace('_', ' '), color=colors[name])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Training Curves')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 2. Comparación final
    ax = axes[0, 1]
    names = list(results.keys())
    losses = [results[n]['best_val_loss'] for n in names]
    bars = ax.bar(range(len(names)), losses, color=[colors[n] for n in names])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=8)
    ax.set_ylabel('Best Validation Loss')
    ax.set_title('Final Performance')
    ax.set_yscale('log')

    # 3. Efecto temporalidad (E1 vs E3, E2 vs E4)
    ax = axes[1, 0]
    hrm_temp = results['E1_HRM_temporal']['best_val_loss']
    hrm_stat = results['E3_HRM_static']['best_val_loss']
    vae_temp = results['E2_VAE_temporal']['best_val_loss']
    vae_stat = results['E4_VAE_static']['best_val_loss']

    hrm_effect = (hrm_stat - hrm_temp) / hrm_stat * 100
    vae_effect = (vae_stat - vae_temp) / vae_stat * 100

    bars = ax.bar(['HRM\n(E1 vs E3)', 'VAE\n(E2 vs E4)'], [hrm_effect, vae_effect],
                  color=['#2ecc71', '#3498db'])
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Mejora por Temporalidad (%)')
    ax.set_title('Efecto de la Dimensión Temporal')
    for bar, val in zip(bars, [hrm_effect, vae_effect]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}%',
                ha='center', va='bottom')

    # 4. HRM vs VAE
    ax = axes[1, 1]
    temp_diff = (vae_temp - hrm_temp) / vae_temp * 100
    stat_diff = (vae_stat - hrm_stat) / vae_stat * 100

    bars = ax.bar(['Temporal\n(E1 vs E2)', 'Estático\n(E3 vs E4)'], [temp_diff, stat_diff],
                  color=['#27ae60', '#c0392b'])
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Ventaja HRM sobre VAE (%)')
    ax.set_title('Superioridad Arquitectónica')
    for bar, val in zip(bars, [temp_diff, stat_diff]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}%',
                ha='center', va='bottom')

    plt.suptitle(f'Experimentos Rigurosos - Analizador 5.0\n({n_samples} muestras, {epochs} epochs)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'experiments_5.0.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Reporte markdown
    report = f"""# Resultados Experimentos Rigurosos - Analizador 5.0

## Configuración
- **Muestras**: {n_samples}
- **Bins**: {n_bins}
- **Max frames**: {max_frames}
- **Epochs**: {epochs}
- **Fecha**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Resultados por Experimento

| Experimento | Descripción | Val Loss | Parámetros |
|-------------|-------------|----------|------------|
| E1 | HRM Temporal | {results['E1_HRM_temporal']['best_val_loss']:.4f} | {results['E1_HRM_temporal']['total_params']:,} |
| E2 | VAE Temporal | {results['E2_VAE_temporal']['best_val_loss']:.4f} | {results['E2_VAE_temporal']['total_params']:,} |
| E3 | HRM Estático | {results['E3_HRM_static']['best_val_loss']:.4f} | {results['E3_HRM_static']['total_params']:,} |
| E4 | VAE Estático | {results['E4_VAE_static']['best_val_loss']:.4f} | {results['E4_VAE_static']['total_params']:,} |

## Análisis

### Efecto de la Temporalidad
- **HRM**: {(hrm_stat - hrm_temp) / hrm_stat * 100:+.1f}% (E1 vs E3)
- **VAE**: {(vae_stat - vae_temp) / vae_stat * 100:+.1f}% (E2 vs E4)

### Superioridad Arquitectónica (HRM vs VAE)
- **Con temporalidad**: HRM {(vae_temp - hrm_temp) / vae_temp * 100:+.1f}% mejor
- **Sin temporalidad**: HRM {(vae_stat - hrm_stat) / vae_stat * 100:+.1f}% mejor

## Conclusiones

1. **Ganador general**: {'E1 (HRM Temporal)' if hrm_temp == min(losses) else 'Otro'}
2. **La temporalidad ayuda**: {'Sí' if hrm_effect > 0 or vae_effect > 0 else 'No significativamente'}
3. **HRM > VAE**: {'Confirmado' if hrm_temp < vae_temp and hrm_stat < vae_stat else 'Parcial'}
"""

    with open(output_dir / 'report_experiments_5.0.md', 'w') as f:
        f.write(report)

    # Guardar resultados JSON
    results_json = {k: {kk: vv for kk, vv in v.items() if kk != 'train_losses' and kk != 'val_losses'}
                    for k, v in results.items()}
    with open(output_dir / 'results_5.0.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    logger.info(f"\nResultados guardados en: {output_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Experimentos rigurosos Analizador 5.0')
    parser.add_argument('--data', type=str, required=True, help='Dataset .npz')
    parser.add_argument('--output', type=str, default='data/training_outputs/experiments_5.0')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-frames', type=int, default=100)

    args = parser.parse_args()

    main(args.data, args.output, args.epochs, args.batch_size, args.max_frames)
