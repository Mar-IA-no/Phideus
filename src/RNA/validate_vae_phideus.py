#!/usr/bin/env python3
"""
Validate VAE Phideus - Sistema completo de validación para VAE entrenado

Métricas y visualizaciones para evaluar calidad del VAE:
- Reconstrucción de histogramas
- Análisis del espacio latente
- Interpolación entre muestras
- Clustering de embeddings
- Detección de ratios armónicos en latent space

Uso:
    python validate_vae_phideus.py --checkpoint best_model.pth --data test_data.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import seaborn as sns

# Imports locales
from vae_phideus_v1 import PhideusVAE, PhideusDataset, create_model_and_dataset

warnings.filterwarnings("ignore")


class VAEValidator:
    """Validador completo para VAE Phideus."""
    
    def __init__(self, model: PhideusVAE, device: str):
        self.model = model
        self.device = device
        self.model.eval()
        
        # Para guardar resultados
        self.latent_codes = []
        self.reconstructions = []
        self.original_samples = []
        self.file_names = []
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Cargar checkpoint del modelo."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint
    
    def encode_dataset(self, dataset: PhideusDataset) -> Dict[str, np.ndarray]:
        """Encodear todo el dataset al espacio latente."""
        print("🧠 Encoding dataset to latent space...")
        
        self.latent_codes = []
        self.reconstructions = []
        self.original_samples = []
        self.file_names = []
        
        with torch.no_grad():
            for i, sample in enumerate(dataset):
                sample = sample.unsqueeze(0).to(self.device)  # Add batch dim
                
                # Forward pass
                output = self.model(sample)
                
                # Store results
                self.latent_codes.append(output['z'].cpu().numpy())
                self.reconstructions.append(output['reconstruction'].cpu().numpy())
                self.original_samples.append(sample.cpu().numpy())
                self.file_names.append(dataset.files[i])
        
        # Concatenar todo
        results = {
            'latent_codes': np.concatenate(self.latent_codes, axis=0),
            'reconstructions': np.concatenate(self.reconstructions, axis=0),
            'originals': np.concatenate(self.original_samples, axis=0),
            'file_names': self.file_names
        }
        
        print(f"📊 Encoded {len(results['latent_codes'])} samples")
        print(f"🧠 Latent space shape: {results['latent_codes'].shape}")
        
        return results
    
    def compute_reconstruction_metrics(self, results: Dict) -> Dict[str, float]:
        """Métricas de reconstrucción."""
        originals = results['originals']
        reconstructions = results['reconstructions']
        
        # MSE por muestra
        mse_per_sample = np.mean((originals - reconstructions) ** 2, axis=(1, 2))
        
        # MSE por canal
        mse_per_channel = np.mean((originals - reconstructions) ** 2, axis=(0, 2))
        
        # Correlación por muestra
        correlations = []
        for i in range(len(originals)):
            orig_flat = originals[i].flatten()
            recon_flat = reconstructions[i].flatten()
            corr = np.corrcoef(orig_flat, recon_flat)[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0.0)
        
        metrics = {
            'mse_mean': float(np.mean(mse_per_sample)),
            'mse_std': float(np.std(mse_per_sample)),
            'mse_channel_0': float(mse_per_channel[0]),  # Proporción
            'mse_channel_1': float(mse_per_channel[1]),  # Energía
            'mse_channel_2': float(mse_per_channel[2]),  # Entropía
            'correlation_mean': float(np.mean(correlations)),
            'correlation_std': float(np.std(correlations)),
            'reconstruction_quality': 1.0 / (1.0 + np.mean(mse_per_sample))  # Métrica 0-1
        }
        
        return metrics
    
    def analyze_latent_space(self, results: Dict, save_dir: Path) -> Dict:
        """Análisis completo del espacio latente."""
        print("🔍 Analyzing latent space...")
        
        latent_codes = results['latent_codes']
        
        # Estadísticas básicas
        latent_stats = {
            'mean': np.mean(latent_codes, axis=0),
            'std': np.std(latent_codes, axis=0),
            'min': np.min(latent_codes, axis=0),
            'max': np.max(latent_codes, axis=0)
        }
        
        # PCA
        pca = PCA(n_components=min(10, latent_codes.shape[1]))
        latent_pca = pca.fit_transform(latent_codes)
        
        # t-SNE (si el dataset no es muy grande)
        latent_tsne = None
        if len(latent_codes) <= 500:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_codes)-1))
            latent_tsne = tsne.fit_transform(latent_codes)
        
        # Clustering
        n_clusters = min(5, len(latent_codes) // 3)
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(latent_codes)
        else:
            cluster_labels = np.zeros(len(latent_codes))
        
        # Visualizaciones
        self._plot_latent_analysis(latent_codes, latent_pca, latent_tsne, 
                                  cluster_labels, latent_stats, save_dir)
        
        return {
            'stats': latent_stats,
            'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
            'n_clusters': n_clusters,
            'cluster_labels': cluster_labels.tolist()
        }
    
    def _plot_latent_analysis(self, latent_codes, latent_pca, latent_tsne, 
                             cluster_labels, latent_stats, save_dir):
        """Plots de análisis del espacio latente."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Distribución de dimensiones latentes
        axes[0, 0].hist(latent_codes.flatten(), bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('Latent Code Distribution')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. PCA componentes principales
        if latent_pca.shape[1] >= 2:
            scatter = axes[0, 1].scatter(latent_pca[:, 0], latent_pca[:, 1], 
                                        c=cluster_labels, cmap='tab10', alpha=0.7)
            axes[0, 1].set_title('PCA Projection (PC1 vs PC2)')
            axes[0, 1].set_xlabel('PC1')
            axes[0, 1].set_ylabel('PC2')
            axes[0, 1].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[0, 1])
        
        # 3. t-SNE si disponible
        if latent_tsne is not None:
            scatter = axes[0, 2].scatter(latent_tsne[:, 0], latent_tsne[:, 1], 
                                        c=cluster_labels, cmap='tab10', alpha=0.7)
            axes[0, 2].set_title('t-SNE Projection')
            axes[0, 2].set_xlabel('t-SNE 1')
            axes[0, 2].set_ylabel('t-SNE 2')
            plt.colorbar(scatter, ax=axes[0, 2])
        else:
            axes[0, 2].text(0.5, 0.5, 'Dataset too large\nfor t-SNE', 
                           ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('t-SNE (Skipped)')
        
        # 4. Varianza por dimensión
        axes[1, 0].plot(latent_stats['std'][:min(50, len(latent_stats['std']))], 'o-')
        axes[1, 0].set_title('Latent Dimension Std Dev')
        axes[1, 0].set_xlabel('Dimension')
        axes[1, 0].set_ylabel('Std Dev')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Matriz de correlación (primeras 20 dims)
        n_dims = min(20, latent_codes.shape[1])
        corr_matrix = np.corrcoef(latent_codes[:, :n_dims].T)
        im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[1, 1].set_title(f'Latent Correlation Matrix (first {n_dims} dims)')
        axes[1, 1].set_xlabel('Dimension')
        axes[1, 1].set_ylabel('Dimension')
        plt.colorbar(im, ax=axes[1, 1])
        
        # 6. Distribución por cluster
        if len(np.unique(cluster_labels)) > 1:
            for cluster in np.unique(cluster_labels):
                cluster_data = latent_codes[cluster_labels == cluster]
                axes[1, 2].hist(cluster_data.mean(axis=1), bins=20, 
                               alpha=0.6, label=f'Cluster {cluster}')
            axes[1, 2].set_title('Mean Latent Value by Cluster')
            axes[1, 2].set_xlabel('Mean Latent Value')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, 'No clustering\nperformed', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'latent_space_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_reconstructions(self, results: Dict, save_dir: Path, n_samples: int = 6):
        """Plot comparación original vs reconstrucción."""
        print(f"🎨 Plotting {n_samples} reconstruction examples...")
        
        # Seleccionar muestras aleatoriamente
        indices = np.random.choice(len(results['originals']), n_samples, replace=False)
        
        fig, axes = plt.subplots(n_samples, 2, figsize=(15, 3*n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, idx in enumerate(indices):
            original = results['originals'][idx]  # (3, 512)
            reconstruction = results['reconstructions'][idx]  # (3, 512)
            filename = results['file_names'][idx]
            
            # Plot original
            axes[i, 0].plot(original[0], 'b-', label='Proporción', alpha=0.8)
            axes[i, 0].plot(original[1], 'g-', label='Energía', alpha=0.8)  
            axes[i, 0].plot(original[2], 'r-', label='Entropía', alpha=0.8)
            axes[i, 0].set_title(f'Original: {filename}')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            
            # Plot reconstrucción
            axes[i, 1].plot(reconstruction[0], 'b--', label='Proporción', alpha=0.8)
            axes[i, 1].plot(reconstruction[1], 'g--', label='Energía', alpha=0.8)
            axes[i, 1].plot(reconstruction[2], 'r--', label='Entropía', alpha=0.8)
            axes[i, 1].set_title(f'Reconstruction (MSE: {np.mean((original-reconstruction)**2):.4f})')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'reconstructions.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def interpolate_latent(self, results: Dict, save_dir: Path, n_steps: int = 8):
        """Interpolación en espacio latente."""
        print("🌈 Performing latent space interpolation...")
        
        latent_codes = results['latent_codes']
        
        # Seleccionar dos muestras aleatorias
        idx1, idx2 = np.random.choice(len(latent_codes), 2, replace=False)
        z1 = torch.tensor(latent_codes[idx1], device=self.device).unsqueeze(0)
        z2 = torch.tensor(latent_codes[idx2], device=self.device).unsqueeze(0)
        
        # Interpolación lineal
        alphas = np.linspace(0, 1, n_steps)
        interpolations = []
        
        with torch.no_grad():
            for alpha in alphas:
                z_interp = (1 - alpha) * z1 + alpha * z2
                reconstruction = self.model.decode(z_interp)
                interpolations.append(reconstruction.cpu().numpy()[0])
        
        # Plot
        fig, axes = plt.subplots(3, n_steps, figsize=(2*n_steps, 9))
        channel_names = ['Proporción', 'Energía', 'Entropía']
        
        for i in range(3):  # Canales
            for j in range(n_steps):  # Interpolación
                axes[i, j].plot(interpolations[j][i])
                if i == 0:
                    axes[i, j].set_title(f'α={alphas[j]:.2f}')
                if j == 0:
                    axes[i, j].set_ylabel(channel_names[i])
                axes[i, j].grid(True, alpha=0.3)
        
        plt.suptitle(f'Latent Interpolation: {results["file_names"][idx1]} → {results["file_names"][idx2]}')
        plt.tight_layout()
        plt.savefig(save_dir / 'latent_interpolation.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_validation_report(self, results: Dict, metrics: Dict, 
                                  latent_analysis: Dict, save_dir: Path):
        """Generar reporte completo de validación."""
        
        report = {
            'dataset_info': {
                'n_samples': len(results['latent_codes']),
                'latent_dim': results['latent_codes'].shape[1],
                'input_shape': results['originals'].shape[1:]
            },
            'reconstruction_metrics': metrics,
            'latent_analysis': {
                'pca_explained_variance_top5': latent_analysis['pca_explained_variance'][:5],
                'n_clusters': latent_analysis['n_clusters'],
                'latent_std_mean': float(np.mean(latent_analysis['stats']['std'])),
                'latent_range': float(np.max(latent_analysis['stats']['max']) - 
                                    np.min(latent_analysis['stats']['min']))
            }
        }
        
        # Guardar reporte
        with open(save_dir / 'validation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Imprimir resumen
        print("\n" + "="*60)
        print("📊 VAE VALIDATION REPORT")
        print("="*60)
        print(f"🎯 Dataset: {report['dataset_info']['n_samples']} samples")
        print(f"🧠 Latent space: {report['dataset_info']['latent_dim']} dimensions")
        print(f"🔄 Reconstruction quality: {metrics['reconstruction_quality']:.3f}")
        print(f"📉 MSE mean: {metrics['mse_mean']:.6f}")
        print(f"📈 Correlation mean: {metrics['correlation_mean']:.3f}")
        print(f"🎨 PCA variance (top 5): {latent_analysis['pca_explained_variance'][:5]}")
        print(f"🏷️  Clusters found: {latent_analysis['n_clusters']}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Validate VAE Phideus')
    parser.add_argument('--checkpoint', type=Path, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=Path, required=True,
                       help='Path to validation JSON dataset')
    parser.add_argument('--save-dir', type=Path, default=Path('vae_validation'),
                       help='Directory to save validation results')
    parser.add_argument('--use-attention', action='store_true',
                       help='Use linear attention in model')
    
    args = parser.parse_args()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔍 Validating VAE Phideus on {device}")
    
    args.save_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear modelo y dataset
    model, dataset = create_model_and_dataset(args.data, device, args.use_attention)
    
    # Validador
    validator = VAEValidator(model, device)
    checkpoint = validator.load_checkpoint(args.checkpoint)
    
    # Análisis completo
    print("\n🚀 Starting VAE validation...")
    
    # 1. Encodear dataset
    results = validator.encode_dataset(dataset)
    
    # 2. Métricas de reconstrucción
    metrics = validator.compute_reconstruction_metrics(results)
    
    # 3. Análisis espacio latente
    latent_analysis = validator.analyze_latent_space(results, args.save_dir)
    
    # 4. Visualizaciones
    validator.plot_reconstructions(results, args.save_dir)
    validator.interpolate_latent(results, args.save_dir)
    
    # 5. Reporte final
    validator.generate_validation_report(results, metrics, latent_analysis, args.save_dir)
    
    print(f"\n✅ Validation completed! Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()