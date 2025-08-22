#!/usr/bin/env python3
"""
Script simplificado para usar Attention-Based Temporal VAE
Una vez entrenado, permite analizar archivos de audio temporalmente
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
import time

from attention_temporal_vae import RTX3090OptimizedTemporalVAE
from temporal_dataset import TemporalHistogramDataset

class TemporalAnalyzer:
    """
    Analizador temporal usando modelo VAE entrenado
    """
    def __init__(self, model_path, config_path=None, device='auto'):
        self.device = self._setup_device(device)
        self.config = self._load_config(config_path)
        self.model = self._load_model(model_path)
        
    def _setup_device(self, device):
        """Setup del device de computación"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
        
    def _load_config(self, config_path):
        """Cargar configuración"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Configuración por defecto
        return {
            "data": {
                "window_size": 1.0,
                "overlap": 0.5,
                "sample_rate": 44100
            },
            "model": {
                "max_sequence_length": 60
            }
        }
        
    def _load_model(self, model_path):
        """Cargar modelo entrenado"""
        print(f"Loading model from {model_path}...")
        
        model = RTX3090OptimizedTemporalVAE()
        
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Manejar diferentes formatos de checkpoint
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            print("✅ Model loaded successfully")
        else:
            print("⚠️  Model file not found, using untrained model")
            
        model.to(self.device)
        model.eval()
        
        return model
        
    def analyze_audio_file(self, audio_path, save_results=True):
        """
        Analizar un archivo de audio específico
        
        Args:
            audio_path: Path al archivo WAV
            save_results: Guardar resultados a disco
            
        Returns:
            Dict con análisis temporal completo
        """
        print(f"Analyzing audio: {audio_path}")
        
        # Crear dataset temporal para este audio
        dataset = TemporalHistogramDataset(
            [str(audio_path)],
            window_size=self.config['data']['window_size'],
            overlap=self.config['data']['overlap'],
            max_sequence_length=self.config['model']['max_sequence_length']
        )
        
        if len(dataset) == 0:
            print("❌ Could not process audio file")
            return None
            
        # Obtener secuencia temporal
        histogram_sequence, metadata = dataset[0]
        
        # Añadir batch dimension
        histogram_sequence = histogram_sequence.unsqueeze(0).to(self.device)
        
        # Análisis temporal completo
        with torch.no_grad():
            start_time = time.time()
            
            # Forward pass
            reconstructed, mu, logvar, attention_weights = self.model(histogram_sequence)
            
            # Temporal summary
            summary = self.model.get_temporal_summary(histogram_sequence)
            
            analysis_time = time.time() - start_time
        
        # Preparar resultados
        results = {
            'metadata': metadata,
            'analysis_time': analysis_time,
            'sequence_length': histogram_sequence.shape[1],
            'latent_representation': mu.cpu().numpy().tolist(),
            'reconstruction_quality': self._compute_reconstruction_quality(
                histogram_sequence, reconstructed
            ),
            'temporal_patterns': self._analyze_temporal_patterns(
                attention_weights, summary
            ),
            'harmonic_evolution': self._analyze_harmonic_evolution(
                histogram_sequence, attention_weights
            )
        }
        
        print(f"✅ Analysis completed in {analysis_time:.3f}s")
        print(f"   Sequence length: {results['sequence_length']} frames")
        print(f"   Reconstruction quality: {results['reconstruction_quality']:.3f}")
        print(f"   Temporal patterns found: {len(results['temporal_patterns']['strong_correlations'])}")
        
        # Guardar resultados
        if save_results:
            results_path = Path(audio_path).with_suffix('.temporal_analysis.json')
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
                
            print(f"📄 Results saved to {results_path}")
            
            # Crear visualización
            self._create_visualization(results, Path(audio_path).stem)
        
        return results
        
    def _compute_reconstruction_quality(self, original, reconstructed):
        """Calcular calidad de reconstrucción"""
        original_mean = original.mean(dim=1)  # Promedio temporal
        mse = torch.nn.functional.mse_loss(reconstructed, original_mean)
        
        # Convertir a score 0-1 (1 = perfect)
        return float(torch.exp(-mse))
        
    def _analyze_temporal_patterns(self, attention_weights, summary):
        """Analizar patrones temporales en attention"""
        # Promedio across heads y batch
        avg_attention = attention_weights.mean(dim=1).squeeze(0).cpu().numpy()
        
        influence_scores = summary['influence_scores'].squeeze(0).cpu().numpy()
        correlations = summary['temporal_correlations'][0]
        
        # Identificar frames más influyentes
        top_influential = np.argsort(influence_scores)[-5:].tolist()
        
        # Filtrar correlaciones fuertes
        strong_correlations = [
            corr for corr in correlations if corr[2] > 0.6
        ]
        
        # Detectar patrones repetitivos
        repetitive_patterns = self._detect_repetitive_patterns(avg_attention)
        
        return {
            'attention_matrix': avg_attention.tolist(),
            'influence_scores': influence_scores.tolist(),
            'top_influential_frames': top_influential,
            'strong_correlations': strong_correlations,
            'repetitive_patterns': repetitive_patterns
        }
        
    def _detect_repetitive_patterns(self, attention_matrix):
        """Detectar patrones repetitivos en attention matrix"""
        patterns = []
        seq_len = attention_matrix.shape[0]
        
        # Buscar patrones de periodicidad
        for period in range(2, min(seq_len // 3, 20)):
            correlation = 0
            comparisons = 0
            
            for i in range(seq_len - period):
                if i + period < seq_len:
                    corr = np.corrcoef(
                        attention_matrix[i, :], 
                        attention_matrix[i + period, :]
                    )[0, 1]
                    
                    if not np.isnan(corr):
                        correlation += corr
                        comparisons += 1
            
            if comparisons > 0:
                avg_correlation = correlation / comparisons
                
                if avg_correlation > 0.5:  # Threshold para patrón significativo
                    patterns.append({
                        'period': period,
                        'strength': avg_correlation,
                        'type': 'periodic'
                    })
        
        return patterns
        
    def _analyze_harmonic_evolution(self, histogram_sequence, attention_weights):
        """Analizar evolución de contenido harmónico"""
        seq = histogram_sequence.squeeze(0).cpu().numpy()  # (seq_len, 512, 3)
        
        evolution = {
            'energy_evolution': [],
            'spectral_centroid': [],
            'harmonic_density': []
        }
        
        for t in range(seq.shape[0]):
            frame = seq[t]
            
            # Energía total por frame
            total_energy = np.sum(frame[:, 1])  # Canal de energía
            evolution['energy_evolution'].append(float(total_energy))
            
            # Centroide espectral
            freqs = np.arange(512)
            if total_energy > 0:
                centroid = np.sum(freqs * frame[:, 0]) / np.sum(frame[:, 0])
            else:
                centroid = 256  # Centro por defecto
            evolution['spectral_centroid'].append(float(centroid))
            
            # Densidad harmónica (picos en el histograma)
            peaks = np.sum(frame[:, 0] > np.mean(frame[:, 0]) + 2 * np.std(frame[:, 0]))
            evolution['harmonic_density'].append(int(peaks))
        
        return evolution
        
    def _create_visualization(self, results, audio_name):
        """Crear visualización de los resultados"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Temporal Analysis: {audio_name}', fontsize=16)
        
        # 1. Attention matrix
        ax = axes[0, 0]
        attention_matrix = np.array(results['temporal_patterns']['attention_matrix'])
        
        im = ax.imshow(attention_matrix, cmap='viridis', aspect='auto')
        ax.set_title('Attention Patterns')
        ax.set_xlabel('Time (frames)')
        ax.set_ylabel('Time (frames)')
        plt.colorbar(im, ax=ax)
        
        # 2. Influence scores
        ax = axes[0, 1]
        influence = results['temporal_patterns']['influence_scores']
        ax.plot(influence, 'b-', linewidth=2)
        ax.set_title('Frame Influence Scores')
        ax.set_xlabel('Time (frames)')
        ax.set_ylabel('Influence')
        ax.grid(True, alpha=0.3)
        
        # 3. Harmonic evolution
        ax = axes[1, 0]
        evolution = results['harmonic_evolution']
        
        ax.plot(evolution['energy_evolution'], 'r-', label='Energy', linewidth=2)
        ax2 = ax.twinx()
        ax2.plot(evolution['spectral_centroid'], 'g-', label='Centroid', linewidth=2)
        
        ax.set_title('Harmonic Evolution')
        ax.set_xlabel('Time (frames)')
        ax.set_ylabel('Energy', color='r')
        ax2.set_ylabel('Spectral Centroid', color='g')
        ax.grid(True, alpha=0.3)
        
        # 4. Strong correlations
        ax = axes[1, 1]
        correlations = results['temporal_patterns']['strong_correlations']
        
        if correlations:
            x_coords = [corr[0] for corr in correlations]
            y_coords = [corr[1] for corr in correlations]
            strengths = [corr[2] for corr in correlations]
            
            scatter = ax.scatter(x_coords, y_coords, c=strengths, 
                               cmap='plasma', s=100, alpha=0.7)
            ax.set_title('Strong Temporal Correlations')
            ax.set_xlabel('Frame 1')
            ax.set_ylabel('Frame 2')
            plt.colorbar(scatter, ax=ax, label='Correlation Strength')
        else:
            ax.text(0.5, 0.5, 'No strong correlations found', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Strong Temporal Correlations')
        
        plt.tight_layout()
        
        # Guardar visualización
        viz_path = f"{audio_name}_temporal_analysis.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization saved to {viz_path}")

def main():
    """Main script para análisis temporal"""
    parser = argparse.ArgumentParser(description='Temporal Audio Analysis with VAE')
    parser.add_argument('audio_file', help='Path to WAV file to analyze')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--device', default='auto', help='Device (auto, cuda, cpu)')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save results')
    
    args = parser.parse_args()
    
    # Verificar archivo de audio
    if not Path(args.audio_file).exists():
        print(f"❌ Audio file not found: {args.audio_file}")
        return
        
    # Crear analizador
    try:
        analyzer = TemporalAnalyzer(
            model_path=args.model,
            config_path=args.config,
            device=args.device
        )
    except Exception as e:
        print(f"❌ Failed to load analyzer: {e}")
        return
    
    # Analizar archivo
    try:
        results = analyzer.analyze_audio_file(
            args.audio_file,
            save_results=not args.no_save
        )
        
        if results:
            print("\n🎯 Analysis Summary:")
            print(f"   Duration: {results['metadata'].get('original_duration', 'Unknown')}s")
            print(f"   Frames analyzed: {results['sequence_length']}")
            print(f"   Reconstruction quality: {results['reconstruction_quality']:.3f}")
            print(f"   Processing time: {results['analysis_time']:.3f}s")
            
            # Mostrar patrones encontrados
            patterns = results['temporal_patterns']
            print(f"   Strong correlations: {len(patterns['strong_correlations'])}")
            print(f"   Repetitive patterns: {len(patterns['repetitive_patterns'])}")
            
            if patterns['strong_correlations']:
                print("\n   Top correlations:")
                for i, (f1, f2, strength) in enumerate(patterns['strong_correlations'][:3]):
                    print(f"     Frame {f1} ↔ Frame {f2}: {strength:.3f}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()