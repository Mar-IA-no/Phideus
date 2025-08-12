#!/usr/bin/env python3
"""
test_enriched_validation.py

Script de validación para histogramas enriquecidos del analizador_4.1_Enriched.py
Verifica formato, normalización, balance entre canales y detección de ratios conocidos.
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np


class EnrichedHistogramValidator:
    """Validador de histogramas enriquecidos con 3 canales."""
    
    def __init__(self, json_path: Path, max_ratio: float = 6.0, n_bins: int = 256):
        self.json_path = json_path
        self.max_ratio = max_ratio
        self.n_bins = n_bins
        self.log_max_ratio = math.log2(max_ratio)
        
        # Cargar datos
        with open(json_path, 'r') as f:
            self.data = json.load(f)
    
    def validate_shape_and_format(self) -> Dict[str, bool]:
        """Test 1: Verificar shape y formato correcto."""
        results = {}
        
        for wav_name, info in self.data.items():
            try:
                enriched = np.array(info['ratio_hist_enriched'])
                
                # Shape debe ser (n_bins, 3)
                shape_ok = enriched.shape == (self.n_bins, 3)
                
                # Verificar que es numérico y finito
                numeric_ok = np.isfinite(enriched).all()
                
                results[wav_name] = {
                    'shape_ok': shape_ok,
                    'numeric_ok': numeric_ok,
                    'actual_shape': enriched.shape
                }
            except Exception as e:
                results[wav_name] = {
                    'shape_ok': False,
                    'numeric_ok': False,
                    'error': str(e)
                }
        
        return results
    
    def validate_normalization(self) -> Dict[str, Dict]:
        """Test 2: Verificar normalización de cada canal."""
        results = {}
        
        for wav_name, info in self.data.items():
            try:
                enriched = np.array(info['ratio_hist_enriched'])
                channel_sums = enriched.sum(axis=0)
                
                # Cada canal debe sumar ~1.0 (±5% tolerance)
                norm_ok = [(0.95 < s < 1.05) for s in channel_sums]
                
                results[wav_name] = {
                    'channel_sums': channel_sums.tolist(),
                    'normalized_ok': norm_ok,
                    'all_normalized': all(norm_ok)
                }
            except Exception as e:
                results[wav_name] = {'error': str(e)}
        
        return results
    
    def validate_non_negative(self) -> Dict[str, bool]:
        """Test 3: Verificar valores no negativos."""
        results = {}
        
        for wav_name, info in self.data.items():
            try:
                enriched = np.array(info['ratio_hist_enriched'])
                non_neg_ok = (enriched >= 0).all()
                min_val = enriched.min()
                
                results[wav_name] = {
                    'non_negative_ok': non_neg_ok,
                    'min_value': float(min_val)
                }
            except Exception as e:
                results[wav_name] = {'error': str(e)}
        
        return results
    
    def validate_channel_balance(self, max_ratio_threshold: float = 10.0) -> Dict[str, Dict]:
        """Test 4: Verificar balance entre canales (ninguno domina extremamente)."""
        results = {}
        
        for wav_name, info in self.data.items():
            try:
                enriched = np.array(info['ratio_hist_enriched'])
                channel_means = enriched.mean(axis=0)
                
                # Ratio entre canal máximo y mínimo
                ratio = channel_means.max() / (channel_means.min() + 1e-12)
                balanced_ok = ratio < max_ratio_threshold
                
                results[wav_name] = {
                    'channel_means': channel_means.tolist(),
                    'max_min_ratio': float(ratio),
                    'balanced_ok': balanced_ok
                }
            except Exception as e:
                results[wav_name] = {'error': str(e)}
        
        return results
    
    def validate_musical_ratios(self) -> Dict[str, Dict]:
        """Test 5: Verificar detección de ratios musicales conocidos (MÁXIMA SENSIBILIDAD)."""
        # Ratios musicales esperados y sus posiciones en bins
        musical_ratios = {
            'octave': 2.0,      # log2(2) = 1.0
            'fifth': 1.5,       # log2(1.5) ≈ 0.585  
            'fourth': 4/3,      # log2(4/3) ≈ 0.415
            'major_third': 5/4, # log2(5/4) ≈ 0.322
            'minor_third': 6/5  # log2(6/5) ≈ 0.263
        }
        
        results = {}
        
        for wav_name, info in self.data.items():
            try:
                enriched = np.array(info['ratio_hist_enriched'])
                
                # Los 3 canales para validación multicanal
                proportion_channel = enriched[:, 0]
                energy_channel = enriched[:, 1]
                entropy_channel = enriched[:, 2]
                
                detected_ratios = {}
                
                for ratio_name, ratio_value in musical_ratios.items():
                    if ratio_value <= self.max_ratio:
                        log_ratio = math.log2(ratio_value)
                        expected_bin = int((log_ratio / self.log_max_ratio) * self.n_bins)
                        
                        # Ventana variable según tipo de ratio (OPTIMIZADA PARA 512 BINS)
                        if self.n_bins >= 512:
                            # Umbrales optimizados para alta resolución (512+ bins)
                            if ratio_value < 1.3:  # commas, microintervalos
                                window_size = 7  # Mayor ventana para capturar dispersión
                                sensitivity = 0.4  # Más conservador debido a dispersión
                            elif ratio_value < 2.0:  # terceras, quintas, cuartas
                                window_size = 5
                                sensitivity = 0.5  # Balance entre sensibilidad y precisión
                            else:  # octavas y mayores
                                window_size = 4
                                sensitivity = 0.6  # Ligeramente más sensible para picos claros
                        else:
                            # Umbrales originales para baja resolución (256 bins)
                            if ratio_value < 1.3:  # commas, microintervalos
                                window_size = 5
                                sensitivity = 0.3  # muy sensible
                            elif ratio_value < 2.0:  # terceras, quintas, cuartas
                                window_size = 4
                                sensitivity = 0.5  # sensible
                            else:  # octavas y mayores
                                window_size = 3
                                sensitivity = 0.7  # menos sensible (picos más claros)
                        
                        window = slice(max(0, expected_bin - window_size), 
                                     min(self.n_bins, expected_bin + window_size + 1))
                        
                        # Análisis multicanal con umbrales muy permisivos
                        prop_window = proportion_channel[window]
                        energy_window = energy_channel[window]
                        entropy_window = entropy_channel[window]
                        
                        # Umbrales adaptativos muy sensibles
                        prop_baseline = proportion_channel.mean()
                        prop_noise = proportion_channel.std()
                        prop_threshold = prop_baseline + sensitivity * prop_noise
                        
                        energy_baseline = energy_channel.mean()
                        energy_noise = energy_channel.std()
                        energy_threshold = energy_baseline + sensitivity * energy_noise
                        
                        entropy_baseline = entropy_channel.mean()
                        entropy_noise = entropy_channel.std()
                        entropy_threshold = entropy_baseline + sensitivity * entropy_noise
                        
                        # Detectar picos en cada canal
                        prop_peak = prop_window.max() > prop_threshold
                        energy_peak = energy_window.max() > energy_threshold
                        entropy_peak = entropy_window.max() > entropy_threshold
                        
                        # Detección híbrida optimizada para 512 bins
                        if self.n_bins >= 512:
                            # Para alta resolución: Requerir al menos 1 canal FUERTE o 2 canales débiles
                            strong_prop = prop_window.max() > prop_baseline + (sensitivity + 0.2) * prop_noise
                            strong_energy = energy_window.max() > energy_baseline + (sensitivity + 0.2) * energy_noise  
                            strong_entropy = entropy_window.max() > entropy_baseline + (sensitivity + 0.2) * entropy_noise
                            
                            strong_peaks = sum([strong_prop, strong_energy, strong_entropy])
                            weak_peaks = sum([prop_peak, energy_peak, entropy_peak])
                            
                            # Detectado si: 1 pico fuerte OR 2+ picos débiles
                            peak_detected = (strong_peaks >= 1) or (weak_peaks >= 2)
                        else:
                            # Para baja resolución: MÁXIMA SENSIBILIDAD (cualquier canal)
                            peak_detected = prop_peak or energy_peak or entropy_peak
                        
                        # Encontrar el bin con máximo pico en cualquier canal
                        all_peaks = np.maximum(np.maximum(prop_window, energy_window), entropy_window)
                        peak_bin = expected_bin - window_size + np.argmax(all_peaks)
                        peak_value = float(all_peaks.max())
                        
                        # Información adicional para debug
                        channel_peaks = {
                            'proportion': prop_peak,
                            'energy': energy_peak,
                            'entropy': entropy_peak,
                            'window_size': window_size,
                            'sensitivity': sensitivity
                        }
                        
                        detected_ratios[ratio_name] = {
                            'expected_bin': expected_bin,
                            'peak_detected': peak_detected,
                            'peak_bin': int(peak_bin),
                            'peak_value': peak_value,
                            'ratio_value': ratio_value,
                            'channel_details': channel_peaks
                        }
                
                results[wav_name] = detected_ratios
                
            except Exception as e:
                results[wav_name] = {'error': str(e)}
        
        return results
    
    def generate_visual_report(self, output_dir: Path = Path("validation_plots")):
        """Generar plots de validación visual."""
        output_dir.mkdir(exist_ok=True)
        
        for wav_name, info in self.data.items():
            try:
                enriched = np.array(info['ratio_hist_enriched'])
                
                # Plot de los 3 canales
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                channel_names = ['Proporción', 'Energía', 'Entropía']
                
                # Eje X en valores de ratio (no bins)
                bin_centers = np.linspace(0, self.log_max_ratio, self.n_bins)
                ratio_values = 2 ** bin_centers  # convertir de log2 a ratio lineal
                
                for ch, (ax, name) in enumerate(zip(axes, channel_names)):
                    ax.plot(ratio_values, enriched[:, ch], linewidth=1.5)
                    ax.set_title(f'{name} - {wav_name}')
                    ax.set_xlabel('Ratio de frecuencia')
                    ax.set_ylabel('Peso normalizado')
                    ax.grid(True, alpha=0.3)
                    ax.set_xscale('log')
                
                plt.tight_layout()
                
                # Sanitizar nombre de archivo
                safe_name = wav_name.replace('/', '_').replace('.wav', '')
                plt.savefig(output_dir / f'{safe_name}_channels.png', dpi=150, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"Error generando plot para {wav_name}: {e}")
    
    def run_all_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Ejecutar todos los tests de validación."""
        print("🔍 Iniciando validación de histogramas enriquecidos...")
        print(f"📁 Datos: {self.json_path}")
        print(f"📊 Archivos a procesar: {len(self.data)}")
        print("-" * 60)
        
        results = {
            'shape_format': self.validate_shape_and_format(),
            'normalization': self.validate_normalization(),
            'non_negative': self.validate_non_negative(),
            'channel_balance': self.validate_channel_balance(),
            'musical_ratios': self.validate_musical_ratios()
        }
        
        if verbose:
            self._print_results_summary(results)
        
        # Generar plots
        print("\n📈 Generando plots de validación...")
        self.generate_visual_report()
        
        return results
    
    def _print_results_summary(self, results: Dict[str, Any]):
        """Imprimir resumen de resultados."""
        print("\n📋 RESUMEN DE VALIDACIÓN")
        print("=" * 60)
        
        # Test 1: Shape y formato
        shape_results = results['shape_format']
        shape_ok = sum(1 for r in shape_results.values() if r.get('shape_ok', False))
        print(f"✅ Test 1 - Shape/Formato: {shape_ok}/{len(shape_results)} archivos OK")
        
        # Test 2: Normalización  
        norm_results = results['normalization']
        norm_ok = sum(1 for r in norm_results.values() if r.get('all_normalized', False))
        print(f"✅ Test 2 - Normalización: {norm_ok}/{len(norm_results)} archivos OK")
        
        # Test 3: No negativos
        nonneg_results = results['non_negative']
        nonneg_ok = sum(1 for r in nonneg_results.values() if r.get('non_negative_ok', False))
        print(f"✅ Test 3 - No negativos: {nonneg_ok}/{len(nonneg_results)} archivos OK")
        
        # Test 4: Balance de canales
        balance_results = results['channel_balance']
        balance_ok = sum(1 for r in balance_results.values() if r.get('balanced_ok', False))
        print(f"✅ Test 4 - Balance canales: {balance_ok}/{len(balance_results)} archivos OK")
        
        # Test 5: Ratios musicales (con detalles de sensibilidad)
        musical_results = results['musical_ratios']
        print(f"\n🎵 Test 5 - Detección ratios musicales (MÁXIMA SENSIBILIDAD):")
        
        total_detections = 0
        total_possible = 0
        
        for wav_name, ratios_info in musical_results.items():
            if 'error' not in ratios_info:
                detected = sum(1 for r in ratios_info.values() if r.get('peak_detected', False))
                total = len(ratios_info)
                total_detections += detected
                total_possible += total
                
                # Mostrar detalles por archivo
                print(f"   {wav_name}: {detected}/{total} ratios detectados", end="")
                if detected > 0:
                    detected_names = [name for name, r in ratios_info.items() if r.get('peak_detected', False)]
                    print(f" ({', '.join(detected_names)})")
                else:
                    print()
        
        detection_rate = (total_detections / total_possible * 100) if total_possible > 0 else 0
        print(f"\n📊 RESUMEN SENSIBILIDAD: {total_detections}/{total_possible} ratios detectados ({detection_rate:.1f}%)")


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validar histogramas enriquecidos")
    parser.add_argument("json_path", type=Path, help="Ruta al JSON de histogramas enriquecidos")
    parser.add_argument("--max-ratio", type=float, default=6.0, help="Ratio máximo del análisis")
    parser.add_argument("--bins", type=int, default=256, help="Número de bins del histograma")
    parser.add_argument("--quiet", action="store_true", help="Modo silencioso")
    
    args = parser.parse_args()
    
    if not args.json_path.exists():
        print(f"❌ Error: No se encuentra el archivo {args.json_path}")
        sys.exit(1)
    
    validator = EnrichedHistogramValidator(
        json_path=args.json_path,
        max_ratio=args.max_ratio,
        n_bins=args.bins
    )
    
    results = validator.run_all_tests(verbose=not args.quiet)
    
    print(f"\n✅ Validación completada. Plots guardados en 'validation_plots/'")


if __name__ == "__main__":
    main()