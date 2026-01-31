#!/usr/bin/env python3
"""
Sweep de Configuraciones del Extractor v2.2

Este script ejecuta múltiples configuraciones del extractor y evalúa
la discriminabilidad de los histogramas resultantes.

Fase 1 del Roadmap - Implementación completa.

Uso:
    python experiments/sweep_extractor.py \
        --input-dir data/datasets/UOEMD/raw/2_CSV_Data_Files \
        --output sweep_results.json

Configuraciones evaluadas:
    - top_k_peaks: [8, 12, 16]
    - min_prominence: [0.1, 0.2, 0.3]
    - temporal_stability_threshold: [0.5, 0.7]
    - use_warped_bins: [False, True]

Métricas calculadas:
    - Entropía normalizada
    - Gap aligned-shuffled (con encoder mínimo)
    - Similitud con media global
    - Cobertura de picos estables

Criterio de selección: Frontera de Pareto
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

import numpy as np

# Add src and experiments to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from evaluate_discriminability import evaluate_discriminability


def generate_sweep_configs(
    top_k_peaks: List[int] = [8, 12, 16],
    min_prominence: List[float] = [0.1, 0.2, 0.3],
    temporal_stability_threshold: List[float] = [0.5, 0.7],
    use_warped_bins: List[bool] = [False, True],
) -> List[Dict]:
    """
    Genera lista de configuraciones para sweep.

    Returns:
        Lista de dicts con configuraciones
    """
    configs = []

    for tk, mp, ts, wb in itertools.product(
        top_k_peaks,
        min_prominence,
        temporal_stability_threshold,
        use_warped_bins,
    ):
        config = {
            'top_k_peaks': tk,
            'min_prominence': mp,
            'temporal_stability_threshold': ts,
            'use_warped_bins': wb,
            # Fixed parameters
            'temporal_window_frames': 10,
            'temporal_freq_tolerance_hz': 20.0,
            'min_peaks_fallback': 3,
            'warped_gamma': 0.5,
            'peak_thr_factor': 2.5,
            'min_peak_distance_hz': 50.0,
        }
        configs.append(config)

    return configs


def _process_single_csv_worker(args):
    """Worker function for parallel CSV processing (must be at module level for pickle)."""
    csv_path, input_dir, params = args
    from analizador.analizador_roseta import process_uoemd_csv

    rel = csv_path.relative_to(input_dir).as_posix()
    try:
        result = process_uoemd_csv(csv_path, **params)
        return (rel, result)
    except Exception as e:
        return (rel, {'error': str(e)})


def run_extractor_with_config(
    input_dir: Path,
    output_npz: Path,
    config: Dict,
    max_files: int | None = None,
    workers: int = 12,  # Default to 12 workers for parallelization
) -> Dict:
    """
    Ejecuta el extractor con una configuración específica.

    Returns:
        Dict con métricas de ejecución
    """
    import multiprocessing as mp
    from analizador.analizador_roseta import (
        save_roseta_dataset,
    )

    # Find CSV files
    csv_paths = list(input_dir.rglob("*.csv"))
    if max_files:
        csv_paths = csv_paths[:max_files]

    if not csv_paths:
        return {'error': f'No CSV files found in {input_dir}'}

    # Process with config
    params = {
        'sr': 42000,
        'n_fft': 4096,
        'hop_length': 1024,
        'local_window': 30,
        'rel_peak_tol': 0.01,
        'max_band_hz': None,
        'min_ratio': 1.0,
        'max_ratio': 6.0,
        'n_ratio_bins': 256,
        # v2.2 params from config
        'peak_thr_factor': config.get('peak_thr_factor', 2.5),
        'top_k_peaks': config['top_k_peaks'],
        'min_prominence': config['min_prominence'],
        'min_peak_distance_hz': config.get('min_peak_distance_hz', 50.0),
        'temporal_window_frames': config.get('temporal_window_frames', 10),
        'temporal_stability_threshold': config['temporal_stability_threshold'],
        'temporal_freq_tolerance_hz': config.get('temporal_freq_tolerance_hz', 20.0),
        'min_peaks_fallback': config.get('min_peaks_fallback', 3),
        'use_warped_bins': config['use_warped_bins'],
        'warped_gamma': config.get('warped_gamma', 0.5),
    }

    # Prepare tasks for parallel processing
    tasks = [(csv_path, input_dir, params) for csv_path in csv_paths]

    dataset = {}

    if workers > 1 and len(csv_paths) > 1:
        # Parallel processing
        with mp.Pool(processes=min(workers, len(csv_paths))) as pool:
            results = pool.map(_process_single_csv_worker, tasks)
        for rel, result in results:
            if 'error' not in result:
                dataset[rel] = result
    else:
        # Sequential processing
        for task in tasks:
            rel, result = _process_single_csv_worker(task)
            if 'error' not in result:
                dataset[rel] = result

    if not dataset:
        return {'error': 'No files processed successfully'}

    # Save
    save_roseta_dataset(dataset, output_npz)

    return {
        'n_files': len(dataset),
        'output_path': str(output_npz),
    }


def calculate_pareto_score(result: Dict) -> Dict:
    """
    Calcula scores para selección de Pareto.

    Returns:
        Dict con scores (mayor = mejor para todos)
    """
    entropy = max(
        result.get('entropy', {}).get('audio', 1.0),
        result.get('entropy', {}).get('vibration', 1.0),
    )
    global_sim = max(
        result.get('global_mean_similarity', {}).get('audio', 1.0),
        result.get('global_mean_similarity', {}).get('vibration', 1.0),
    )
    gap = result.get('aligned_shuffled_gap', {}).get('encoder', {}).get('gap', 0.0)

    return {
        # Invertir para que mayor = mejor
        'score_entropy': 1.0 - entropy,  # Queremos baja entropía
        'score_global_sim': 1.0 - global_sim,  # Queremos baja similitud
        'score_gap': gap,  # Queremos alto gap
        # Combined score (weighted)
        'combined_score': (1.0 - entropy) * 0.3 + (1.0 - global_sim) * 0.3 + gap * 0.4,
    }


def is_pareto_optimal(scores: List[Dict], idx: int) -> bool:
    """
    Verifica si un resultado está en la frontera de Pareto.
    """
    current = scores[idx]
    for i, other in enumerate(scores):
        if i == idx:
            continue
        # Check if 'other' dominates 'current'
        if (other['score_entropy'] >= current['score_entropy'] and
            other['score_global_sim'] >= current['score_global_sim'] and
            other['score_gap'] >= current['score_gap'] and
            (other['score_entropy'] > current['score_entropy'] or
             other['score_global_sim'] > current['score_global_sim'] or
             other['score_gap'] > current['score_gap'])):
            return False
    return True


def run_sweep(
    input_dir: Path,
    output_dir: Path,
    configs: List[Dict],
    max_files: int | None = None,
    workers: int = 1,
    verbose: bool = True,
) -> Dict:
    """
    Ejecuta sweep completo de configuraciones.

    Returns:
        Dict con todos los resultados
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    n_configs = len(configs)

    for i, config in enumerate(configs):
        config_name = f"config_{i:03d}"
        npz_path = output_dir / f"{config_name}.npz"

        if verbose:
            print(f"\n{'='*60}")
            print(f"[{i+1}/{n_configs}] Configuración: {config_name}")
            print(f"{'='*60}")
            print(f"  top_k={config['top_k_peaks']}, prom={config['min_prominence']}, "
                  f"stability={config['temporal_stability_threshold']}, warped={config['use_warped_bins']}")

        # Run extractor
        if verbose:
            print(f"\n  Ejecutando extractor...")

        extract_result = run_extractor_with_config(
            input_dir=input_dir,
            output_npz=npz_path,
            config=config,
            max_files=max_files,
            workers=workers,
        )

        if 'error' in extract_result:
            if verbose:
                print(f"  ERROR: {extract_result['error']}")
            results.append({
                'config': config,
                'config_name': config_name,
                'error': extract_result['error'],
            })
            continue

        # Evaluate discriminability
        if verbose:
            print(f"  Evaluando discriminabilidad...")

        try:
            eval_result = evaluate_discriminability(npz_path, verbose=False)
        except Exception as e:
            if verbose:
                print(f"  ERROR en evaluación: {e}")
            results.append({
                'config': config,
                'config_name': config_name,
                'error': str(e),
            })
            continue

        # Calculate Pareto scores
        pareto_scores = calculate_pareto_score(eval_result)

        result = {
            'config': config,
            'config_name': config_name,
            'n_files': extract_result['n_files'],
            'npz_path': str(npz_path),
            'metrics': {
                'entropy_audio': eval_result['entropy']['audio'],
                'entropy_vib': eval_result['entropy']['vibration'],
                'global_sim_audio': eval_result['global_mean_similarity']['audio'],
                'global_sim_vib': eval_result['global_mean_similarity']['vibration'],
                'aligned_shuffled_gap': eval_result['aligned_shuffled_gap']['encoder'].get('gap', 0.0),
            },
            'pareto_scores': pareto_scores,
            'go_nogo': eval_result['go_nogo'],
        }
        results.append(result)

        if verbose:
            print(f"\n  Resultados:")
            print(f"    Entropía: audio={result['metrics']['entropy_audio']:.3f}, "
                  f"vib={result['metrics']['entropy_vib']:.3f}")
            print(f"    Sim global: audio={result['metrics']['global_sim_audio']:.3f}, "
                  f"vib={result['metrics']['global_sim_vib']:.3f}")
            print(f"    Gap aligned-shuffled: {result['metrics']['aligned_shuffled_gap']:.4f}")
            print(f"    Combined score: {pareto_scores['combined_score']:.4f}")
            print(f"    GO/NO-GO: {result['go_nogo']['decision']}")

    # Calculate Pareto optimality
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        pareto_scores_list = [r['pareto_scores'] for r in valid_results]
        for i, r in enumerate(valid_results):
            r['is_pareto_optimal'] = is_pareto_optimal(pareto_scores_list, i)

    # Sort by combined score
    valid_results.sort(key=lambda x: x['pareto_scores']['combined_score'], reverse=True)

    # Summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'n_configs_total': n_configs,
        'n_configs_successful': len(valid_results),
        'n_configs_failed': n_configs - len(valid_results),
        'top_3_configs': [r['config_name'] for r in valid_results[:3]],
        'pareto_optimal_configs': [r['config_name'] for r in valid_results if r.get('is_pareto_optimal', False)],
    }

    return {
        'summary': summary,
        'results': results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sweep de configuraciones del Extractor v2.2"
    )
    parser.add_argument(
        "--input-dir", "-i",
        type=Path,
        required=True,
        help="Carpeta con CSVs de UOEMD",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("data/sweep_v22"),
        help="Directorio para datasets generados",
    )
    parser.add_argument(
        "--results-json", "-r",
        type=Path,
        default=Path("sweep_results.json"),
        help="Archivo JSON para resultados",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Máximo de archivos a procesar (para testing rápido)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Número de workers para procesamiento paralelo (default: 12)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Modo rápido: menos configuraciones",
    )

    args = parser.parse_args()

    # Generate configs
    if args.quick:
        # Reduced sweep for quick testing
        configs = generate_sweep_configs(
            top_k_peaks=[8, 12],
            min_prominence=[0.1, 0.2],
            temporal_stability_threshold=[0.6],
            use_warped_bins=[False],
        )
    else:
        configs = generate_sweep_configs()

    print(f"\n{'='*60}")
    print("SWEEP DE CONFIGURACIONES - EXTRACTOR v2.2")
    print(f"{'='*60}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Configuraciones: {len(configs)}")
    if args.max_files:
        print(f"Max archivos: {args.max_files}")

    # Run sweep
    results = run_sweep(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        configs=configs,
        max_files=args.max_files,
        workers=args.workers,
        verbose=True,
    )

    # Save results
    with open(args.results_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("RESUMEN FINAL")
    print(f"{'='*60}")
    print(f"Configs exitosas: {results['summary']['n_configs_successful']}/{results['summary']['n_configs_total']}")
    print(f"Top 3 configuraciones:")
    for i, name in enumerate(results['summary']['top_3_configs'][:3], 1):
        r = next((x for x in results['results'] if x['config_name'] == name), None)
        if r:
            print(f"  {i}. {name}: score={r['pareto_scores']['combined_score']:.4f}")
    print(f"\nResultados guardados en: {args.results_json}")


if __name__ == "__main__":
    main()
