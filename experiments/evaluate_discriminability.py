#!/usr/bin/env python3
"""
Evaluación de Discriminabilidad de Histogramas v2.2

Evalúa si los histogramas de ratios son suficientemente discriminativos
para permitir aprendizaje cross-modal (pre-entrenamiento).

Fase 1 del Roadmap - Implementación completa.

Uso:
    python experiments/evaluate_discriminability.py --data data/datasets/roseta_full.npz

Métricas calculadas:
    1. Entropía normalizada (H / H_max)
       - Target: < 85%

    2. Gap aligned-shuffled
       - Usando encoder mínimo
       - Target: > 5%

    3. Similitud con media global
       - Correlación descriptor-media
       - Target: < 0.85

    4. Similitud inter-condición
       - Matriz de similitud entre condiciones de falla
       - Target: < 0.92

Criterio GO/NO-GO:
    - Si 2/3 métricas pasan → GO para entrenamiento
    - Si fallan → Iterar extractor o ir a Grupo 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_roseta_dataset(npz_path: Path) -> Dict:
    """
    Carga dataset Roseta desde NPZ.

    Returns:
        Dict con:
        - filenames: list of filenames
        - conditions: list of conditions
        - audio_frames: list of [T, B, 3] arrays
        - vibration_frames: list of [T, B, 3] arrays
    """
    data = np.load(npz_path, allow_pickle=True)

    n_files = int(data['n_files'])
    filenames = list(data['filenames'])
    conditions = list(data['conditions'])

    audio_frames = []
    vibration_frames = []

    for idx in range(n_files):
        audio_frames.append(data[f'audio_frames_{idx}'])
        vibration_frames.append(data[f'vibration_frames_{idx}'])

    return {
        'filenames': filenames,
        'conditions': conditions,
        'audio_frames': audio_frames,
        'vibration_frames': vibration_frames,
        'n_files': n_files,
    }


def calculate_entropy(histogram: np.ndarray) -> float:
    """
    Calcula entropía normalizada de un histograma.

    Args:
        histogram: [T, B, 3] o [B, 3] o [B]

    Returns:
        Entropía normalizada (0-1)
    """
    # Extraer canal de proporción
    if histogram.ndim == 3:
        hist = histogram[:, :, 0].mean(axis=0)
    elif histogram.ndim == 2:
        hist = histogram[:, 0]
    else:
        hist = histogram

    # Normalizar
    total = hist.sum()
    if total <= 0:
        return 1.0  # Máxima incertidumbre

    p = hist / total
    p = p[p > 0]

    # Entropía
    entropy = -np.sum(p * np.log(p))
    max_entropy = np.log(len(hist))

    return float(entropy / max_entropy) if max_entropy > 0 else 1.0


def calculate_global_mean_similarity(
    audio_frames: List[np.ndarray],
    vibration_frames: List[np.ndarray],
) -> Tuple[float, float]:
    """
    Calcula similitud promedio con la media global.

    Returns:
        (audio_sim, vib_sim): Correlación promedio con media global
    """
    # Calcular media global
    all_audio = np.concatenate([f[:, :, 0] for f in audio_frames], axis=0)
    all_vib = np.concatenate([f[:, :, 0] for f in vibration_frames], axis=0)

    audio_mean = all_audio.mean(axis=0)
    vib_mean = all_vib.mean(axis=0)

    # Calcular correlación con media
    audio_sims = []
    vib_sims = []

    for audio_f, vib_f in zip(audio_frames, vibration_frames):
        for frame in audio_f:
            corr = np.corrcoef(frame[:, 0], audio_mean)[0, 1]
            if not np.isnan(corr):
                audio_sims.append(corr)

        for frame in vib_f:
            corr = np.corrcoef(frame[:, 0], vib_mean)[0, 1]
            if not np.isnan(corr):
                vib_sims.append(corr)

    return float(np.mean(audio_sims)), float(np.mean(vib_sims))


def calculate_inter_condition_similarity(
    frames_by_condition: Dict[str, List[np.ndarray]],
) -> Dict[str, Dict[str, float]]:
    """
    Calcula matriz de similitud entre condiciones.

    Returns:
        Dict con similitud entre cada par de condiciones
    """
    conditions = list(frames_by_condition.keys())
    n_cond = len(conditions)

    # Calcular media por condición
    means = {}
    for cond, frames_list in frames_by_condition.items():
        all_frames = np.concatenate([f[:, :, 0] for f in frames_list], axis=0)
        means[cond] = all_frames.mean(axis=0)

    # Calcular similitud entre condiciones
    similarity_matrix = {}
    for i, c1 in enumerate(conditions):
        similarity_matrix[c1] = {}
        for j, c2 in enumerate(conditions):
            if i == j:
                similarity_matrix[c1][c2] = 1.0
            else:
                corr = np.corrcoef(means[c1], means[c2])[0, 1]
                similarity_matrix[c1][c2] = float(corr) if not np.isnan(corr) else 0.0

    return similarity_matrix


def calculate_aligned_shuffled_gap(
    audio_frames: List[np.ndarray],
    vibration_frames: List[np.ndarray],
    n_samples: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Calcula gap entre pares aligned vs shuffled usando similitud coseno.

    Returns:
        (aligned_sim, shuffled_sim, gap)
    """
    np.random.seed(seed)

    # Flatten frames para muestreo
    audio_flat = []
    vib_flat = []
    file_ids = []

    for file_idx, (af, vf) in enumerate(zip(audio_frames, vibration_frames)):
        n_frames = min(len(af), len(vf))
        for frame_idx in range(n_frames):
            audio_flat.append(af[frame_idx, :, 0])
            vib_flat.append(vf[frame_idx, :, 0])
            file_ids.append(file_idx)

    audio_flat = np.array(audio_flat)
    vib_flat = np.array(vib_flat)
    file_ids = np.array(file_ids)

    n_total = len(audio_flat)
    if n_total < n_samples:
        n_samples = n_total

    # Sample indices
    indices = np.random.choice(n_total, n_samples, replace=False)

    # Aligned similarity
    aligned_sims = []
    for idx in indices:
        a = audio_flat[idx]
        v = vib_flat[idx]
        norm_a = np.linalg.norm(a)
        norm_v = np.linalg.norm(v)
        if norm_a > 0 and norm_v > 0:
            sim = np.dot(a, v) / (norm_a * norm_v)
            aligned_sims.append(sim)

    # Shuffled similarity (shuffle vib indices)
    shuffled_sims = []
    shuffled_indices = np.random.permutation(indices)
    for i, idx in enumerate(indices):
        a = audio_flat[idx]
        v = vib_flat[shuffled_indices[i]]
        norm_a = np.linalg.norm(a)
        norm_v = np.linalg.norm(v)
        if norm_a > 0 and norm_v > 0:
            sim = np.dot(a, v) / (norm_a * norm_v)
            shuffled_sims.append(sim)

    aligned_mean = float(np.mean(aligned_sims))
    shuffled_mean = float(np.mean(shuffled_sims))
    gap = aligned_mean - shuffled_mean

    return aligned_mean, shuffled_mean, gap


def minimal_encoder_gap(
    audio_frames: List[np.ndarray],
    vibration_frames: List[np.ndarray],
    embed_dim: int = 32,
    n_epochs: int = 20,
    batch_size: int = 128,  # Increased for GPU
    seed: int = 42,
    use_gpu: bool = True,
) -> Dict:
    """
    Entrena encoder mínimo y calcula gap aligned vs shuffled.

    Returns:
        Dict con métricas del encoder mínimo
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        return {'error': 'PyTorch not available', 'gap': 0.0}

    # Use GPU if available
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Preparar datos
    audio_flat = []
    vib_flat = []

    for af, vf in zip(audio_frames, vibration_frames):
        n_frames = min(len(af), len(vf))
        for frame_idx in range(n_frames):
            audio_flat.append(af[frame_idx, :, 0])
            vib_flat.append(vf[frame_idx, :, 0])

    audio_flat = np.array(audio_flat, dtype=np.float32)
    vib_flat = np.array(vib_flat, dtype=np.float32)

    n_total = len(audio_flat)
    input_dim = audio_flat.shape[1]

    # Simple MLP encoder
    class MinimalEncoder(nn.Module):
        def __init__(self, input_dim, embed_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, embed_dim),
            )

        def forward(self, x):
            return self.encoder(x)

    encoder_audio = MinimalEncoder(input_dim, embed_dim).to(device)
    encoder_vib = MinimalEncoder(input_dim, embed_dim).to(device)

    # Contrastive loss (InfoNCE simplified)
    def contrastive_loss(emb_a, emb_v, temperature=0.1):
        # Normalize
        emb_a = emb_a / (emb_a.norm(dim=1, keepdim=True) + 1e-8)
        emb_v = emb_v / (emb_v.norm(dim=1, keepdim=True) + 1e-8)

        # Similarity matrix
        logits = torch.mm(emb_a, emb_v.t()) / temperature
        labels = torch.arange(len(emb_a), device=emb_a.device)

        # Cross entropy loss
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss

    optimizer = optim.Adam(
        list(encoder_audio.parameters()) + list(encoder_vib.parameters()),
        lr=1e-3
    )

    # Training loop
    n_batches = n_total // batch_size
    for epoch in range(n_epochs):
        indices = np.random.permutation(n_total)
        total_loss = 0.0

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = start + batch_size
            batch_indices = indices[start:end]

            audio_batch = torch.FloatTensor(audio_flat[batch_indices]).to(device)
            vib_batch = torch.FloatTensor(vib_flat[batch_indices]).to(device)

            optimizer.zero_grad()
            emb_a = encoder_audio(audio_batch)
            emb_v = encoder_vib(vib_batch)
            loss = contrastive_loss(emb_a, emb_v)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    # Evaluate
    encoder_audio.eval()
    encoder_vib.eval()

    with torch.no_grad():
        audio_tensor = torch.FloatTensor(audio_flat).to(device)
        vib_tensor = torch.FloatTensor(vib_flat).to(device)

        emb_audio = encoder_audio(audio_tensor).cpu().numpy()
        emb_vib = encoder_vib(vib_tensor).cpu().numpy()

        # Normalize
        emb_audio = emb_audio / (np.linalg.norm(emb_audio, axis=1, keepdims=True) + 1e-8)
        emb_vib = emb_vib / (np.linalg.norm(emb_vib, axis=1, keepdims=True) + 1e-8)

        # Aligned similarity
        aligned_sims = np.sum(emb_audio * emb_vib, axis=1)
        aligned_mean = float(np.mean(aligned_sims))

        # Shuffled similarity
        shuffled_indices = np.random.permutation(len(emb_vib))
        shuffled_sims = np.sum(emb_audio * emb_vib[shuffled_indices], axis=1)
        shuffled_mean = float(np.mean(shuffled_sims))

        gap = aligned_mean - shuffled_mean

    return {
        'aligned_sim': aligned_mean,
        'shuffled_sim': shuffled_mean,
        'gap': gap,
        'gap_ratio': aligned_mean / (shuffled_mean + 1e-8),
        'n_samples': n_total,
        'embed_dim': embed_dim,
        'n_epochs': n_epochs,
    }


def evaluate_discriminability(npz_path: Path, verbose: bool = True) -> Dict:
    """
    Evalúa discriminabilidad completa de un dataset.

    Returns:
        Dict con todas las métricas y veredicto GO/NO-GO
    """
    if verbose:
        print(f"\n{'='*60}")
        print("EVALUACIÓN DE DISCRIMINABILIDAD v2.2")
        print(f"{'='*60}")
        print(f"Dataset: {npz_path}")

    # Cargar datos
    dataset = load_roseta_dataset(npz_path)
    n_files = dataset['n_files']

    if verbose:
        print(f"Archivos: {n_files}")

    # 1. Entropía normalizada
    if verbose:
        print("\n[1/4] Calculando entropía normalizada...")

    entropies_audio = []
    entropies_vib = []

    for af, vf in zip(dataset['audio_frames'], dataset['vibration_frames']):
        entropies_audio.append(calculate_entropy(af))
        entropies_vib.append(calculate_entropy(vf))

    mean_entropy_audio = float(np.mean(entropies_audio))
    mean_entropy_vib = float(np.mean(entropies_vib))

    if verbose:
        print(f"  Audio: {mean_entropy_audio:.3f} | Vibración: {mean_entropy_vib:.3f}")
        print(f"  Target: < 0.85 | Status: {'PASS' if max(mean_entropy_audio, mean_entropy_vib) < 0.85 else 'FAIL'}")

    # 2. Similitud con media global
    if verbose:
        print("\n[2/4] Calculando similitud con media global...")

    audio_mean_sim, vib_mean_sim = calculate_global_mean_similarity(
        dataset['audio_frames'],
        dataset['vibration_frames'],
    )

    if verbose:
        print(f"  Audio: {audio_mean_sim:.3f} | Vibración: {vib_mean_sim:.3f}")
        print(f"  Target: < 0.85 | Status: {'PASS' if max(audio_mean_sim, vib_mean_sim) < 0.85 else 'FAIL'}")

    # 3. Similitud inter-condición
    if verbose:
        print("\n[3/4] Calculando similitud inter-condición...")

    # Agrupar por condición
    frames_by_condition = {}
    for cond, af, vf in zip(
        dataset['conditions'],
        dataset['audio_frames'],
        dataset['vibration_frames'],
    ):
        if cond not in frames_by_condition:
            frames_by_condition[cond] = []
        frames_by_condition[cond].append(af)

    sim_matrix = calculate_inter_condition_similarity(frames_by_condition)

    # Calcular promedio de off-diagonal
    off_diag_sims = []
    conditions = list(sim_matrix.keys())
    for i, c1 in enumerate(conditions):
        for j, c2 in enumerate(conditions):
            if i != j:
                off_diag_sims.append(sim_matrix[c1][c2])

    mean_inter_cond = float(np.mean(off_diag_sims)) if off_diag_sims else 0.0

    if verbose:
        print(f"  Similitud promedio inter-condición: {mean_inter_cond:.3f}")
        print(f"  Target: < 0.92 | Status: {'PASS' if mean_inter_cond < 0.92 else 'FAIL'}")

    # 4. Gap aligned-shuffled
    if verbose:
        print("\n[4/4] Calculando gap aligned-shuffled...")

    # Raw cosine similarity
    aligned_raw, shuffled_raw, gap_raw = calculate_aligned_shuffled_gap(
        dataset['audio_frames'],
        dataset['vibration_frames'],
    )

    if verbose:
        print(f"  [Raw cosine] Aligned: {aligned_raw:.4f} | Shuffled: {shuffled_raw:.4f} | Gap: {gap_raw:.4f}")

    # Minimal encoder
    encoder_results = minimal_encoder_gap(
        dataset['audio_frames'],
        dataset['vibration_frames'],
    )

    if 'error' not in encoder_results:
        if verbose:
            print(f"  [Encoder] Aligned: {encoder_results['aligned_sim']:.4f} | "
                  f"Shuffled: {encoder_results['shuffled_sim']:.4f} | "
                  f"Gap: {encoder_results['gap']:.4f}")
            print(f"  Target gap: > 0.05 | Status: {'PASS' if encoder_results['gap'] > 0.05 else 'FAIL'}")
    else:
        if verbose:
            print(f"  [Encoder] {encoder_results['error']}")

    # Compilar resultados
    results = {
        'entropy': {
            'audio': mean_entropy_audio,
            'vibration': mean_entropy_vib,
            'pass': max(mean_entropy_audio, mean_entropy_vib) < 0.85,
        },
        'global_mean_similarity': {
            'audio': audio_mean_sim,
            'vibration': vib_mean_sim,
            'pass': max(audio_mean_sim, vib_mean_sim) < 0.85,
        },
        'inter_condition_similarity': {
            'mean': mean_inter_cond,
            'matrix': sim_matrix,
            'pass': mean_inter_cond < 0.92,
        },
        'aligned_shuffled_gap': {
            'raw': {
                'aligned': aligned_raw,
                'shuffled': shuffled_raw,
                'gap': gap_raw,
            },
            'encoder': encoder_results,
            'pass': encoder_results.get('gap', 0) > 0.05,
        },
        'n_files': n_files,
        'dataset_path': str(npz_path),
    }

    # GO/NO-GO decision
    passes = [
        results['entropy']['pass'],
        results['global_mean_similarity']['pass'],
        results['aligned_shuffled_gap']['pass'],
    ]
    n_pass = sum(passes)

    results['go_nogo'] = {
        'criteria_passed': n_pass,
        'criteria_total': len(passes),
        'decision': 'GO' if n_pass >= 2 else 'NO-GO',
    }

    if verbose:
        print(f"\n{'='*60}")
        print("VEREDICTO GO/NO-GO")
        print(f"{'='*60}")
        print(f"Criterios pasados: {n_pass}/{len(passes)}")
        print(f"  - Entropía < 0.85: {'PASS' if results['entropy']['pass'] else 'FAIL'}")
        print(f"  - Sim. media < 0.85: {'PASS' if results['global_mean_similarity']['pass'] else 'FAIL'}")
        print(f"  - Gap > 0.05: {'PASS' if results['aligned_shuffled_gap']['pass'] else 'FAIL'}")
        print(f"\nDecisión: {results['go_nogo']['decision']}")
        print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evalúa discriminabilidad de histogramas pre-entrenamiento"
    )
    parser.add_argument(
        "--data", "-d",
        type=Path,
        required=True,
        help="Path al dataset NPZ (roseta_*.npz)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Path para guardar resultados JSON (opcional)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Modo silencioso (sin prints)",
    )

    args = parser.parse_args()

    results = evaluate_discriminability(args.data, verbose=not args.quiet)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Resultados guardados en: {args.output}")

    # Exit code based on GO/NO-GO
    sys.exit(0 if results['go_nogo']['decision'] == 'GO' else 1)


if __name__ == "__main__":
    main()
