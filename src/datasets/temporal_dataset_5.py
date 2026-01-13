#!/usr/bin/env python3
"""
TemporalDataset5 - Dataset para Analizador Phideus 5.0
======================================================

Soporta formatos:
- NPZ (binario, recomendado): 12x más compacto, carga rápida
- JSON (legible): Para debugging o inspección manual

Estrategias de procesamiento:
- 'sequence': Devuelve secuencia completa (pad/truncate a max_frames)
- 'average': Promedia frames temporalmente (compatible con 4.1)
- 'frames': Cada frame como muestra independiente

Uso:
    # Para HRM (aprovecha temporalidad)
    dataset = TemporalDataset5(path, strategy='sequence', max_frames=100)

    # Para VAE (compatible con formato 4.1)
    dataset = TemporalDataset5(path, strategy='average')
"""

from __future__ import annotations

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Literal, Tuple, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


class TemporalDataset5(Dataset):
    """
    Dataset para formato temporal del Analizador 5.0.

    Soporta:
    - Formato NPZ (binario): Recomendado, 12x más compacto
    - Formato JSON: Para compatibilidad

    Estructura NPZ:
    - filenames: array de nombres
    - metadata: dict con info por archivo
    - frames_<idx>: array [T, B, 3] por archivo
    - frame_times_<idx>: array [T] por archivo

    Estructura JSON:
    {
        "archivo.wav": {
            "n_frames": int,
            "n_ratio_bins": int,
            "ratio_hist_enriched_frames": [[[p,m,e], ...], ...]
        }
    }
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        strategy: Literal['sequence', 'average', 'frames'] = 'sequence',
        max_frames: Optional[int] = 100,
        n_bins: Optional[int] = None,
        normalize: bool = True,
        dtype: torch.dtype = torch.float32
    ):
        """
        Args:
            data_path: Ruta al archivo .npz o .json
            strategy:
                - 'sequence': Secuencia completa [max_frames, B, 3]
                - 'average': Promedio temporal [B, 3] (compatible con 4.1)
                - 'frames': Cada frame como muestra [B, 3]
            max_frames: Máximo de frames para 'sequence' (pad/truncate)
            n_bins: Número de bins esperado (validación)
            normalize: Normalizar datos al rango [0, 1]
            dtype: Tipo de datos de PyTorch
        """
        self.data_path = Path(data_path)
        self.strategy = strategy
        self.max_frames = max_frames
        self.n_bins = n_bins
        self.normalize = normalize
        self.dtype = dtype

        # Detectar formato y cargar
        if self.data_path.suffix == '.npz':
            self._load_npz()
        elif self.data_path.suffix == '.json':
            self._load_json()
        else:
            raise ValueError(f"Formato no soportado: {self.data_path.suffix}. Use .npz o .json")

        # Preparar índices según estrategia
        self._prepare_indices()

        # Estadísticas
        self._compute_stats()

    def _load_npz(self):
        """Carga dataset desde formato binario NPZ."""
        logger.info(f"Cargando dataset NPZ: {self.data_path}")

        self._npz_data = np.load(self.data_path, allow_pickle=True)
        self.filenames = list(self._npz_data['filenames'])
        self.metadata = self._npz_data['metadata'].item()
        self._format = 'npz'

        logger.info(f"Archivos cargados: {len(self.filenames)}")

    def _load_json(self):
        """Carga dataset desde formato JSON."""
        logger.info(f"Cargando dataset JSON: {self.data_path}")

        with open(self.data_path, 'r') as f:
            self._json_data = json.load(f)

        self.filenames = list(self._json_data.keys())
        self.metadata = {
            k: {
                'idx': i,
                'n_frames': v['n_frames'],
                'n_ratio_bins': v['n_ratio_bins'],
                'sr': v.get('sr'),
                'n_fft': v.get('n_fft'),
                'hop_length': v.get('hop_length'),
            }
            for i, (k, v) in enumerate(self._json_data.items())
        }
        self._format = 'json'

        logger.info(f"Archivos cargados: {len(self.filenames)}")

    def _get_frames(self, idx: int) -> np.ndarray:
        """Obtiene los frames de un archivo por índice."""
        if self._format == 'npz':
            return self._npz_data[f'frames_{idx}']
        else:
            filename = self.filenames[idx]
            return np.array(
                self._json_data[filename]['ratio_hist_enriched_frames'],
                dtype=np.float32
            )

    def _prepare_indices(self):
        """Prepara índices según la estrategia seleccionada."""
        if self.strategy == 'frames':
            # Cada frame es una muestra independiente
            self.indices = []
            for file_idx, filename in enumerate(self.filenames):
                n_frames = self.metadata[filename]['n_frames']
                for frame_idx in range(n_frames):
                    self.indices.append((file_idx, frame_idx))
            logger.info(f"Estrategia 'frames': {len(self.indices)} muestras totales")
        else:
            # Cada archivo es una muestra
            self.indices = [(i, None) for i in range(len(self.filenames))]
            logger.info(f"Estrategia '{self.strategy}': {len(self.indices)} muestras")

    def _compute_stats(self):
        """Calcula estadísticas del dataset."""
        frame_counts = [self.metadata[f]['n_frames'] for f in self.filenames]
        first_file = self.filenames[0]

        self.stats = {
            'n_files': len(self.filenames),
            'total_frames': sum(frame_counts),
            'avg_frames': np.mean(frame_counts),
            'min_frames': min(frame_counts),
            'max_frames_in_data': max(frame_counts),
            'n_bins': self.metadata[first_file]['n_ratio_bins']
        }

        if self.n_bins is None:
            self.n_bins = self.stats['n_bins']

        logger.info(
            f"Stats: {self.stats['n_files']} archivos, "
            f"{self.stats['total_frames']:,} frames totales, "
            f"avg={self.stats['avg_frames']:.1f} frames/archivo, "
            f"{self.n_bins} bins"
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_idx, frame_idx = self.indices[idx]

        # Obtener frames
        frames = self._get_frames(file_idx)

        if self.strategy == 'frames':
            # Devolver un solo frame: [B, 3]
            data = frames[frame_idx]

        elif self.strategy == 'average':
            # Promediar temporalmente: [B, 3]
            data = np.mean(frames, axis=0)

        elif self.strategy == 'sequence':
            # Secuencia completa con padding/truncate: [max_frames, B, 3]
            T, B, C = frames.shape

            if T >= self.max_frames:
                # Truncate (tomar los primeros max_frames)
                data = frames[:self.max_frames]
            else:
                # Pad con ceros
                data = np.zeros((self.max_frames, B, C), dtype=np.float32)
                data[:T] = frames
        else:
            raise ValueError(f"Estrategia no soportada: {self.strategy}")

        # Normalización
        if self.normalize:
            data = np.clip(data, 0, 1)

        return torch.tensor(data, dtype=self.dtype)

    def get_filename(self, idx: int) -> str:
        """Obtiene el nombre de archivo para un índice."""
        file_idx, _ = self.indices[idx]
        return self.filenames[file_idx]

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        """Obtiene metadatos de una muestra."""
        file_idx, frame_idx = self.indices[idx]
        filename = self.filenames[file_idx]
        meta = self.metadata[filename].copy()
        meta['filename'] = filename
        meta['frame_idx'] = frame_idx
        return meta

    def get_output_shape(self) -> Tuple[int, ...]:
        """Retorna la forma de salida de cada muestra."""
        if self.strategy in ('frames', 'average'):
            return (self.n_bins, 3)
        elif self.strategy == 'sequence':
            return (self.max_frames, self.n_bins, 3)
        else:
            raise ValueError(f"Estrategia no soportada: {self.strategy}")


def load_dataset_as_static(
    data_path: Union[str, Path],
    n_bins: Optional[int] = None
) -> Tuple[np.ndarray, list]:
    """
    Carga dataset y lo convierte a formato estático (promedio).
    Compatible con los scripts de entrenamiento del 4.1.

    Args:
        data_path: Ruta a .npz o .json
        n_bins: Número de bins esperado (opcional)

    Returns:
        data: np.array de shape (N, n_bins, 3)
        filenames: Lista de nombres de archivo
    """
    ds = TemporalDataset5(data_path, strategy='average', n_bins=n_bins)

    data = []
    for i in range(len(ds)):
        data.append(ds[i].numpy())

    return np.array(data, dtype=np.float32), ds.filenames


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Uso: python temporal_dataset_5.py <data_path.npz|.json>")
        sys.exit(1)

    data_path = sys.argv[1]

    print("\n" + "="*60)
    print("TEST TemporalDataset5")
    print("="*60)

    # Test estrategia 'sequence'
    print("\n1. Estrategia 'sequence' (para HRM):")
    ds_seq = TemporalDataset5(data_path, strategy='sequence', max_frames=50)
    print(f"   Muestras: {len(ds_seq)}")
    print(f"   Shape muestra: {ds_seq[0].shape}")
    print(f"   Output shape: {ds_seq.get_output_shape()}")

    # Test estrategia 'average'
    print("\n2. Estrategia 'average' (para VAE estático):")
    ds_avg = TemporalDataset5(data_path, strategy='average')
    print(f"   Muestras: {len(ds_avg)}")
    print(f"   Shape muestra: {ds_avg[0].shape}")

    # Test estrategia 'frames'
    print("\n3. Estrategia 'frames' (máximos datos):")
    ds_frames = TemporalDataset5(data_path, strategy='frames')
    print(f"   Muestras: {len(ds_frames)}")
    print(f"   Shape muestra: {ds_frames[0].shape}")

    # Estadísticas
    print("\n" + "="*60)
    print("ESTADÍSTICAS")
    print("="*60)
    for k, v in ds_seq.stats.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.2f}")
        else:
            print(f"   {k}: {v}")

    print("\n✔ Todas las pruebas pasaron")
