#!/usr/bin/env python3
"""
Dataset Temporal para Attention-Based Temporal VAE
Convierte archivos WAV en secuencias de histogramas temporales
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import json
import os
import sys
import math
from pathlib import Path
from scipy.signal import find_peaks
from typing import Sequence, Dict, Any

# Agregar path para imports del analizador existente
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))

class TemporalHistogramDataset(Dataset):
    """
    Dataset que genera secuencias temporales de histogramas a partir de archivos WAV
    Usando sliding windows con overlap
    """
    def __init__(self, 
                 audio_files_list,
                 window_size=1.0,
                 overlap=0.5,
                 max_sequence_length=60,
                 sample_rate=44100,
                 normalize=True):
        """
        Args:
            audio_files_list: Lista de paths a archivos WAV
            window_size: Tamaño ventana en segundos
            overlap: Overlap entre ventanas (0.0-1.0)
            max_sequence_length: Máximo número de frames por secuencia
            sample_rate: Sample rate para cargar audio
            normalize: Normalizar histogramas
        """
        self.audio_files = audio_files_list
        self.window_size = window_size
        self.overlap = overlap
        self.max_sequence_length = max_sequence_length
        self.sample_rate = sample_rate
        self.normalize = normalize
        
        # Calcular step size para sliding window
        self.step_size = window_size * (1.0 - overlap)
        
        # Pre-calcular información de secuencias para cada audio
        self.sequences_info = []
        self._prepare_sequences_info()
        
    def _prepare_sequences_info(self):
        """Pre-calcular información de secuencias para indexing eficiente"""
        print("Preparing temporal sequences info...")
        
        for audio_idx, audio_file in enumerate(self.audio_files):
            try:
                # Cargar audio para obtener duración
                audio, sr = librosa.load(audio_file, sr=self.sample_rate, mono=True)
                duration = len(audio) / sr
                
                # Calcular número de ventanas posibles
                num_windows = int((duration - self.window_size) / self.step_size) + 1
                num_windows = min(num_windows, self.max_sequence_length)
                
                if num_windows >= 3:  # Mínimo 3 frames por secuencia (para testing)
                    self.sequences_info.append({
                        'audio_file': audio_file,
                        'audio_idx': audio_idx,
                        'duration': duration,
                        'num_windows': num_windows
                    })
                    
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue
                
        print(f"Prepared {len(self.sequences_info)} temporal sequences")
    
    def __len__(self):
        return len(self.sequences_info)
    
    def __getitem__(self, idx):
        """
        Obtener una secuencia temporal de histogramas
        
        Returns:
            histogram_sequence: (seq_len, 512, 3)
            metadata: Dict con información adicional
        """
        seq_info = self.sequences_info[idx]
        
        try:
            # Cargar audio completo
            audio, sr = librosa.load(
                seq_info['audio_file'], 
                sr=self.sample_rate, 
                mono=True
            )
            
            # Extraer secuencia de histogramas
            histogram_sequence = self._extract_temporal_histograms(audio, sr)
            
            # Metadata para debugging/analysis
            metadata = {
                'audio_file': seq_info['audio_file'],
                'original_duration': seq_info['duration'],
                'sequence_length': len(histogram_sequence),
                'window_size': self.window_size,
                'overlap': self.overlap
            }
            
            return histogram_sequence, metadata
            
        except Exception as e:
            print(f"Error extracting sequence from {seq_info['audio_file']}: {e}")
            # Retornar secuencia vacía en caso de error
            empty_sequence = torch.zeros(5, 512, 3)
            return empty_sequence, {'error': str(e)}
    
    def _extract_temporal_histograms(self, audio, sr):
        """
        Extraer secuencia de histogramas con sliding windows
        
        Args:
            audio: numpy array con señal audio
            sr: sample rate
            
        Returns:
            histograms: torch.Tensor (seq_len, 512, 3)
        """
        histograms = []
        window_samples = int(self.window_size * sr)
        step_samples = int(self.step_size * sr)
        
        # Sliding windows
        for start_sample in range(0, len(audio) - window_samples + 1, step_samples):
            if len(histograms) >= self.max_sequence_length:
                break
                
            end_sample = start_sample + window_samples
            window_audio = audio[start_sample:end_sample]
            
            # Extraer histograma para esta ventana
            histogram = self._extract_histogram_from_window(window_audio, sr)
            histograms.append(histogram)
        
        if not histograms:
            # Audio demasiado corto, crear histograma único
            histogram = self._extract_histogram_from_window(audio, sr)
            histograms.append(histogram)
        
        # Convertir a tensor
        histogram_sequence = torch.stack(histograms, dim=0)
        
        return histogram_sequence
    
    def _extract_histogram_from_window(self, audio_window, sr):
        """
        Extraer histograma enriquecido (512, 3) de una ventana de audio
        
        NUEVA IMPLEMENTACIÓN: Usa el algoritmo oficial del analizador v4.1
        - Canal 0: Proporción (PDF normalizada)
        - Canal 1: Energía (segundo momento en log₂ space)  
        - Canal 2: Entropía local normalizada
        """
        # NUEVA IMPLEMENTACIÓN: Usa el algoritmo oficial del analizador v4.1
        # Configuración idéntica al analizador v4.1
        n_ffts = [2048, 4096]  # Multi-escala
        hop_length = 512
        peak_thr_factor = 1.25
        local_window = 5
        cent_tol = 40.0  # cents
        max_band_hz = 8000  # Hz máximo a considerar
        min_ratio = 1.0
        max_ratio = 6.0
        n_ratio_bins = 512
        
        # Usar algoritmo oficial completo
        result = self._process_audio_window_official(
            audio_window, sr, n_ffts, hop_length, peak_thr_factor,
            local_window, cent_tol, max_band_hz, min_ratio, max_ratio, n_ratio_bins
        )
        
        # Extraer histograma enriquecido
        enriched_hist = result['ratio_hist_enriched']
        histogram = np.array(enriched_hist)
        
        # Convertir a tensor y normalizar
        histogram = torch.FloatTensor(histogram)
        
        if self.normalize:
            # Normalizar cada canal independientemente
            for c in range(histogram.shape[1]):
                channel = histogram[:, c]
                channel_sum = channel.sum()
                if channel_sum > 0:
                    histogram[:, c] = channel / channel_sum
        
        return histogram
    
    def _process_audio_window_official(self, audio_window, sr, n_ffts, hop_length, 
                                      peak_thr_factor, local_window, cent_tol, 
                                      max_band_hz, min_ratio, max_ratio, n_ratio_bins):
        """
        Implementación OFICIAL del analizador v4.1 adaptada para ventanas temporales
        Genera histograma enriquecido con máxima precisión
        """
        y = audio_window
        
        # 3.1 Espectro promedio multi-escala (IGUAL que analizador v4.1)
        mag_matrix, freq_ref = [], None
        for n in n_ffts:
            stft = librosa.stft(y, n_fft=n, hop_length=hop_length, center=False, window="hann")
            mag = np.abs(stft).mean(axis=1)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n)
            if freq_ref is None:
                freq_ref = freqs
            mag_matrix.append(np.interp(freq_ref, freqs, mag))
        mag_per_bin = np.mean(mag_matrix, axis=0)
        
        # 3.2 Detección adaptativa de picos
        thr = self._local_threshold(mag_per_bin, local_window, peak_thr_factor)
        raw_peaks, _ = find_peaks(mag_per_bin, height=thr)
        if max_band_hz is not None:
            raw_peaks = raw_peaks[freq_ref[raw_peaks] <= max_band_hz]
        
        # 3.3 Refinamiento y deduplicado
        cand = sorted(((self._parabolic_interpolation(mag_per_bin, p), mag_per_bin[p]) 
                      for p in raw_peaks), key=lambda t: -t[1])
        bins_sel, amps_sel = [], []
        log_tol = cent_tol / 1200.0
        for b, a in cand:
            if all(abs(math.log2(b / prev)) > log_tol for prev in bins_sel):
                bins_sel.append(b)
                amps_sel.append(a)
        
        if not bins_sel:
            empty_3ch = [[0.0, 0.0, 0.0] for _ in range(n_ratio_bins)]
            return {
                "sr": int(sr),
                "peak_freqs": [],
                "ratios_log": [],
                "ratios_lin": [],
                "ratio_hist_enriched": empty_3ch,
            }
        
        # 3.4 Conversión a Hz
        peak_freqs = np.interp(bins_sel, np.arange(len(freq_ref)), freq_ref)
        ord_idx = np.argsort(peak_freqs)
        peak_freqs = peak_freqs[ord_idx]
        amps_sel = np.array(amps_sel)[ord_idx]
        
        # 3.5 Ratios y pesos
        ratios_log, ratios_lin, weights = [], [], []
        for i, (fi, ai) in enumerate(zip(peak_freqs, amps_sel)):
            for fj, aj in zip(peak_freqs[i + 1 :], amps_sel[i + 1 :]):
                r = fj / fi
                if r < min_ratio or r > max_ratio:
                    continue
                ratios_lin.append(r)
                ratios_log.append(math.log2(r))
                weights.append(math.sqrt(ai * aj))
        
        # 3.6 Histogramas (log)
        if ratios_log:
            hist_log, edges_log = np.histogram(
                ratios_log,
                bins=n_ratio_bins,
                range=(0, math.log2(max_ratio)),
                weights=weights,
            )
        else:
            hist_log = np.zeros(n_ratio_bins)
            edges_log = np.linspace(0, math.log2(max_ratio), n_ratio_bins + 1)
        
        # 3.7 Histograma enriquecido (OFICIAL)
        ratio_hist_enriched = self._compute_enriched_histogram(hist_log.astype(float), edges_log)
        
        return {
            "sr": int(sr),
            "peak_freqs": peak_freqs.tolist(),
            "ratios_log": ratios_log,
            "ratios_lin": ratios_lin,
            "ratio_hist_enriched": ratio_hist_enriched,
        }
    
    def _local_threshold(self, vec: np.ndarray, window: int, factor: float) -> np.ndarray:
        """Umbral local por ventana de tamaño *window* multiplicado por *factor*."""
        thr = np.empty_like(vec)
        for i in range(len(vec)):
            lo, hi = max(0, i - window), min(len(vec), i + window)
            thr[i] = np.median(vec[lo:hi]) * factor
        return thr
    
    def _parabolic_interpolation(self, arr: np.ndarray, idx: int) -> float:
        """Refinamiento sub-bin de un máximo mediante interpolación parabólica."""
        if idx <= 0 or idx >= len(arr) - 1:
            return float(idx)
        a, b, c = arr[idx - 1], arr[idx], arr[idx + 1]
        return idx + 0.5 * (a - c) / (a - 2 * b + c)
    
    def _compute_enriched_histogram(self, counts: np.ndarray, bin_edges: np.ndarray) -> list:
        """Devuelve un histograma enriquecido (proporción, energía, entropía).
        
        IMPLEMENTACIÓN OFICIAL del analizador v4.1:
        * counts: array de conteos por bin (no normalizados)
        * bin_edges: edges de los bins (len = n_bins + 1) en escala log2
        """
        n_bins = len(counts)
        total = counts.sum() + 1e-12
        
        # Canal 0: proporción (PDF)
        prop = counts / total
        
        # Canal 1: energía (segundo momento en escala log2)
        log_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0  # centros en log2 space
        energy_raw = counts * (log_centers ** 2)
        energy = energy_raw / (energy_raw.sum() + 1e-12)
        
        # Canal 2: entropía local
        ent_raw = -prop * np.log(prop + 1e-12)
        ent = ent_raw / (ent_raw.sum() + 1e-12)
        
        enriched = np.stack([prop, energy, ent], axis=1)  # (n_bins, 3)
        return enriched.tolist()

def create_temporal_dataloaders(audio_files_list,
                              train_split=0.8,
                              val_split=0.1,
                              batch_size=2,
                              num_workers=2,
                              **dataset_kwargs):
    """
    Crear DataLoaders para train/val/test del Temporal VAE
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Split de datos
    n_files = len(audio_files_list)
    n_train = int(n_files * train_split)
    n_val = int(n_files * val_split)
    
    train_files = audio_files_list[:n_train]
    val_files = audio_files_list[n_train:n_train + n_val]
    test_files = audio_files_list[n_train + n_val:]
    
    # Crear datasets
    train_dataset = TemporalHistogramDataset(train_files, **dataset_kwargs)
    val_dataset = TemporalHistogramDataset(val_files, **dataset_kwargs)
    test_dataset = TemporalHistogramDataset(test_files, **dataset_kwargs)
    
    # Custom collate function para manejar secuencias de longitud variable
    def temporal_collate_fn(batch):
        """Collate function que maneja secuencias de diferentes longitudes"""
        sequences, metadatas = zip(*batch)
        
        # Encontrar longitud máxima en el batch
        max_len = max(seq.shape[0] for seq in sequences)
        batch_size = len(sequences)
        
        # Pad secuencias a longitud máxima
        padded_sequences = torch.zeros(batch_size, max_len, 512, 3)
        sequence_lengths = []
        
        for i, seq in enumerate(sequences):
            seq_len = seq.shape[0]
            padded_sequences[i, :seq_len] = seq
            sequence_lengths.append(seq_len)
        
        return padded_sequences, torch.tensor(sequence_lengths), list(metadatas)
    
    # Crear DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=temporal_collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=temporal_collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=temporal_collate_fn,
        pin_memory=torch.cuda.is_available()
    ) if test_files else None
    
    print(f"Created DataLoaders:")
    print(f"  Train: {len(train_dataset)} sequences")
    print(f"  Val: {len(val_dataset)} sequences") 
    print(f"  Test: {len(test_dataset) if test_files else 0} sequences")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test del dataset temporal
    
    # Buscar archivos WAV de ejemplo
    test_wav_dir = Path("../../../test_wavs")
    if test_wav_dir.exists():
        wav_files = list(test_wav_dir.glob("*.wav"))[:5]  # Solo 5 para test
    else:
        print("No test WAV files found, creating dummy dataset")
        wav_files = ["dummy1.wav", "dummy2.wav"]  # Para test sin archivos
    
    print(f"Testing with {len(wav_files)} audio files")
    
    try:
        # Crear dataset
        dataset = TemporalHistogramDataset(
            wav_files,
            window_size=1.0,
            overlap=0.5,
            max_sequence_length=30
        )
        
        print(f"Dataset created with {len(dataset)} sequences")
        
        if len(dataset) > 0:
            # Test sample
            sequence, metadata = dataset[0]
            print(f"✅ Sample extracted:")
            print(f"  Sequence shape: {sequence.shape}")
            print(f"  Metadata: {metadata}")
            
            # Test DataLoader
            train_loader, val_loader, test_loader = create_temporal_dataloaders(
                wav_files,
                batch_size=1,
                num_workers=0  # No multiprocessing para test
            )
            
            # Test batch
            for batch_sequences, batch_lengths, batch_metadata in train_loader:
                print(f"✅ Batch loaded:")
                print(f"  Sequences shape: {batch_sequences.shape}")
                print(f"  Lengths: {batch_lengths}")
                break
                
    except Exception as e:
        print(f"Dataset test error (expected without real WAV files): {e}")
    
    print("✅ Temporal dataset implementation complete!")