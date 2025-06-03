# !/usr/bin/env python3
# """
# Generador de WAVs sintéticos v2.0
# ================================
#
# • Produce un *circuito ninja* completo de pruebas para el analizador de ratios.
# • Cada archivo WAV contiene 2 o 3 tonos puros (senos) que guardan la relación
#   indicada en la tabla STRESS_TEST. Opcionalmente se añade ruido rosa a SNR dB.
# • Todos los parámetros globales se editan en la sección CONFIG.
# • Usa solo NumPy + soundfile (pip install soundfile) → formatos WAV de 32‑bit float.
# • Mantiene la estructura y comentarios de generar_wavs_con_ration_v1.2.py pero
#   amplía funcionalidad y lotes. La función original generate_wav() sigue estando
#   por compatibilidad.
# """

from __future__ import annotations

import math, os, random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

# ╭──────────────────────────╮
# │ CONFIGURACIÓN GLOBAL     │
# ╰──────────────────────────╯
SR          = 48_000       # Hz
DURATION    = 5.0          # seg por archivo (>= 5 permite 1 Hz)
AMPLITUDE   = 0.3          # pico por tono antes de mezcla
OUTPUT_DIR  = Path("synthetic_wavs")  # carpeta destino (se crea auto)
SEED        = 42           # reproducibilidad de ruido

# ╭──────────────────────────╮
# │  TABLA STRESS‑TEST       │
# ╰──────────────────────────╯
# Cada entrada: (nombre_base,  [ratios], base_freq (Hz),  snr_db (None→sin ruido))
STRESS_TEST: List[Tuple[str, List[float], float, float | None]] = [
    # 1. Serie armónica extendida
    ("9_8",           [9/8],        220.0, None),
    ("8_7",           [8/7],        220.0, None),
    ("7_6",           [7/6],        220.0, None),
    ("6_5",           [6/5],        220.0, None),
    ("11_8",          [11/8],       220.0, None),
    ("13_8",          [13/8],       220.0, None),
    # 2. Subarmónicos (invertidos)
    ("sub_9_8",       [9/8],        110.0*9/8, None),  # base * ratio = 110
    ("sub_8_7",       [8/7],        110.0*8/7, None),
    ("sub_7_6",       [7/6],        110.0*7/6, None),
    # 3. Micro‑intervalos comma
    ("comma_81_80",   [81/80],      400.0, None),
    ("comma_pyth",    [531441/524288], 400.0, None),
    ("comma_33_32",   [33/32],      400.0, None),
    # 4. Irracionales
    ("irr_sqrt2",     [math.sqrt(2)], 300.0, None),
    ("irr_sqrt3",     [math.sqrt(3)], 300.0, None),
    ("irr_sqrt5",     [math.sqrt(5)], 300.0, None),
    ("phi",           [ (1+math.sqrt(5))/2 ], 300.0, None),
    ("phi2",          [ ((1+math.sqrt(5))/2)**2 ], 180.0, None),
    ("sigma",         [ 1+math.sqrt(2) ], 220.0, None),
    ("bronze",        [ 3.302775637731995 ], 150.0, None),
    # 5. Séptimas
    ("7_4",           [7/4],        220.0, None),
    ("9_5",           [9/5],        220.0, None),
    # 6. Pares casi duplicados
    ("10_9",          [10/9],       440.0, None),
    ("16_15",         [16/15],      440.0, None),
    # 7. Octava fraccionada
    ("3_1",           [3/1],        110.0, None),
    ("5_2",           [5/2],        110.0, None),
    ("7_3",           [7/3],        110.0, None),
    # 8. Triplete phi (tres tonos)
    ("phi_triplet",   [ (1+math.sqrt(5))/2, ((1+math.sqrt(5))/2)**2 ], 200.0, None),
    # 9. Ruido + phi
    ("phi_noise",     [ (1+math.sqrt(5))/2 ], 300.0, 10.0),  # SNR 10 dB
    # 10. Baja frecuencia (1 Hz vs 2 Hz) → base 1 Hz tono muy grave
    ("octave_sub",    [2.0],        1.0, None),
]

# ╭──────────────────────────╮
# │  FUNCIONES               │
# ╰──────────────────────────╯

def pink_noise(n: int) -> np.ndarray:
    """Genera ruido rosa rápido (aprox) vía filtrado 1/f en frecuencia."""
    # FFT‑based (not perfect but good enough for test)
    uneven = n % 2
    X = (np.random.normal(size=n//2 + 1 + uneven) + 1j*np.random.normal(size=n//2 + 1 + uneven))
    S = np.sqrt(np.arange(len(X)) + 1.)  # 1/sqrt(f) amplitude
    y = (np.fft.irfft(X / S)).real
    if uneven:
        y = y[:-1]
    y -= y.mean()
    y /= np.max(np.abs(y) + 1e-12)
    return y


def mix_with_snr(signal: np.ndarray, snr_db: float) -> np.ndarray:
    noise = pink_noise(len(signal))
    sig_power = np.mean(signal**2)
    noise_power = sig_power / (10**(snr_db/10))
    noise *= math.sqrt(noise_power)
    return signal + noise


def generate_wav(out_path: Path, ratios: List[float], base_freq: float, sr: int = SR, duration: float = DURATION, snr_db: float | 
None = None):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    waves = [AMPLITUDE * np.sin(2*math.pi*base_freq*t)]
    for r in ratios:
        waves.append(AMPLITUDE * np.sin(2*math.pi*base_freq*r*t))
    signal = np.sum(waves, axis=0)
    # Normalización suave
    signal /= max(1.0, np.max(np.abs(signal)))
    if snr_db is not None:
        signal = mix_with_snr(signal, snr_db)
        signal /= max(1.0, np.max(np.abs(signal)))
    sf.write(out_path, signal.astype(np.float32), sr)

# ╭──────────────────────────╮
# │  MAIN LOTE               │
# ╰──────────────────────────╯

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generando {len(STRESS_TEST)} WAVs en {OUTPUT_DIR.resolve()}/ …")
    for name, ratios, base, snr in STRESS_TEST:
        fname = f"{name}.wav"
        out_path = OUTPUT_DIR / fname
        generate_wav(out_path, ratios, base)
        if snr is not None:
            # regenerar con mismo nombre ya que función maneja SNR
            generate_wav(out_path, ratios, base, snr_db=snr)
        print(f"  ✔ {fname}  (base={base:.3f} Hz  ratios={ratios}  SNR={snr})")
    print("Listo. Revisa la carpeta y prueba el analizador.")


if __name__ == "__main__":
    main()

