#!/usr/bin/env python3
# generar_wavs_con_ratios_balanceados.py
#
# Genera WAVs con dos senoidales superpuestas en relaciones precisas,
# balanceando sus amplitudes para mejorar la detección espectral.

import numpy as np
import os
from scipy.io.wavfile import write

# ==== CONFIGURACIÓN GENERAL ====

OUTPUT_DIR = "wavs_sinteticos"
DURATION = 2.0        # duración del audio en segundos
SR = 44100            # sample rate
AMPLITUDE = 0.9       # amplitud total (valor máximo 1.0)
FUNDAMENTAL = 220.0   # frecuencia base en Hz

# Ratios a generar: etiqueta → valor
RATIOS = {
    "quinta_justa_3_2": 1.5,
    "tercera_mayor_5_4": 1.25,
    "phi": 1.618,
    "raiz_2": 1.414,
    "octava": 2.0,
    "tritono_45_32": 1.40625,
    "cuarta_justa_4_3": 1.333,
}

# ==== FUNCIONES ====

def generar_dos_tonos_balanceados(f1, f2, duration, sr, amplitude):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Cada tono ocupa la mitad de la amplitud total
    onda1 = (amplitude / 2) * np.sin(2 * np.pi * f1 * t)
    onda2 = (amplitude / 2) * np.sin(2 * np.pi * f2 * t)
    return onda1 + onda2

def guardar_wav(nombre, señal, sr):
    señal = np.clip(señal, -1.0, 1.0)
    señal_int16 = np.int16(señal * 32767)
    write(nombre, sr, señal_int16)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for nombre, ratio in RATIOS.items():
        f1 = FUNDAMENTAL
        f2 = f1 * ratio
        señal = generar_dos_tonos_balanceados(f1, f2, DURATION, SR, AMPLITUDE)
        path = os.path.join(OUTPUT_DIR, f"{nombre}.wav")
        guardar_wav(path, señal, SR)
        print(f"✔ Generado: {path} — f1 = {f1:.2f} Hz, f2 = {f2:.2f} Hz, ratio = {ratio}")

if __name__ == "__main__":
    main()
