#!/usr/bin/env python3
# plot_ratio_histograms.py
#
# Crea un gráfico por cada entrada de ratios_dataset.json, visualizando el vector ratio_hist
# Guarda cada uno como archivo PNG nombrado según el WAV original

import os
import json
import matplotlib.pyplot as plt

# ============ CONFIGURACIÓN ============

INPUT_JSON = "ratios_dataset.json"  # Ruta al JSON generado
OUTPUT_DIR = "histogramas"          # Carpeta de salida
MAX_RATIO = 5.0                     # Máximo ratio representado
N_BINS = 100                        # Debe coincidir con el script original

# ============ FUNCIONES ============

def plot_histogram(name, hist, output_path):
    x_vals = [i * (MAX_RATIO / N_BINS) for i in range(N_BINS)]
    plt.figure(figsize=(10, 4))
    plt.bar(x_vals, hist, width=MAX_RATIO / N_BINS, color='teal', edgecolor='black')
    plt.title(f"Histograma de ratios – {name}")
    plt.xlabel("Ratio (log₂ escala aprox.)")
    plt.ylabel("Peso relativo")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# ============ EJECUCIÓN ============

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    for wav_name, info in data.items():
        hist = info.get("ratio_hist", [])
        if not hist:
            continue
        base = os.path.splitext(wav_name)[0]
        output_path = os.path.join(OUTPUT_DIR, f"{base}.png")
        plot_histogram(wav_name, hist, output_path)

    print(f"{len(data)} histogramas generados en '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()

