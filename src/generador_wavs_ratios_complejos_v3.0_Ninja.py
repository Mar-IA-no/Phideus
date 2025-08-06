#!/usr/bin/env python3
"""
generar_wavs_ratios_complejos_v3.0.py
Generador robusto de WAVs sintéticos con ratios armónicos precisos.
- Envolventes suaves (fade-in/out)
- Amplitudes iguales por componente
- Duración y muestreo controlados
- Pink noise aproximado para casos coloreados
- Subarmónicos, microintervalos, irracionales, tripletes, subhertzianas
"""
import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path

# Parámetros globales
duration = 2.0       # segundos por archivo
sr = 44100           # frecuencia de muestreo
fade_ms = 10         # duración de fade-in y fade-out en milisegundos
out_dir = Path("./test_wavs")
out_dir.mkdir(parents=True, exist_ok=True)

# Envolvente de fade-in/fade-out
def apply_fade(signal, sr, fade_ms):
    n_fade = int(sr * fade_ms / 1000)
    envelope = np.ones_like(signal)
    # fade-in
    ramp = np.linspace(0, 1, n_fade)
    envelope[:n_fade] = ramp
    # fade-out
    envelope[-n_fade:] = ramp[::-1]
    return signal * envelope

# Genera onda sinusoidal pura
def sine(freq, duration, sr):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    return np.sin(2*np.pi*freq*t)

# Genera ruido rosa aproximado (Voss-McCartney)
def pink_noise(n_samples, n_rows=16, seed=0):
    rng = np.random.RandomState(seed)
    result = np.zeros(n_samples)
    for i in range(n_rows):
        step = 2**i
        shape = int(np.ceil(n_samples/step))
        vals = rng.randn(shape)
        result += np.repeat(vals, step)[:n_samples]
    return result / n_rows

# Lista de casos: nombre -> generador de señal
cases = []
# 1. Serie armónica extendida
for num, den in [(9,8),(8,7),(7,6),(6,5),(5,4),(11,8),(13,8)]:
    label = f"{num}_{den}.wav"
    ratio = num/den
    cases.append((label, lambda r=ratio: [1.0, r]))
# 2. Subarmónicos y batidos (inversos en frecuencia)
for num, den in [(9,8),(8,7),(7,6)]:
    inv = den/num
    label = f"sub_{num}_{den}.wav"
    cases.append((label, lambda r=inv: [1.0, r]))
# 3. Microintervalos "comma"
for num, den in [(81,80),(531441,524288),(33,32)]:
    ratio = num/den
    label = f"comma_{num}_{den}.wav"
    cases.append((label, lambda r=ratio: [1.0, r]))
# 4. Means irracionales
phi = (1 + np.sqrt(5))/2
sqrt2, sqrt3, sqrt5 = np.sqrt(2), np.sqrt(3), np.sqrt(5)
sigma = 1 + sqrt2
bronze = (3 + np.sqrt(13)) / 2  # aproximado bronze mean
irr_list = [(sqrt2, "sqrt2"), (sqrt3, "sqrt3"), (sqrt5, "sqrt5"), (phi**2, "phi2"), (sigma, "sigma"), (bronze, "bronze")]
for val, name in irr_list:
    cases.append((f"irr_{name}.wav", lambda r=val: [1.0, r]))
# 5. Intervalos de séptima
for num, den in [(7,4),(9,5)]:
    cases.append((f"{num}_{den}.wav", lambda r=num/den: [1.0, r]))
# 6. Pares casi duplicados
for num, den in [(10,9),(16,15),(17,16)]:
    cases.append((f"{num}_{den}.wav", lambda r=num/den: [1.0, r]))
# 7. Octava fraccionada
for num, den in [(3,1),(5,2),(7,3)]:
    cases.append((f"{num}_{den}.wav", lambda r=num/den: [1.0, r]))
# 8. Triplete 1:phi:phi^2
cases.append(("phi_triplet.wav", lambda: [1.0, phi, phi**2]))
# 9. Ruido coloreado + tono phi
cases.append(("phi_noise.wav", lambda: [1.0, phi, "pink"]))
# 10. Subhertzianas (1 Hz + 2 Hz)
cases.append(("sub_1_2.wav", lambda: [1.0, 2.0]))

# Generación de cada archivo
def mix_and_write(label, components):
    # crear base de senos
    sig = np.zeros(int(sr*duration))
    amps = []
    # si incluye ruido
    is_noise = any(c == "pink" for c in components)
    for c in components:
        if c == "pink":
            noise = pink_noise(len(sig), seed=42)
            sig += noise
        else:
            tone = sine(c, duration, sr)
            sig += tone
            amps.append(np.max(np.abs(tone)))
    # normalizar amplitud de tonos puros
    if amps:
        max_tone = max(amps)
        sig = sig / max_tone
    # aplicar envolventes
    sig = apply_fade(sig, sr, fade_ms)
    # normalizar global
    sig = sig / np.max(np.abs(sig))
    # guardar
    wav.write(out_dir/label, sr, (sig*32767).astype(np.int16))

if __name__ == '__main__':
    # frecuencia base
    base_freq = 220.0
    for fname, gen in cases:
        comps = gen()
        # transformar razones a frecuencias absolutas
        freqs = []
        for c in comps:
            if isinstance(c, (int, float)):
                freqs.append(base_freq * c)
            else:
                freqs.append(c)
        mix_and_write(fname, freqs)
    print(f"Generados {len(cases)} WAVs en {out_dir}")

