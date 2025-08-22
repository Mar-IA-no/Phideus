#!/usr/bin/env python3
"""
Generador Masivo de Dataset para Temporal VAE
Genera 500+ WAVs sintéticos con variaciones controladas para entrenamiento
"""
import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path
import itertools

# Configuración
sr = 44100
fade_ms = 10
output_dir = Path("./train/synthetic_dataset_500")
output_dir.mkdir(parents=True, exist_ok=True)

def apply_fade(signal, sr, fade_ms):
    """Aplicar fade-in/fade-out"""
    n_fade = int(sr * fade_ms / 1000)
    envelope = np.ones_like(signal)
    ramp = np.linspace(0, 1, n_fade)
    envelope[:n_fade] = ramp
    envelope[-n_fade:] = ramp[::-1]
    return signal * envelope

def sine_wave(freq, duration, sr):
    """Generar onda seno"""
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    return np.sin(2*np.pi*freq*t)

def pink_noise(n_samples, seed=0):
    """Ruido rosa aproximado"""
    rng = np.random.RandomState(seed)
    result = np.zeros(n_samples)
    for i in range(16):
        step = 2**i
        shape = int(np.ceil(n_samples/step))
        vals = rng.randn(shape)
        result += np.repeat(vals, step)[:n_samples]
    return result / 16

def generate_wav(filename, frequencies, duration, noise_level=0.0, seed=0):
    """Generar y guardar WAV con múltiples componentes"""
    sig = np.zeros(int(sr * duration))
    
    # Agregar componentes tonales
    for freq in frequencies:
        if freq > 20 and freq < 20000:  # Rango audible
            sig += sine_wave(freq, duration, sr)
    
    # Agregar ruido si se especifica
    if noise_level > 0:
        noise = pink_noise(len(sig), seed=seed)
        sig += noise_level * noise
    
    # Normalizar
    if len(frequencies) > 0:
        sig = sig / len(frequencies)
    
    # Aplicar fade
    sig = apply_fade(sig, sr, fade_ms)
    
    # Normalización final
    if np.max(np.abs(sig)) > 0:
        sig = sig / np.max(np.abs(sig))
    
    # Guardar
    wav.write(output_dir / filename, sr, (sig * 32767).astype(np.int16))

def main():
    """Generar dataset masivo"""
    generated_count = 0
    
    print("🎵 Generando Dataset Masivo para Temporal VAE...")
    print(f"📁 Directorio: {output_dir}")
    
    # =============================================
    # 1. RATIOS ARMÓNICOS BÁSICOS (100 variaciones)
    # =============================================
    print("1️⃣ Generando ratios armónicos básicos...")
    
    harmonic_ratios = [
        (9, 8), (8, 7), (7, 6), (6, 5), (5, 4), (11, 8), (4, 3),
        (7, 5), (3, 2), (8, 5), (5, 3), (9, 5), (7, 4), (11, 6),
        (15, 8), (2, 1), (17, 8), (9, 4), (5, 2), (12, 5), (13, 5)
    ]
    
    base_frequencies = [110, 220, 440, 660, 880]
    durations = [2.0, 3.0, 4.0, 5.0]
    
    for (num, den), base_freq, duration in itertools.product(harmonic_ratios, base_frequencies, durations):
        ratio = num / den
        frequencies = [base_freq, base_freq * ratio]
        filename = f"harmonic_{num}_{den}_f{int(base_freq)}_d{int(duration)}.wav"
        generate_wav(filename, frequencies, duration)
        generated_count += 1
        
        if generated_count % 20 == 0:
            print(f"   Generados: {generated_count}")
    
    # =============================================
    # 2. RATIOS IRRACIONALES (80 variaciones)
    # =============================================
    print("2️⃣ Generando ratios irracionales...")
    
    phi = (1 + np.sqrt(5)) / 2
    sqrt2, sqrt3, sqrt5 = np.sqrt(2), np.sqrt(3), np.sqrt(5)
    bronze = (3 + np.sqrt(13)) / 2
    
    irrational_ratios = [
        (sqrt2, "sqrt2"), (sqrt3, "sqrt3"), (sqrt5, "sqrt5"),
        (phi, "phi"), (phi**2, "phi2"), (bronze, "bronze"),
        (1/phi, "phi_inv"), (np.e/2, "euler_half"), (np.pi/2, "pi_half")
    ]
    
    for (ratio, name), base_freq, duration in itertools.product(irrational_ratios, base_frequencies, durations):
        if ratio > 0.5 and ratio < 8.0:  # Rango viable
            frequencies = [base_freq, base_freq * ratio]
            filename = f"irrational_{name}_f{int(base_freq)}_d{int(duration)}.wav"
            generate_wav(filename, frequencies, duration)
            generated_count += 1
    
    # =============================================
    # 3. MICROINTERVALOS Y COMMAS (60 variaciones)
    # =============================================
    print("3️⃣ Generando microintervalos...")
    
    micro_ratios = [
        (81, 80), (531441, 524288), (33, 32), (256, 243),
        (2187, 2048), (1024, 1000), (128, 125), (648, 625)
    ]
    
    for (num, den), base_freq, duration in itertools.product(micro_ratios, base_frequencies[:3], durations):
        ratio = num / den
        frequencies = [base_freq, base_freq * ratio]
        filename = f"micro_{num}_{den}_f{int(base_freq)}_d{int(duration)}.wav"
        generate_wav(filename, frequencies, duration)
        generated_count += 1
    
    # =============================================
    # 4. ACORDES COMPLEJOS (100 variaciones)
    # =============================================
    print("4️⃣ Generando acordes complejos...")
    
    chord_templates = [
        [1.0, 5/4, 3/2],              # Mayor
        [1.0, 6/5, 3/2],              # Menor
        [1.0, 5/4, 3/2, 9/5],         # Mayor séptima
        [1.0, 6/5, 3/2, 16/9],        # Menor séptima
        [1.0, phi, phi**2],           # Phi chord
        [1.0, sqrt2, 2.0],            # Sqrt2 chord
        [1.0, 4/3, 3/2, 2.0],         # Cuarta suspendida
        [1.0, 9/8, 5/4, 3/2],         # Add9
    ]
    
    for template, base_freq, duration in itertools.product(chord_templates, base_frequencies, durations[:2]):
        frequencies = [base_freq * r for r in template]
        chord_name = f"chord_{len(template)}notes_f{int(base_freq)}_d{int(duration)}"
        filename = f"{chord_name}.wav"
        generate_wav(filename, frequencies, duration)
        generated_count += 1
    
    # =============================================
    # 5. VARIACIONES CON RUIDO (60 variaciones)
    # =============================================
    print("5️⃣ Generando variaciones con ruido...")
    
    noise_levels = [0.1, 0.2, 0.3]
    simple_ratios = [(3, 2), (5, 4), (4, 3), (7, 4), (2, 1)]
    
    for (num, den), base_freq, noise_level, duration in itertools.product(
        simple_ratios, base_frequencies[:2], noise_levels, durations[:2]
    ):
        ratio = num / den
        frequencies = [base_freq, base_freq * ratio]
        filename = f"noisy_{num}_{den}_f{int(base_freq)}_n{int(noise_level*100)}_d{int(duration)}.wav"
        generate_wav(filename, frequencies, duration, noise_level=noise_level, seed=generated_count)
        generated_count += 1
    
    # =============================================
    # 6. SUBARMÓNICOS (40 variaciones)
    # =============================================
    print("6️⃣ Generando subarmónicos...")
    
    subharmonic_ratios = [(1, 2), (2, 3), (3, 4), (4, 5), (3, 5), (5, 7), (5, 8)]
    
    for (num, den), base_freq, duration in itertools.product(subharmonic_ratios, base_frequencies[:2], durations):
        ratio = num / den
        frequencies = [base_freq, base_freq * ratio]
        filename = f"sub_{num}_{den}_f{int(base_freq)}_d{int(duration)}.wav"
        generate_wav(filename, frequencies, duration)
        generated_count += 1
    
    # =============================================
    # 7. CASOS ESPECIALES (20 variaciones)
    # =============================================
    print("7️⃣ Generando casos especiales...")
    
    # Batidos lentos
    for beat_hz in [0.5, 1.0, 2.0, 4.0]:
        for base_freq in [220, 440]:
            frequencies = [base_freq, base_freq + beat_hz]
            filename = f"beat_{beat_hz}hz_f{int(base_freq)}.wav"
            generate_wav(filename, frequencies, 4.0)
            generated_count += 1
    
    # Tritono y intervalos disonantes
    for ratio in [17/12, 25/18, 36/25, 45/32]:  # Aproximaciones de tritono
        for base_freq in [220, 440]:
            frequencies = [base_freq, base_freq * ratio]
            filename = f"dissonant_{ratio:.3f}_f{int(base_freq)}.wav"
            generate_wav(filename, frequencies, 3.0)
            generated_count += 1
    
    print(f"\n✅ Dataset generado exitosamente!")
    print(f"📊 Total de archivos: {generated_count}")
    print(f"📁 Ubicación: {output_dir}")
    print(f"💾 Espacio aproximado: ~{generated_count * 4 * 44.1:.1f} KB")

if __name__ == "__main__":
    main()