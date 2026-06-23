# `src/voz_expresiva/` — módulo de descriptores para Voz Expresiva Phideus

Implementación de la Fase 0A del frente `Documents/01_FRENTES_ACTIVOS/Voz_Expresiva_Phideus/`.

## Qué hace

Cada utterance pasa por cuatro familias de descriptores:

| Familia | Origen | Granularidad | Dim raw | Dim post-pooling |
|---|---|---|---|---|
| **A — Phideus-ratio** | `src/bias_control/vocal_descriptors.py` (reuso) | frame-level | 12 | 48 (4-stat pool) |
| **B — Voice quality** | `voice_quality.py` (parselmouth + librosa) | utterance-level | 9 | 9 (sin re-pool) |
| **C — Control no-ratio** | `src/bias_control/vocal_descriptors.compute_a4_16k` (reuso) | frame-level | 8 | 32 (4-stat pool) |
| **D — eGeMAPSv02** | `voice_quality.compute_egemaps_functionals` (openSMILE) | utterance-level | 88 | 88 (sin re-pool) |

**Vector compuesto Phideus+VQ+Control = 89d** (A 48 + B 9 + C 32). Familia D se reporta aparte (88d).

Para más contexto sobre por qué los conteos canónicos (12 + 9 + 8 = 29d) y los post-pooling (48 + 9 + 32 = 89d) coexisten, ver el plan de Fase 0A.

## Familia B — qué es directo y qué es proxy

| Medida | Tipo | Fuente |
|---|---|---|
| HNR | directa | `Sound.to_harmonicity_cc → Get mean` |
| CPP | directa | `to_power_cepstrogram → Get CPPS` |
| jitter_local | directa | `PointProcess → Get jitter (local)` |
| shimmer_local | directa | `[Sound, PointProcess] → Get shimmer (local)` |
| F2_F1, F3_F1 | directa | `Sound.to_formant_burg`, medias sobre frames voiced |
| alpha_ratio | directa | `Spectrum → Get band energy (1k-5k) - (50-1k)` |
| H1_H2_proxy | **proxy** | STFT magnitude at F0 vs 2·F0, sin corrección de formantes |
| H1_A3_proxy | **proxy** | STFT magnitude at F0 vs F3-band peak |

Los proxies NO son medidas clínicas completas — se declaran como tales en outputs y plots.

## Política de pooling

Sólo se aplica pooling 4-estadísticos (mean, std, max, min) a descriptores frame-level (Familias A y C). Familias B y D ya son utterance-level y se usan sin re-poolear.

## Política de normalización

Esta fase NO normaliza dentro del extractor. La normalización (z-score por hablante intra-corpus, transductiva) se aplica al cargar el NPZ en el script de análisis.

## Uso típico

```python
from src.voz_expresiva import ESDLoader, compute_all_descriptors

loader = ESDLoader("/path/to/ESD", language="EN")
print(loader.summary())  # {n_utterances: ~17500, ...}

for utt in loader.iter_utterances():
    vec = compute_all_descriptors(utt.wav_path)
    # vec.family_A (48,), vec.family_B (9,), vec.family_C (32,), vec.family_D (88,)
    # vec.compound (89,) = A + B + C
```

## Dependencias

- `praat-parselmouth >= 0.4.3` (Praat wrapper)
- `opensmile >= 2.5.0` (eGeMAPSv02)
- `librosa >= 0.10.0` (PYIN F0, STFT)
- `torch`, `numpy`
- `src.bias_control.vocal_descriptors` (mismo repo)

Listadas en `requirements.txt` raíz del repo.
