# `experiments/voz_expresiva/` — pipeline experimental del frente

Scripts experimentales del frente `Voz Expresiva Phideus`.

Estado del árbol al corte `2026-06-24`:

- `Fase 0A` cerrada: extracción + análisis descriptor-only sobre `ESD` English.
- `Fase 0B` cerrada: clasificación clásica `LOSO` con lectura dual `N-strict / N-adapt`.
- `Fase 1` cerrada sobre `ESD` English: `WavLM` frozen + inyección `concat / FiLM / xattn`.
- Próximo corte esperado: réplica de `Fase 1` sobre el subset chino de `ESD`.

Documentos guía:

- `Documents/01_FRENTES_ACTIVOS/Voz_Expresiva_Phideus/README.md`
- `Documents/01_FRENTES_ACTIVOS/Voz_Expresiva_Phideus/ROADMAP_VOZ_EXPRESIVA_PHIDEUS.md`

## Orden de ejecución

### 1) Adquirir ESD (manual)

ESD se distribuye via Google Drive tras un formulario de registro:

```bash
python experiments/voz_expresiva/download_esd.py
# imprime instrucciones manuales

# Tras descargar y extraer:
python experiments/voz_expresiva/download_esd.py \
    --verify data/esd/raw/'Emotional Speech Dataset'
```

### 2) Instalar dependencias del frente

Las nuevas deps están listadas en `requirements.txt`:

```bash
pip install -r requirements.txt
```

Si `praat-parselmouth` u `opensmile` fallan al compilar, ver la sección de troubleshooting al final.

### 3) Extracción de descriptores

```bash
python experiments/voz_expresiva/0A_extract.py \
    --esd-root data/esd/raw/'Emotional Speech Dataset' \
    --output data/esd/descriptors_0A_en.npz \
    --language EN --workers 14
```

Para un debug rápido:

```bash
python experiments/voz_expresiva/0A_extract.py \
    --esd-root data/esd/raw/'Emotional Speech Dataset' \
    --output /tmp/descriptors_smoke.npz \
    --language EN --workers 4 --limit 100
```

### 4) Análisis + reporte 0A

```bash
python experiments/voz_expresiva/0A_analyze.py \
    --input data/esd/descriptors_0A_en.npz \
    --output-dir data/visualizations/voz_expresiva/0A
```

Genera plots PCA/UMAP por familia + compuesto, boxplots, JSONs de métricas (eta², MI, KW, silhouette, varianza intra/inter speaker) y `REPORTE_0A.md`.

### 5) Clasificación clásica 0B

```bash
python experiments/voz_expresiva/0B_classify.py \
    --input data/esd/descriptors_0A_en.npz \
    --output-dir data/voz_expresiva/0B

python experiments/voz_expresiva/0B_report.py \
    --input-dir data/voz_expresiva/0B
```

Genera `uar_results.json`, `diff_bootstrap.json`, matrices de confusión y `REPORTE_0B.md`.

### 6) Pre-caches Fase 1

```bash
python experiments/voz_expresiva/1_precache_wavlm.py
python experiments/voz_expresiva/1_precache_descriptors.py
```

Genera:

- `data/voz_expresiva/wavlm_cache/`
- `data/voz_expresiva/descriptors_cache/`

### 7) Training + reporte Fase 1

```bash
python experiments/voz_expresiva/1_train.py
python experiments/voz_expresiva/1_report.py
```

Genera `data/voz_expresiva/1/REPORTE_1.md` y los artefactos comparativos de `UAR`, `CKA`, predicciones y trazabilidad de calibración.

## Política operativa histórica por bloque

- **0A / 0B**: CPU-only.
- **Fase 1**: GPU recomendada para precache de `WavLM` y training loop.
- **Workers = 14** para extracción/precache CPU.
- **Outputs NO van a git**: `/data/` está globalmente excluido en el `.gitignore` raíz.

## Decisiones congeladas ya materializadas

- Vector compuesto = Familias A + B + C = 89d post-pool (raw canónico 29d).
- Pooling 4-stat (mean, std, max, min) sólo sobre frame-level (A y C). Familias B y D no se re-poolean.
- Normalización: z-score por hablante intra-corpus, declarada **transductiva** explícitamente.
- Scope: ESD English (10 speakers). Mandarin queda para más tarde.
- Splits: no aplica (no training); plan mode Fase 0B los congela.
- Familia B distingue **medidas directas** (7) de **proxies acústicos** (2). Los proxies se reportan como tales.
- Familia D (eGeMAPSv02) se reporta aparte, NO se concatena al compuesto.
- En `Fase 1`, los mecanismos `concat / FiLM / xattn` operan todos **frame-level post-WavLM, pre-pool**.
- `Fase 1` usa `WavLM` frozen + multi-seed + `CKA` post-pool pre-head.

## Scripts incluidos

- `0A_extract.py`, `0A_analyze.py`
- `0B_classify.py`, `0B_report.py`
- `1_precache_wavlm.py`, `1_precache_descriptors.py`
- `1_train.py`, `1_report.py`
- `download_esd.py`
- `SPIKE_FASE_1_0.md`

## Troubleshooting

### `praat-parselmouth` falla en compilación

- Linux: instalar `libpraat-dev` o usar wheel pre-construido (`pip install praat-parselmouth --only-binary :all:`).
- Si falla en venv local, usar Python 3.10/3.11 (3.12 puede no tener wheels).

### `opensmile` falla en import

- Probar `pip install --upgrade opensmile` (versión ≥ 2.5.0).
- Alternativa: ejecutar el binario openSMILE CLI vía subprocess y parsear su CSV. No implementado en esta fase pero es viable como fallback.

### NPZ muy grande

- `family_D_egemaps` (88d × N) puede crecer. Si se quiere descartar para iteración rápida, modificar `0A_extract.py` para pasar `include_egemaps=False` a `compute_all_descriptors`.
