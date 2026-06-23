# `experiments/voz_expresiva/` — Fase 0A piloto

Scripts del piloto Fase 0A del frente `Voz Expresiva Phideus`. Ver:

- Plan general: `Documents/01_FRENTES_ACTIVOS/Voz_Expresiva_Phideus/ROADMAP_VOZ_EXPRESIVA_PHIDEUS.md`
- Plan específico Fase 0A: archivo de plan en `~/.claude/plans/`

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

### 4) Análisis + reporte

```bash
python experiments/voz_expresiva/0A_analyze.py \
    --input data/esd/descriptors_0A_en.npz \
    --output-dir data/visualizations/voz_expresiva/0A
```

Genera plots PCA/UMAP por familia + compuesto, boxplots, JSONs de métricas (eta², MI, KW, silhouette, varianza intra/inter speaker) y `REPORTE_0A.md`.

## Política operativa

- **CPU only**. openSMILE y Praat son CPU-bound; no se usa GPU en esta fase.
- **Workers = 14** (16 cores - 2 al sistema).
- **No tmux necesario** — extracción estimada < 60 min en 17,500 utterances EN.
- **Outputs NO van a git**: `/data/` está globalmente excluido en el `.gitignore` raíz.

## Decisiones congeladas en Fase 0A (no reabrir sin plan mode)

- Vector compuesto = Familias A + B + C = 89d post-pool (raw canónico 29d).
- Pooling 4-stat (mean, std, max, min) sólo sobre frame-level (A y C). Familias B y D no se re-poolean.
- Normalización: z-score por hablante intra-corpus, declarada **transductiva** explícitamente.
- Scope: ESD English (10 speakers). Mandarin queda para más tarde.
- Splits: no aplica (no training); plan mode Fase 0B los congela.
- Familia B distingue **medidas directas** (7) de **proxies acústicos** (2). Los proxies se reportan como tales.
- Familia D (eGeMAPSv02) se reporta aparte, NO se concatena al compuesto.

## Métricas y cierre

El reporte cierra con la pregunta operativa:

> **¿qué familia justifica Fase 0B?**

— no con un GO/NO-GO global del frente. Lecturas esperables y umbrales orientativos están en el plan de Fase 0A.

## Troubleshooting

### `praat-parselmouth` falla en compilación

- Linux: instalar `libpraat-dev` o usar wheel pre-construido (`pip install praat-parselmouth --only-binary :all:`).
- Si falla en venv local, usar Python 3.10/3.11 (3.12 puede no tener wheels).

### `opensmile` falla en import

- Probar `pip install --upgrade opensmile` (versión ≥ 2.5.0).
- Alternativa: ejecutar el binario openSMILE CLI vía subprocess y parsear su CSV. No implementado en esta fase pero es viable como fallback.

### NPZ muy grande

- `family_D_egemaps` (88d × N) puede crecer. Si se quiere descartar para iteración rápida, modificar `0A_extract.py` para pasar `include_egemaps=False` a `compute_all_descriptors`.
