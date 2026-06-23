# Voz Expresiva Phideus

> Frente exploratorio de Phideus que prueba si el patrón **descriptor ratio-based + mecanismo de inyección** (validado en Escalón 1 sobre música) transfiere al territorio de la expresión vocal y los correlatos paralingüísticos/afectivos del habla.

## Estado actual: Fase 0A + 0B CERRADAS — señal específica bajo adaptación, pendiente validación SSL (2026-06-22)

### Fase 0B — formulación calibrada del cierre

> Fase 0B **no valida todavía la hipótesis fuerte en speaker-independent estricto**. Sí valida, bajo adaptación mínima por hablante (N-adapt, 25 utts label-agnostic), que la familia A tiene **señal específica no reducible a un control espectral genérico**, y muestra una **mejora pequeña sobre eGeMAPS** que merece ser testeada en Fase 1 con WavLM.

Lectura desagregada:

| Condición | Resultado |
|---|---|
| **N-strict** (sin per-speaker en test) | Nadie supera chance. La hipótesis fuerte NO queda validada en generalización honesta a hablante nuevo con descriptors clásicos. |
| **N-adapt** — especificidad ratio | **Robusta**. A-only > C-only (+0.161-0.173), A+D > C+D (+0.112-0.114), C+D < D-only (-0.103). |
| **N-adapt** — A+D > D-only | Pequeña y clf-dependiente. +0.009 LogReg (marginal), +0.013 SVM RBF (Δ>0 robusto). |
| **N-adapt** — A+B > A-only | +0.047-0.085 robusto. B complementaria. |

Hallazgo metodológico aparte: 25 utts label-agnostic de calibración por hablante sacan a D-only y A+D de chance hasta ~0.6 UAR. Es hallazgo con potencial aplicado, no lectura directa de producto.

### Pendiente al cierre de Fase 0B

- Codex propaga el cierre a `00_TRONCAL/bitacora_desarrollo.md`, `INDICE_DOCUMENTACION.md`, `Proyecto_Estado_Actual.md` (nota en `NOTAS_CLAUDE-CODEX.md` S59 cont.).
- Plan mode formal para **Fase 1** — inyección Phideus-ratio en WavLM frozen, con foco en si SSL levanta el techo de N-strict y si A injection aporta sobre WavLM-only.

---

### Fase 0A — cerrada con GO direccional (referencia)

**Resultado Fase 0A**: los descriptores ratio-based muestran señal univariada significativa sobre ESD English (17,500 utterances).

| Familia | Top dim | eta² | Lectura |
|---|---|---|---|
| D — eGeMAPS baseline | F0semitone_percentile80 | **0.589** | el F0 lidera |
| **A — Phideus-ratio** | Hseries_d5_mean (concentración armónica) | **0.385** | señal específica ratio-based, 5× sobre el control |
| B — Voice quality | alpha_ratio | 0.262 | señal moderada |
| **C — control no-ratio** | A416k_d2_max | 0.076 | control cumplió rol (5× debajo de A) |

Lectura operativa:
- El control (Familia C) se quedó muy por debajo de A y B → la señal de A NO se reduce a "información espectral genérica".
- Phideus-ratio (Familia A, `Hseries_d5`) discrimina 5× sobre el control sin entrenar nada.
- eGeMAPS lidera por F0 — la pregunta "¿A aporta sobre D?" queda abierta para Fase 0B (clasificador real).

Decisión del usuario: **GO direccional** para Fase 0B.

### Caveats declarados (por trazabilidad)

- **Silhouette ~ 0 en todas las familias**: las distribuciones de emoción se superponen mucho geométricamente, pero hay señal univariada clara. Es típico de SER actuado, no es problema de descriptores.
- **CPP en escala comprimida**: la implementación manual da ~10× menos que los valores clínicos estándar. Discrimina pero no es directamente comparable a la literatura. Fix posible en 0B.
- **H1-H2 / H1-A3 son proxies**: sin corrección formántica. Capturan parte de la señal del modo de fonación; implementación clínica completa va si Fase 0B la requiere.
- **Normalización transductiva**: z-score por hablante sobre todo el corpus EN. Fase 0B debe redefinir splits y re-normalizar dentro del train.
- **ESD es actuado**: cualquier resultado acá necesita pasar por Fase 3 (MSP-Podcast naturalístico) antes de generalizarse.

### Artefactos de Fase 0A

- `data/esd/descriptors_0A_en.npz` — 15 MB, 17,500 vectores × 4 familias.
- `data/visualizations/voz_expresiva/0A/` — 17 plots + 3 JSONs + REPORTE_0A.md.
- `src/voz_expresiva/` — código del pipeline.
- `experiments/voz_expresiva/` — scripts de extract + analyze.

### Pendiente al cierre de Fase 0A (histórico — ya superado por Fase 0B)

Cumplido. Fase 0B ejecutada y cerrada el mismo 2026-06-22; resultados arriba.

## Estatus de `EIR-EMR/`

`EIR-EMR/` queda preservado como **antecedente exploratorio, no como nombre vigente ni como roadmap activo**. Su contenido fue reformulado en este frente tras la investigación bibliográfica de junio 2026. La nomenclatura "EIR / EMR" como marca maestra del frente fue descartada por ser un lenguaje preconcebido antes de tener evidencia. Para trazabilidad histórica ver `../EIR-EMR/`.

## Documentación clave

- **Roadmap general del frente**: `./ROADMAP_VOZ_EXPRESIVA_PHIDEUS.md`
- **Investigación bibliográfica de junio 2026**: referencia editorial externa al repo Phideus, en `editorial-altermundi/Biblioteca/analisis-carga-emocional-del-habla/`. Cubre 7 reportes de subagentes (productos, OSS, datasets, métodos, entrenamiento/licencias, substrato físico EGG/cross-modal, disentanglement/voice conversion/expressive TTS), un NARRATED_REPORT integrador (~60 KB), bibliografía deduplicada y un CROSS_REPORT denso.
- **Antecedente exploratorio**: `../EIR-EMR/` (preservado, no activo).

## Próximo paso

Plan mode formal para **Fase 1** — inyección Phideus-ratio (familia A) en WavLM frozen, con foco en:

1. ¿WavLM solo levanta el techo de N-strict que limitó a los descriptors clásicos en Fase 0B?
2. ¿La inyección de A en WavLM (concat / FiLM / xattn) aporta sobre WavLM-only bajo generalización honesta a hablante nuevo?

Si WavLM saltea el techo de N-strict y A injection aporta encima, ese es el caso target real. Si Fase 1 también queda atorada en N-strict, sabremos que la generalización speaker-independent estricta requiere otra estrategia.

### Spike Fase 1.0 (cerrado 2026-06-22)

Documento: `experiments/voz_expresiva/SPIKE_FASE_1_0.md`.

Hallazgo: los mecanismos heredados de E2 (`SpeechEGGEncoderAug`, `SpeechEGGEncoderXAttn`, `ConditionedProjectionHead`) **NO son drop-in para WavLM**. Asumen una topología CNN+Transformer propio de 512d hardcoded, mientras WavLM da `[B, T, 1024]` frozen.

- `ConditionedProjectionHead` (FiLM): **drop-in utterance-level** con `input_dim=1024, cond_dim=12`.
- Concat: re-implementar el algoritmo (near-identity init) a 1024d. ~25 líneas.
- xattn: re-implementar con `MultiheadAttention(embed_dim=1024)` y `xattn_scale=0.01`. ~40 líneas.
- Total adaptación: ~105 líneas en `src/voz_expresiva/wavlm_injection.py`. Algoritmos heredados aplicables tal cual; cambian dimensiones y punto de inserción (post-WavLM en vez de inter-CNN-Transformer).

El plan general de Fase 1 se sostiene; la sección de reuso del plan archivado se actualiza para reflejar el wrapper en lugar de "import as-is".
