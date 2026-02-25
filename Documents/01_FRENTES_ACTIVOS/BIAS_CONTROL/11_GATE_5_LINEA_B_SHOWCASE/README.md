# Gate 5 Linea B — Showcase Cross-Modal Extremo

**Estado**: EN CURSO (Fase A parcialmente cerrada: Test12 + Test01)
**Fecha de actualizacion**: 2026-02-25
**Origen**: bateria de tests cientificos + visualizaciones para validacion extrema y comunicacion

---

## Concepto

Tomar el mejor modelo del proyecto, entrenarlo largo para maximo rendimiento,
y someterlo a una bateria de 13 tests cientificos ordenados por relevancia
para la tesis Phideus ("ratios como lenguaje informacional cross-modal").

## Estado operativo al 2026-02-25

### Checkpoints Gate 5B evaluados

- `models/gate5b/D0/best_model.pt`
- `models/gate5b/d4a4/best_model.pt`
- `models/gate5b/a4r/best_model.pt`
- `models/gate5b/d4-a4r/best_model.pt`

### Hitos cerrados

1. Loader universal + fix de evaluación:
   - `experiments/bias_control/gate5b/checkpoint_loader.py`
   - `experiments/bias_control/evaluate_structured_pool.py`
2. Test12 Scoreboard (config canonica, seed 42):
   - D0: 73.4%
   - d4a4: 83.8%
   - a4r: 82.0%
   - d4-a4r: 79.8%
3. Test01 Causal Ablation cerrado para los 4 arms:
   - Colapso fuerte al ablacionar audio descriptor (A4/A4r).
   - Efecto marginal/casi nulo al ablacionar D4 en modelos duales (`d4a4`, `d4-a4r`).

Detalle completo (tablas, interpretacion zero/noise/shuffle, evidencias):
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/plan_implementacion.md` (Anexo A)

### Proximo paso inmediato

- Ejecutar y cerrar Test04 (transposition invariance) reutilizando cache de embeddings ya disponible.

## Roadmap de tests (ordenado por relevancia cientifica)

Los primeros 5 son imprescindibles para cualquier publicacion.
6-9 fortalecen el argumento. 10-13 son comunicacion y completitud.

| # | Test | Dificultad | Relevancia |
|---|------|-----------|------------|
| 1 | Causal ablation (zero-out injection) | ~1h | Fundamental: causalidad |
| 2 | Parameter-matched ablations (ruido) | ~6h + GPU | Control de confound |
| 3 | RatioProbeDecoder + cross-decoding | ~8h | Smoking gun cross-modal |
| 4 | Invariancia transposicion MIDI | ~4h | Test directo hipotesis central |
| 5 | Multi-seed replication | GPU time | Reproducibilidad |
| 6 | RSA/CKA entre capas | ~1 dia | Evidencia mecanistica |
| 7 | Counterfactual Decoder | ~1 semana | Geometria del espacio |
| 8 | Ratio decoding report | ~1 dia | Paper-ready |
| 9 | Invariancia suite completa | ~1-2 dias | Constraints adicionales |
| 10 | UMAP/t-SNE de embeddings | ~3h | Exploratorio |
| 11 | CrossModalSequenceDecoder | ~3-5 dias | Demo generativa |
| 12 | Gate scoreboard reproducible | ~4h | Trazabilidad |
| 13 | Retrieval demo UI | ~2-3 dias | Showcase comunidad |

Detalle completo de cada test: `NOTAS_CLAUDE_PARA_CODEX.md` seccion 32.

---

## Dependencias

- Mejor modelo determinado (Gate 4.3 + Gate 4.4 + Gate 4.5 + Gate 5 Linea A)
- Train largo del mejor modelo para maximo rendimiento

## Fuente

- `ROADMAP_INSUMOS_GPT5.2PRO.md` — propuestas originales de GPT 5.2 Pro
- Seleccion y ordenamiento: equipo Claude + usuario (2026-02-15)
