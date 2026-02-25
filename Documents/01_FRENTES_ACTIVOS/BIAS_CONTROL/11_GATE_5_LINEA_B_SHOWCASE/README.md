# Gate 5 Linea B — Showcase Cross-Modal Extremo

**Estado**: EN CURSO (paquete local cerrado: Test12/01/04/03/06/08/10/Test09; UNC pendiente: Test02/Test05; Test11 en curso local)
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
- `models/gate5b/d4/best_model.pt`
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
3. Test01 Causal Ablation cerrado para 5 arms (`D0`, `d4`, `d4a4`, `a4r`, `d4-a4r`):
   - Colapso fuerte al ablacionar audio descriptor (A4/A4r).
   - Efecto marginal/casi nulo al ablacionar D4 en modelos duales (`d4a4`, `d4-a4r`) y señal débil en `d4` puro.
4. Test04 Transposition cerrado para los 4 checkpoints canónicos (`D0`, `d4a4`, `a4r`, `d4-a4r`).
5. Tests mecanísticos cerrados:
   - Test03 RatioProbe
   - Test06 RSA/CKA
   - Test08 Ratio Decoding
   - Test10 Visualizations
6. Test09 Invariance Suite (cerrado en 4 arms):
   - `D0`, `d4a4`, `a4r`, `d4-a4r` con JSON canónico.
   - Patrón consolidado:
     - robustez temporal aceptable en todos los arms;
     - fragilidad alta a velocity scaling y transposición de octava;
     - robustez a ruido con patrón bimodal (`D0` domina en 40-20 dB, `a4r/d4-a4r` retienen más a 5 dB).
7. Paquete visual Gate 5B consolidado:
   - `24 PNG` + `6 GIF`
   - `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/resultados_compartir/06_gate5b_scientific_validation/`

Detalle completo (tablas, interpretación zero/noise/shuffle y avance de transposición):
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/INFORME_EJECUCION_TEST01_TEST12_2026-02-25.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/Explicacion_de_test.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/Explicacion_A4.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/Explicacion_test_08.md`
- Borrador de paper técnico: `Paper/main.tex` y `Paper/paper_standalone.tex`

### Proximo paso inmediato

- Coordinar fase UNC pendiente: Test02 (parameter-matched) y Test05 (multi-seed).
- Consolidar lectura final de Test11 (decoder suite) como evidencia complementaria generativa/no lineal.

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
