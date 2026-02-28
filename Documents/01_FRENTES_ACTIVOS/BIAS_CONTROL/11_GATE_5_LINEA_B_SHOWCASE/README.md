# Gate 5 Linea B — Showcase Cross-Modal Extremo

**Estado**: EN CURSO (paquete local cerrado: Test12/01/04/03/06/08/10/Test09; **Pre-Proj A/B completo para `D0+a4r`**; `Test13G` corriendo en `Phase A`; UNC con `Test05` en `9/15` sync local, `10/15` runtime reportado y `Test02` `4/4` en cola)
**Fecha de actualizacion**: 2026-02-28
**Origen**: bateria de tests cientificos + visualizaciones para validacion extrema y comunicacion

---

## Concepto

Tomar el mejor modelo del proyecto, entrenarlo largo para maximo rendimiento,
y someterlo a una bateria de 13 tests cientificos ordenados por relevancia
para la tesis Phideus ("ratios como lenguaje informacional cross-modal").

## Estado operativo al 2026-02-28

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

### Test11 (estado operativo 2026-02-28)

- Se detuvo corrida cuantitativa previa en `tmux test11` para priorizar perceptualidad humana.
- Se preservó baseline cuantitativo ya obtenido (`D0`, `a4r`, `baselines`) y samples legacy.
- Pipeline nuevo implementado y en operación:
  - `experiments/bias_control/gate5b/test11_perceptual_suite.py`
  - `experiments/bias_control/gate5b/{midi_event_codec,event_decoder_model,render_midi_audio,eval_perceptual_human}.py`
- Barridos perceptuales cerrados:
  - `midi2events` sweep base en `D0` y `a4r`.
  - sweep fino GPU en `D0` (`v1` + `v2`), con preferencia humana provisional en `config 07` (`t104_k44_p099`).
  - sweep `audio2events` enfocado en `a4r` cerrado (8 configs, best automático `07_t100_k64_p098`).
- A/B pre-projection **completo** para `D0` y `a4r`:
  - `D0 preproj_midi2events`: CE `2.9449`, token_acc `0.3108`, frame F1 `0.1250`, `shuffle_gap=1.1498`;
  - `D0 preproj_audio2events`: CE `3.070`, token_acc `0.290`, frame F1 `0.050`, `shuffle_gap=0.186`;
  - `a4r preproj_midi2events`: CE `2.947`, token_acc `0.306`, frame F1 `0.120`, `shuffle_gap=1.159`;
  - `a4r preproj_audio2events`: CE `3.070`, token_acc `0.290`, frame F1 `0.046`, `shuffle_gap=0.304`.
- Hallazgo principal:
  - `information retention ratio`: `D0=0.597`, `a4r=0.712`;
  - la proyeccion MIDI 512→256 destruye aproximadamente `81-88%` de la informacion condicionante.
- Sincronización de compartidos:
  - `resultados_compartir/06_gate5b_scientific_validation/test11_decoder_suite/` actualizado con `a4r` completo.
  - `resultados_compartir/06_gate5b_scientific_validation/test11_perceptual/` mantiene arbol completo por arm/barrido/config.
  - `resultados_compartir/06_gate5b_scientific_validation/test11_perceptual/compilacion/README.md` documenta el paquete local consolidado de 608 archivos.

Detalle completo (tablas, interpretación zero/noise/shuffle y avance de transposición):
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/INFORME_EJECUCION_TEST01_TEST12_2026-02-25.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/Explicacion_de_test.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/Explicacion_A4.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/Explicacion_test_08.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/Explicaccion_pre-projection_test.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/Explicacion_test_13G.md`
- Borrador de paper técnico: `Paper/main.tex` y `Paper/paper_standalone.tex`

### Proximo paso inmediato

- Monitorear y cerrar `Test13G` Phase A sobre `D0` (`tmux test13g`) para seleccionar `λ*`.
- Sostener la separación entre `sync local 9/15` y `runtime UNC 10/15` hasta que entren nuevos artefactos en `results_unc/`.
- Cerrar el bloque `D0` de Test05 en UNC y luego ejecutar Test02 parameter-matched (`4` modos).

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

Detalle operativo y narrativo de este bloque:
- `Documents/NOTAS_CLAUDE-CODEX.md` (secciones `11.30` a `13.4`)
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/Explicaccion_pre-projection_test.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/Explicacion_test_13G.md`

---

## Dependencias

- Mejor modelo determinado por Gate 4.3 + Gate 4.4 + Gate 4.5; Gate 5 Linea A puede aportar candidatos adicionales en paralelo, sin bloquear Gate 5B
- Train largo del mejor modelo para maximo rendimiento

## Fuente

- `ROADMAP_INSUMOS_GPT5.2PRO.md` — propuestas originales de GPT 5.2 Pro
- Seleccion y ordenamiento: equipo Claude + usuario (2026-02-15)
