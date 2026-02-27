# Gate 5 Linea B — Showcase Cross-Modal Extremo

**Estado**: EN CURSO (paquete local cerrado: Test12/01/04/03/06/08/10/Test09; UNC en progreso: Test05 `9/15` cerradas + Test02 pendiente; línea generativa activa con pre-projection A/B y Test13G listo para ejecución)
**Fecha de actualizacion**: 2026-02-27
**Origen**: bateria de tests cientificos + visualizaciones para validacion extrema y comunicacion

---

## Concepto

Tomar el mejor modelo del proyecto, entrenarlo largo para maximo rendimiento,
y someterlo a una bateria de 13 tests cientificos ordenados por relevancia
para la tesis Phideus ("ratios como lenguaje informacional cross-modal").

## Estado operativo al 2026-02-26

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

### Test11 (estado operativo 2026-02-26)

- Se detuvo corrida cuantitativa previa en `tmux test11` para priorizar perceptualidad humana.
- Se preservó baseline cuantitativo ya obtenido (`D0`, `a4r`, `baselines`) y samples legacy.
- Pipeline nuevo implementado y en operación:
  - `experiments/bias_control/gate5b/test11_perceptual_suite.py`
  - `experiments/bias_control/gate5b/{midi_event_codec,event_decoder_model,render_midi_audio,eval_perceptual_human}.py`
- Barridos perceptuales cerrados:
  - `midi2events` sweep base en `D0` y `a4r`.
  - sweep fino GPU en `D0` (`v1` + `v2`), con preferencia humana provisional en `config 07` (`t104_k44_p099`).
  - sweep `audio2events` enfocado en `a4r` cerrado (8 configs, best automático `07_t100_k64_p098`).
- Estado de ejecución al último corte:
  - sin procesos activos de `test11_*` al cerrar el sweep `audio2events` de `a4r`;
  - entrenamiento `audio2events` (`D0 -> a4r`) queda pendiente de relanzamiento tras validación humana del barrido;
  - enfoque cache-first confirmado (`--skip-train-embs`) para minimizar I/O y tiempo de setup en el siguiente run.
- Sincronización de compartidos:
  - `resultados_compartir/06_gate5b_scientific_validation/test11_decoder_suite/` actualizado con `a4r` completo.
  - `resultados_compartir/06_gate5b_scientific_validation/test11_perceptual/` mantiene árbol completo por arm/barrido/config.

Detalle completo (tablas, interpretación zero/noise/shuffle y avance de transposición):
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/INFORME_EJECUCION_TEST01_TEST12_2026-02-25.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/Explicacion_de_test.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/Explicacion_A4.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/Explicacion_test_08.md`
- Borrador de paper técnico: `Paper/main.tex` y `Paper/paper_standalone.tex`

### Proximo paso inmediato

- Completar A/B pre-projection (`tmux preproj_ab`) para aislar cuello de botella `pre-proj vs post-proj`.
- Ejecutar Test13G (Phase A D0 λ sweep) al liberar GPU local.
- Cerrar `d4-a4r_seed1337`, lanzar bloque `D0` de Test05 y luego Test02 en UNC.

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
