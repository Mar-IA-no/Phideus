# Gate 5 Linea B — Showcase Cross-Modal Extremo

**Estado**: CERRADO (Gate 5B Línea B completada: `Test02` 4/4, `Test13G-B` completo, paquete visual consolidado y cierre formal de Escalón 1-C)
**Fecha de actualizacion**: 2026-03-02
**Origen**: bateria de tests cientificos + visualizaciones para validacion extrema y comunicacion

---

## Concepto

Tomar el mejor modelo del proyecto, entrenarlo largo para maximo rendimiento,
y someterlo a una bateria de 13 tests cientificos ordenados por relevancia
para la tesis Phideus ("ratios como lenguaje informacional cross-modal").

## Estado operativo al 2026-03-02

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

### Test11 (estado operativo 2026-03-01)

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
- Test05 multi-seed ya quedó cerrado en `results_unc/gate5b_multiseed/`:
  - `D0 = 75.2% +/- 2.3pp`
  - `a4r = 80.7% +/- 1.9pp`
  - `d4-a4r = 81.2% +/- 2.5pp`
  - junto con la referencia multi-seed ya cerrada de `d4a4 = 84.1% +/- 2.3pp`, el ranking entre arms queda estadísticamente mucho más estable.
- Test02 parameter-matched quedó **cerrado 4/4**:
  - `real = 83.0%`;
  - `zero = 75.0%`;
  - `random = 73.6%`;
  - `shuffled = 73.6%*`.
  Con la misma arquitectura y la misma receta, las ablaciones sin descriptor real caen a banda `D0`: el argumento causal queda cerrado.
- Test13G Phase A ya cerró sobre `D0`:
  - `λ={0.03,0.1,0.3}` no cambia sustancialmente ni `S` ni reconstrucción;
  - `best_S≈64.4-64.6%`;
  - `audio_f1≈0.114`, `midi_f1≈0.118`;
  - las fases `B/C` originales quedan canceladas.
- Test13G Phase B ya quedó **completa**:
  - script: `experiments/bias_control/gate5b/test13g_posthoc_decoder.py`;
  - decoder post-hoc sobre features pre-pooling congeladas del encoder de audio;
  - resultado final: `D0 pool-188 = 0.1089`, `d4a4 = 0.1037`, `a4r = 0.1024`.
  Lectura: la decodificabilidad pre-pooling es genérica y no muestra ventaja descriptor-guided.
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
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/Explicacion_test_13G_faseB.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/Explicacion_resultados_test13g_y_02.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/INFORME_COMPLETO_GATE5B.md`
- Borrador de paper técnico: `Paper/main.tex` y `Paper/paper_standalone.tex`

### Lectura de cierre

- `Test05` aporta el cierre estadístico.
- `Test02` aporta el cierre causal de capacidad.
- `Test11` deja el hallazgo mecanístico más fuerte sobre el cuello de proyección.
- `13G-A` y `13G-B` cierran la línea generativa con una conclusión negativa útil: la ventaja descriptor-guided no aparece como mejor decodificabilidad de piano-roll.
- Con esto, Gate 5B Línea B queda cerrado y Escalón 2 puede abrirse sin bloqueo metodológico.

\* `shuffled` se tomó como cierre operativo por convergencia clara en `e20`.

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
