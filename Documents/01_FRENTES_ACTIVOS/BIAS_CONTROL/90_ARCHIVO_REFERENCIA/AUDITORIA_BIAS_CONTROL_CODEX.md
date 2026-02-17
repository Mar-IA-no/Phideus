# AUDITORIA_BIAS_CONTROL_CODEX

Fecha: 2026-02-11
Estado: auditoria tecnica v1 + addendums operativos (Gate 4.1 cerrado, diagnostico post Gate 4.1 completado, Bloque A v1.1 activo)
Autor: Codex
Modo colaboracion agentes: OFF durante esta auditoria

> [!NOTE]
> Addendum de vigencia (2026-02-17): este documento se conserva como auditoría histórica.
> El estado operativo actual del frente se sigue en el roadmap y en el informe de Gate 4.3.
> Corte actual Gate 4.3: fase cerrada con 13 brazos + scratch completo.

---

## Addendum de consistencia (2026-02-11)

Estado sincronizado con roadmap vigente:

- El diagnostico post Gate 4.1 ya se ejecutó y cerró:
  - Gate 6: hallazgo causal de `audio encoder` congelado.
  - Gate 4.2 pre-red (H4.2-6): NO-GO por señal no discriminativa.
- Etapa activa actual:
  - `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/05_PLAN_POST_DIAGNOSTICO_BLOQUE_A/PLAN_EJECUCION_POST_DEC005_v1.1.md` (Bloque A: `S0/A/B/C`).
- Esta auditoria queda como base histórica; el estado operativo se sigue en:
  - `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
  - `Documents/00_TRONCAL/Proyecto_Estado_Actual.md`

---

## Addendum operativo (2026-02-11)

Estado post-cierre Gate 4.1 y apertura diagnóstica:

- Gate 4.1 (`DEC-004-A`) cerrado por criterio pre-registrado:
  - `RB0`: A2M R@10=30.2, M2A R@10=38.2, hard_neg=77.6
  - `R1-rescue`: A2M R@10=31.0, M2A R@10=40.2, hard_neg=78.8
  - `dS=+0.8pp` vs `RB0` (insuficiente para promoción)
- `DEC-005` registrada en modo `diagnostic-only`:
  - Track 1: Gate 6 retroanálisis.
  - Track 2: Gate 4.2 pre-red dual-domain (`P0/P1`) con AUC + delta_sim + Wilcoxon + bootstrap CI.
  - Sin entrenamiento automático; requiere DEC posterior explícita.
- Estado de colaboración inter-agente: `COLLAB OFF` (sin intercambio cruzado activo).

Este addendum actualiza la fase activa de ejecución y reemplaza el supuesto “Gate 4.1 en curso” por “Gate 4.1 cerrado + diagnóstico DEC-005”.

## Addendum infraestructura (2026-02-11)

La linea paralela de optimizacion runtime (`VibeTensor`) queda en **pausa operativa** para priorizar el cierre de `BIAS_CONTROL`:

- rama experimental: `exp/vibetensor-spike`,
- worktree sugerido: `/tmp/phideus-vibetensor-spike`,
- `main` conserva la linea cientifica oficial de `BIAS_CONTROL`.

Criterio de reactivacion:
1. completar Bloque A del plan post-diagnostico (S0/A/B/C),
2. cerrar auditoria final de Escalon 1-C,
3. recien entonces evaluar integracion de resultados del spike.

Criterio de integracion a `main`:
1. evidencia de mejora local reproducible (hardware objetivo),
2. sin regresión en métricas del protocolo canónico,
3. costo de mantenimiento razonable.

---

## Addendum operativo (2026-02-10)

Estado post-auditoria v1:

- Gate 4 Run A completado en regimen alineado a Gate 2:
  - `max-batches-per-epoch=1000`
  - `max-val-batches=846`
  - `seed=42`
- Structured pool Run A:
  - `RA5`: `A2M R@10=31.4`, `M2A R@10=40.6`, `hard_neg=79.0`
  - `RA30`: `A2M R@10=29.2`, `M2A R@10=36.4`, `hard_neg=74.8`
- `gate4_ratio_auxiliary.py` quedo actualizado para:
  - limitar train/val por CLI,
  - ajustar scheduler a steps efectivos,
  - fijar seed reproducible A/B,
  - mantener hardening de checkpoint-before-eval y fix de device mismatch en evaluación.
- DEC-004/004-A para Gate 4.1:
  - Fase 0 (`RB0`, `ratio_weight=0.0`, 5 epochs) completada.
  - Resultado causal: `dS=+1.2pp`, `dH=+1.4pp` (`RA5` vs `RB0`), clasificado como zona inconclusa.
  - Siguiente paso habilitado: `R1-rescue` (descriptor enriched, 5 epochs, mismo régimen/seed) antes de cerrar Gate 4.1.

Este addendum no cambia las conclusiones troncales de la auditoria v1; actualiza el estado de ejecución.

---

## 1) Resumen ejecutivo

Esta auditoria revisa roadmap, implementacion y resultados de BIAS_CONTROL (Gates 0-4), con foco en comparabilidad de metricas, causalidad de mejoras y decision de segundo ciclo experimental.

Conclusiones principales (actualizadas):

1. No conviene re-ejecutar todo el roadmap completo ahora.
2. Conviene rerun selectivo, no rerun total.
3. Gate 2 sigue siendo baseline robusto (checkpoint epoch45), y Gate 3 (DANN) queda cerrado/depriorizado.
4. Gate 4.1 quedó cerrado por reglas causales pre-registradas (`DEC-004-A`).
5. Gate 5 no es prioridad ahora.
6. La fase activa es post-diagnóstico: Bloque A v1.1 (`S0/A/B/C`).
7. La auditoria final global debe cerrarse después de completar Bloque A.

## Ubicacion en la escalera Rosetta

Esta auditoria asume explicitamente:

- `BIAS_CONTROL = Escalon 1` de `Documents/00_TRONCAL/ROADMAP_GENERAL/Rosetta_triplescaloneta.md`.
- Estructura interna del Escalon 1:
  - `Escalon 1-A`: Gates 0/1/2.
  - `Escalon 1-B`: Gate 3.
  - `Escalon 1-C`: Gate 4 base + Gate 4.1 + Gate 6.

Implicacion:
- Antes de abrir Escalon 2 (speech<->EGG), debe cerrarse Escalon 1-C con evidencia causal y representacional.

---

## 2) Objetivo, alcance y restricciones

Objetivo:
- Determinar si corresponde rerun total o selectivo.
- Determinar si conviene iterar Gate 2 (hparams/config/arquitectura).
- Priorizar cambios con mejor retorno esperado.

Alcance incluido:
- Roadmap y resultados de Gates 0, 1, 2, 2.5, 3 y estado operativo de 4.
- Consistencia entre documentos, logs, JSON de evaluacion y checkpoints.
- Matriz de decision por gate y backlog de experimentos.

Fuera de alcance en esta fase:
- Ejecutar entrenamientos nuevos para la auditoria.
- Modificar scripts del pipeline.
- Cerrar version final post Gate 4.1 y Gate 6.

---

## 3) Fuentes de verdad auditadas

### Documentacion
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/90_ARCHIVO_REFERENCIA/BIAS_CONTROL_MEDIUM_TEST_RESULTS.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/90_ARCHIVO_REFERENCIA/BIAS_CONTROL_FAST_TEST_RESULTS.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/01_GATES_0_2_5/GATE_2_FOUNDATION/INFORME_GATE2_COMPLETO.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/02_GATE_3_DANN/INFORME_GATE3_COMPLETO.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/03_GATE_4_4_1_RATIO/PLANES/plan_gate4.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/03_GATE_4_4_1_RATIO/PLANES/plan_gate4_codex.md`

### Resultados / metricas
- `data/bias_control_medium/evaluations/structured_pool_epoch45.json`
- `data/bias_control_medium/evaluations/audit_gate2/audit_gate2_results.json`
- `data/bias_control_medium/evaluations/gate2_5/gate2_5_results.json`
- `data/bias_control_medium/evaluations/gate3_comparison/comparison_summary.json`
- `data/bias_control_medium/evaluations/gate3_comparison/runD_best_ep12.json`
- `data/bias_control_medium/training_outputs/gate2/training_history.json`
- `data/bias_control_medium/training_outputs/gate2/gate2_results.json`
- `data/bias_control_medium/training_outputs/gate3_d/gate3_results.json`
- `/tmp/gate4_smoke.log`

### Codigo
- `experiments/bias_control/gate2_foundation.py`
- `experiments/bias_control/gate4_ratio_auxiliary.py`
- `src/bias_control/architectures/cross_modal_model.py`
- `src/bias_control/encoders/midi_encoder.py`
- `src/bias_control/datasets/maestro_segments.py`

---

## 4) Metodo de auditoria aplicado

1. Se normalizaron conclusiones a protocolo canonico de decision: structured pool `256 / 500 / seed 42`.
2. Se separo evidencia comparable vs no comparable.
3. Se analizo cada gate por: validez tecnica, calidad de metrica, retorno esperado de rerun, riesgo y costo.
4. Se evaluo criterio contrafactual para segundo ciclo:
   - rerun total,
   - rerun selectivo,
   - rediseno focalizado.

---

## 5) Hallazgos por gate

## Gate 0 (integridad y alineacion)

Estado: funcional como precondicion.

Evidencia:
- En fast test, Gate 0 pasa alineacion y segmentacion operativa (`FAILED_GATE_2` pero no por Gate 0).
- Auditoria Gate 2 reporta A2 FAIL como falso positivo metodologico (medicion de offset no confiable para concluir desalineacion).

Lectura tecnica:
- No hay evidencia de bug sistemico de datos que justifique rerun total del roadmap.

Decision:
- KEEP (mantener), con mejora metodologica del check A2.

---

## Gate 1 (intra-modal)

Estado: util como sanity gate, no como gate decisor principal.

Evidencia:
- En fast test aparece NO-GO por gap threshold, con metricas intra-modales mezcladas que no reflejan bien el objetivo final cross-modal.

Lectura tecnica:
- Gate 1 sirve para detectar roturas gruesas, pero su criterio actual puede producir ruido de decision.

Decision:
- RE-DESIGN (como validacion ligera, no como cuello de botella de pipeline).

---

## Gate 2 (foundation baseline)

Estado: baseline mas robusto disponible hoy para decision de producto experimental.

Evidencia estructurada (comparable):
- `R@10 a2m = 34.4%`
- `R@10 m2a = 37.6%`
- `MRR a2m = 0.138`
- `Hard negative accuracy = 80.4%`
- `Mean rank a2m = 37.42`

Observaciones clave:
- `training_outputs/gate2/gate2_results.json` marca NO-GO por umbrales globales historicos; la metrica decisora actual del proyecto es structured pool, donde el checkpoint pasa claramente.
- En `training_history.json`, el mejor epoch por gap es 45 (seleccionado), mientras recall de validacion global tiene picos en epocas anteriores (ej. 25), mostrando desacople entre metrica global y metricas de pool estructurado.

Lectura tecnica:
- Gate 2 esta bien como baseline operativo.
- Tiene sentido explorar iteracion selectiva de Gate 2, pero no antes de cerrar evidencia causal de Gate 4.1 y sin romper comparabilidad.

Decision:
- RE-RUN selectivo (condicional y focalizado).

---

## Gate 2.5 (probes)

Estado: diagnostico util, pero insuficiente como criterio causal unico.

Evidencia:
- `linear_separability = 92.675%` (Audio vs MIDI)
- `silhouette_by_piece_audio = -0.111`
- `dead_dims = 0/256`

Lectura tecnica:
- Detecta separabilidad modal alta, pero Gate 3 demostro empiricamente que atacar eso con DANN no mejora retrieval final.
- Debe tratarse como tablero diagnostico, no como regla de accion unica.

Decision:
- RE-DESIGN (probes mas causales y conectados a retrieval final).

---

## Gate 3 (DANN)

Estado: linea cerrada para esta etapa.

Evidencia comparable (structured pool):
- Gate2_ep45: `R@10 a2m 34.4%`, `hard_neg 80.4%`
- RunA_best_ep7: `27.8%`, `74.8%`
- RunB_ep5: `24.6%`, `70.4%`
- RunB_ep10: `29.8%`, `73.6%`
- RunC_best_ep4: `34.6%`, `81.2%` (margen pequeno)
- RunC_ep13: `32.2%`, `76.6%`
- RunD_best_ep12: `27.4%`, `73.2%`

Lectura tecnica:
- No hay mejora robusta y estable sobre Gate 2.
- RunC_ep4 empata marginalmente, pero sin ventaja clara bajo ruido muestral.
- En regimen sostenido, DANN degrada retrieval.

Decision:
- DEPRECATE (por ahora).
- Reabrir solo con hipotesis nueva y disenio experimental acotado.

---

## Gate 4 / Gate 4.1 / Gate 4.2 (ratio auxiliary)

Estado:
- Gate 4 base: completado (Run A de 30 epocas, evaluación estructurada disponible).
- Gate 4.1: cerrado (`DEC-004-A`).
- Gate 4.2: diagnóstico habilitado en `DEC-005` (sin training).

Evidencia de implementacion:
- Script actualizado con correcciones estructurales clave:
  - `use_dann=False` sobre base Gate 2.
  - freeze solo de encoder de audio; resto entrenable.
  - checkpoints duales (full + `*_base.pt`) para compatibilidad de evaluacion.
  - CLI con `--segment-len 4.0`, `--hop 1.0` y regimen alineado a baseline.
  - fix de evaluacion por device mismatch (`piece_idx`/`segment_idx` en CPU).
  - guardado de checkpoint antes de `evaluate()`.

Evidencia de ejecucion:
- Run A (30 epocas) ya evaluado en structured pool:
  - `RA5`: `A2M R@10=31.4`, `M2A R@10=40.6`, `hard_neg=79.0`.
  - `RA30`: `A2M R@10=29.2`, `M2A R@10=36.4`, `hard_neg=74.8`.
- Patron observado: mejor desempeño temprano y degradacion con entrenamiento largo.

Lectura tecnica:
- Hay señal util, pero insuficiente para promoción en Gate 4.1 tras control causal y rescue único.
- El siguiente paso correcto no es entrenar directo, sino cerrar diagnóstico (`DEC-005`) para separar causa de asimetría y viabilidad dual-domain.

Decision:
- Gate 4.1: CERRADO.
- Gate 4.2/Gate 6: DIAGNÓSTICO PRIORITARIO (`DEC-005`).

---

## Gate 5 (curriculum dominio)

Estado: opcional en roadmap.

Lectura tecnica:
- Alto costo y cambia regimen de datos.
- Menor prioridad que cerrar causalidad de Gate 4.1 y analitica de Gate 6.

Decision:
- HOLD (no priorizar ahora).

---

## Gate 6 (retroanalisis RSA/CKA/probes/disagreement)

Estado: pendiente.

Lectura tecnica:
- Alto valor explicativo para:
  - entender por que Gate 4.1 mejora o no,
  - decidir si conviene redisenar arquitectura,
  - evitar reruns ciegos de toda la ruta.

Decision:
- PRIORIDAD ALTA despues de Gate 4.1.

---

## 6) Comparabilidad de metricas: lo que vale para decidir

## Comparable (si)
- Structured pool con `256/500/seed42` para Gate 2 y comparativas Gate 3.

## No comparable (o solo diagnostico)
- Recall de training loop con distintos tamanos de pool de validacion.
- Fast test para concluir performance final.
- Probes de separabilidad modal como criterio unico de accion.

---

## 7) Matriz final GateDecision (v1)

| gate_id | status | evidencia | expected_gain | risk | compute_cost | next_action |
|---|---|---:|---|---|---|---|
| Gate 0 | KEEP | alta | baja | baja | baja | mejorar metrica A2 (alineacion) |
| Gate 1 | RE-DESIGN | media | baja | media | baja | reducir peso decisor, dejarlo como sanity gate |
| Gate 2 | RE-RUN (selectivo) | alta | media-alta | media | media | evaluar mas checkpoints existentes + tuning incremental condicional |
| Gate 2.5 | RE-DESIGN | alta | media | media | baja | ampliar probes causales |
| Gate 3 | DEPRECATE | alta | baja | alta | alta | cerrar linea DANN por ahora |
| Gate 4.1 | CERRADO | alta | baja | baja | ya incurrido | no reabrir sin hipótesis nueva fuerte |
| Gate 4.2 | DIAGNOSTIC-ONLY | media | alta (informacional) | media | media | ejecutar pre-red `P0/P1` y decidir training en DEC posterior |
| Gate 5 | HOLD | media | incierta | media-alta | alta | postergar |
| Gate 6 | PRIORITARIO | media-alta | alta (informacional) | media | media | ejecutar en paralelo con Gate 4.2 (DEC-005) |

---

## 8) Respuesta a la pregunta estrategica

Pregunta: tiene sentido seguir con Gate 4.1, 5 y 6 antes de volver a auditar todo BIAS_CONTROL?

Respuesta:

1. Gate 4.1: ya está cerrado (no corresponde seguir iterando esa rama sin nueva evidencia).
2. Gate 5: NO, por ahora (opcional y costoso).
3. Gate 6: SI, junto con Gate 4.2 pre-red bajo `DEC-005`.
4. Re-auditoría global completa: DESPUÉS de completar `DEC-005`.

Razon:
- Gate 4.1 entrega evidencia causal de mejora practica.
- Gate 6 entrega evidencia explicativa para decisiones de arquitectura y reruns.
- Sin esos dos cierres, una auditoria global nueva quedaria incompleta o especulativa.

---

## 9) ExperimentBacklog propuesto (segundo ciclo)

| exp_id | objetivo | delta_hypothesis | acceptance_threshold | prioridad | dependencias |
|---|---|---|---|---|---|
| EXP-001 | Sweep structured de checkpoints Gate 2 ya entrenados | puede existir checkpoint > ep45 en pool estructurado | >= +1.5 pp en R@10 a2m o hard_neg | P0 | ninguna |
| EXP-002 | Gate 4.1 Fase 0 causal (`RA5 vs RB0`) | ratio aporta efecto propio | `S_RA5 - S_RB0 >= +1.5 pp` y `H_RA5 >= H_RB0 - 1pp` | P0 | EXP-001 opcional |
| EXP-003 | Gate 4.1 Fase 1 (R1-R4) | descriptor/weight puede mejorar simetria y hard neg | seleccionar 1-2 ganadores para 30 epocas | P0 | EXP-002 |
| EXP-004 | Tuning incremental Gate 2 | hay headroom real en baseline | >= +2.0 pp dual criterio vs ep45 | P1 | EXP-001 + EXP-002 |
| EXP-005 | Cambios de arquitectura Gate 2/4 | limite de hparams alcanzado | >= +3.0 pp dual criterio | P2 | EXP-004 fallido |

---

## 10) Interfaces de auditoria utilizadas

## AuditRecord
Campos:
- `source_path`
- `metric_name`
- `value`
- `protocol_tag`
- `comparable_flag`
- `note`

## GateDecision
Campos:
- `gate_id`
- `status` (`KEEP`, `RE-RUN`, `RE-DESIGN`, `DEPRECATE`, `HOLD`, `PRIORITARIO`)
- `evidence_score`
- `expected_gain`
- `risk`
- `compute_cost`
- `next_action`

## ExperimentBacklog
Campos:
- `exp_id`
- `objective`
- `delta_hypothesis`
- `acceptance_threshold`
- `priority`
- `dependencies`

---

## 11) Criterio de cierre de auditoria final (v2)

Esta auditoria v1 se considera parcial. La version final (`AUDITORIA_BIAS_CONTROL_CODEX_FINAL`) debe actualizarse cuando se complete:

1. Gate 4.1 completo con comparacion causal por fases (DEC-004) y structured pool homogeneo.
2. Gate 6 completo con RSA/CKA, probes y disagreement analysis.

Tras eso se consolidara:
- decision definitiva sobre iterar Gate 2,
- necesidad real de cambios de arquitectura,
- y estrategia de rerun final del roadmap.
