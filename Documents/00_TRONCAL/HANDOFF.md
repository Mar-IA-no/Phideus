# HANDOFF Operativo - Proyecto Phideus

---

## Proposito

Este documento es el puente de continuidad entre sesiones, agentes e instancias.
No reemplaza roadmaps ni decisiones formales: sintetiza estado operativo real y proximo paso ejecutable.

---

## Precedencia de fuentes

Cuando `collab_mode=on`:
1. `COLLAB/STATUS.md`: snapshot operativo "ahora" del ciclo collab.
2. `COLLAB/DECISIONS.md`: decisiones formales vigentes del protocolo.
3. `Documents/00_TRONCAL/HANDOFF.md` (este archivo): continuidad entre sesiones/instancias.
4. `COLLAB/HANDOFFS.md`: historial de traspasos entre agentes.

Cuando `collab_mode=off`:
1. `COLLAB/DECISIONS.md` (decisiones históricas válidas).
2. `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md` + `Documents/00_TRONCAL/Proyecto_Estado_Actual.md` (estado operativo real).
3. `Documents/00_TRONCAL/HANDOFF.md` (este archivo).
4. `COLLAB/STATUS.md` y `COLLAB/HANDOFFS.md` como referencia histórica (pueden quedar stale).

---

## Como usar este documento

1. Actualizar solo al cierre de un hito operativo o ante corte de contexto.
2. Mantener formato breve, verificable y con rutas concretas.
3. Registrar una sola "ultima decision valida" por entrada.
4. No duplicar contenido de roadmap: solo referenciarlo.

---

## Plantilla de entrada

```md
## YYYY-MM-DD HH:MM (UTC) - Handoff

### Metadata
- as_of_commit: <hash>
- collab_mode: on|off
- from_to (opcional): <origen -> destino>
- turn_summary_ref (opcional): <ruta o id>

### Estado real verificado
- item

### Ultima decision valida
- item

### Proximo paso unico recomendado
- item

### Bloqueantes / riesgos
- item

### Evidencia y archivos clave
- path
```

## 2026-02-14 06:40 (UTC) - Handoff

## 2026-02-14 14:45 (UTC) - Handoff

### Metadata
- as_of_commit: (worktree local)
- collab_mode: off

### Estado real verificado
- Gate 4.3 pasó de "arranque" a ejecución efectiva en `gate43_20260214_1000`.
- `D0` quedó cerrado en 5ep (best `S=60.2%`, e3).
- `D4` quedó cerrado en 5ep (best `S=63.6%`, e5), con mejora `+3.4pp` vs `D0`.
- `A4` completó e1-e3 (`S=35.4% -> 51.2% -> 61.0%`) y continúa e4-e5.
- `A7`, `D4+A4`, `D4+A7` permanecen en cola de ejecución, pero con ajuste de orden acordado.

### Ultima decision valida
- Mantener evaluación canónica por cada epoch (criterio científico, sin reducción de frecuencia).
- Al terminar `A4`, cortar el loop actual y relanzar desde `A7` con orden:
  `A7 -> A4x -> A7x -> D4+A4 -> D4+A7`.

### Proximo paso unico recomendado
- Terminar Gate 4.3 en secuencia (`A4` cierre -> `A7` -> `A4x` -> `A7x` -> `D4+A4` -> `D4+A7`) y consolidar tabla comparativa final para decisión Gate 4.4.

### Bloqueantes / riesgos
- Conclusiones tempranas sobre `A4` antes de e5 pueden sesgar la lectura (recovery no lineal).
- Si no se corta el script tras `A4`, el orden viejo (`A7 -> duales`) rompería la comparación directa `concat vs cross` antes de duales.

### Evidencia y archivos clave
- `data/bias_control_medium/training_outputs/gate43/gate43_20260214_1000/`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/07_GATE_4_3_RATIO_RE_CENTRICO/INFORME_GATE_4_3_RATIO_RE_CENTRICO.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/00_TRONCAL/Proyecto_Estado_Actual.md`

## 2026-02-14 06:40 (UTC) - Handoff

### Metadata
- as_of_commit: (worktree local)
- collab_mode: off

### Estado real verificado
- Gate 4.2 queda cerrado con run `D4` extendido a 8 epocas.
- Mejor punto del run: `epoch7` con `S=64.2%`, `A2M=65.0%`, `M2A=64.2%`, `hard_neg=91.6%`.
- `D4 8ep` confirma el techo observado en `D4 3ep` (`S=64.2%`) y mejora robustez en `hard_neg`.
- Roadmap operativo pasa a Gate 4.3 (bloque causal bifurcado) con etapa piloto previa.

### Ultima decision valida
- Gate 4.2 cerrado formalmente; no se requieren mas extensiones para `D4` dentro de esta fase.
- Gate 4.3 se inicia con pilotos (`a4`, `a7`, `d4a4`, `d4a7`) antes del barrido 5ep de los 6 brazos.

### Proximo paso unico recomendado
- Ejecutar pilotos 1 epoca/100 batches para `a4`, `a7`, `d4a4`, `d4a7`; si pasan checks de estabilidad, lanzar barrido Gate 4.3 completo (5ep fresh por brazo).

### Bloqueantes / riesgos
- Saltar pilotos puede ocultar problemas de VRAM/NaN/evaluacion en descriptores de audio.
- Mezclar runs resume/fresh en comparacion factorial invalida inferencia causal.

### Evidencia y archivos clave
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/06_GATE_4_2_RATIO_CENTRICO/resultados_gate_4.2.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/06_GATE_4_2_RATIO_CENTRICO/decisiones_gate_4.2.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/07_GATE_4_3_RATIO_RE_CENTRICO/plan_gate_4.3.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`

## 2026-02-14 02:15 (UTC) - Handoff

### Metadata
- as_of_commit: (worktree local)
- collab_mode: off

### Estado real verificado
- Bloque A v1.1 permanece cerrado con foundation lock en `data/bias_control_medium/training_outputs/foundation_locked_e25.pt`.
- Gate 4.2 queda en cierre con extension `D4` a 8 epocas (no se reetiqueta como Gate 4.3).
- Roadmap actualizado con bifurcacion formal:
  - linea MIDI temperada,
  - linea Audio armonia natural,
  - linea Dual.
- Estructura documental creada para:
  - `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/07_GATE_4_3_RATIO_RE_CENTRICO/`
  - `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/08_GATE_4_4_BIFURCACION_RATIO/`

### Ultima decision valida
- Gate 4.3 se ejecuta como bloque causal corto (`D0`, `D4-only`, `A4-only`, `A7-only`, `D4+A4`, `D4+A7`), todo fresh.
- Gate 4.4 absorbe el barrido amplio (MIDI: `D3/D8/D9/D10/D2/D5/D6/D7`; Audio: `A1/A2/A3/A5/A6`).

### Proximo paso unico recomendado
- Finalizar Gate 4.2 (`D4` 8 ep), verificar persistencia de mejora y abrir ejecucion de Gate 4.3.

### Bloqueantes / riesgos
- Mezclar resultados reanudados (`--resume`) con fresh en comparativas factoriales puede sesgar conclusion.
- Perder separacion de paradigma (MIDI temperado vs audio no temperado) invalida la lectura cientifica del nuevo diseño.

### Evidencia y archivos clave
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/07_GATE_4_3_RATIO_RE_CENTRICO/INFORME_GATE_4_3_RATIO_RE_CENTRICO.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/07_GATE_4_3_RATIO_RE_CENTRICO/plan_gate_4.3.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/08_GATE_4_4_BIFURCACION_RATIO/plan_gate_4.4.md`

## 2026-02-13 16:50 (UTC) - Handoff

### Metadata
- as_of_commit: ae40717
- collab_mode: off

### Estado real verificado
- `Run D-02` cerró 30 épocas; mejor single-seed en `epoch25` (`S=61.8%`, `A2M=61.8%`, `M2A=62.4%`, `hard_neg=90.4%`) y empate de `S` con `epoch26`.
- Re-evaluación multi-seed (`42/123/456/789`) entre `e25` y `e26` completada; se prioriza `e25` por estabilidad operativa.
- Foundation lock formal definido en `data/bias_control_medium/training_outputs/foundation_locked_e25.pt`.
- `explore_foundation.py` ejecutado con checkpoint bloqueado y artefactos guardados en `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/resultados_compartir/`.

### Ultima decision valida
- Bloque A v1.1 queda cerrado y la etapa activa pasa a screening Gate 4.2 sobre foundation bloqueado.
- `Gate2R-lite` se mantiene en backlog post Gate 4.2 (higiene metodológica, no bloqueante).

### Proximo paso unico recomendado
- Iniciar screening canónico de Gate 4.2 (D0 -> D1/D4) usando `foundation_locked_e25.pt`.

### Bloqueantes / riesgos
- Usar checkpoints mutables (`best_model_base.pt`) para decisiones de Gate 4.2 rompe trazabilidad.
- Desviarse del protocolo canónico (`pool=256`, `queries=500`, `seed=42`) invalida comparabilidad causal entre descriptores.

### Evidencia y archivos clave
- `data/bias_control_medium/training_outputs/bloqueA_runD-02/final_results.json`
- `data/bias_control_medium/training_outputs/bloqueA_runD-02/multiseed_reeval.json`
- `data/bias_control_medium/training_outputs/foundation_locked_e25.pt`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/resultados_compartir/explore_summary.json`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/05_PLAN_POST_DIAGNOSTICO_BLOQUE_A/Informe_final_bloqueA_Claude.md`

## 2026-02-12 15:55 (UTC) - Handoff

### Metadata
- as_of_commit: 4417542
- collab_mode: off

### Estado real verificado
- `Run D-02` sigue activo y alcanzó nuevo best parcial en `epoch18`: `S=59.6%`, `A2M=60.8%`, `M2A=59.6%`, `hard_neg=91.0%`.
- Foundation provisional se mantiene en `Run D ep5` hasta cierre formal de `Run D-02`.
- Gate 4.2 mantiene codigo implementado en paralelo (`gate42_training.py`, `ratio_descriptors.py`, ajustes de dataset/preflight) y screening bloqueado hasta foundation lock definitivo.
- Visualizaciones 3D de arquitectura publicadas y operativas en `https://altermundi.github.io/Phideus/`.

### Ultima decision valida
- Mantener secuencia cientifica en serie: primero cierre/lock de foundation (`C5 vs D5 vs D-02(best)`), luego screening Gate 4.2.
- La exploracion cualitativa (`explore_foundation.py`) se ejecuta solo con checkpoint inmutable post-lock.

### Proximo paso unico recomendado
- Cerrar `Run D-02`, consolidar tabla canonica C/D/D-02 y fijar foundation lock definitivo.

### Bloqueantes / riesgos
- Ejecutar screening Gate 4.2 o exploracion final sobre `best_model_base.pt` mutable rompe trazabilidad/reproducibilidad.
- Promover `D-02` antes de cierre completo por pico parcial puede sesgar decision.

### Evidencia y archivos clave
- `data/bias_control_medium/training_outputs/bloqueA_runD-02/eval_per_epoch/eval_epoch18.json`
- `data/bias_control_medium/training_outputs/bloqueA_runD-02/training.log`
- `data/bias_control_medium/training_outputs/bloqueA_runD/eval_per_epoch/eval_epoch5.json`
- `experiments/bias_control/explore_foundation.py`
- `README.md`

## 2026-02-12 08:20 (UTC) - Handoff

### Metadata
- as_of_commit: fe64b6c
- collab_mode: off

### Estado real verificado
- `Run D-02` activo en `data/bias_control_medium/training_outputs/bloqueA_runD-02` (modo `run-d`, 30 epocas, base `gate2/checkpoint_epoch45.pt`).
- `Run D ep5` se mantiene como foundation provisional hasta cierre de `Run D-02`.
- Screening de Gate 4.2 sigue bloqueado hasta foundation lock definitivo.

### Ultima decision valida
- Foundation lock final queda diferido a comparativa robusta `C5 vs D5 vs D-02(best)`.
- Paralelizacion permitida se mantiene: codigo Gate 4.2 en paralelo, decision cientifica en serie.

### Proximo paso unico recomendado
- Cerrar `Run D-02`, consolidar tabla canonica C/D/D-02 y fijar foundation lock definitivo.

### Bloqueantes / riesgos
- Iniciar screening Gate 4.2 antes del lock final invalida comparabilidad causal `D0 vs Dx`.
- Si `D-02` no supera/empata robustamente, no debe desplazar foundation provisional por inercia de corrida larga.

### Evidencia y archivos clave
- `data/bias_control_medium/training_outputs/bloqueA_runD-02/config.json`
- `data/bias_control_medium/training_outputs/bloqueA_runD-02/training.log`
- `data/bias_control_medium/training_outputs/bloqueA_runD/eval_per_epoch/eval_epoch5.json`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/00_TRONCAL/Proyecto_Estado_Actual.md`

---

## 2026-02-12 07:46 (UTC) - Handoff

### Metadata
- as_of_commit: e57e2fc
- collab_mode: off

### Estado real verificado
- Run D cerrado en epoch 5 con metricas canonicas: `S=51.0%`, `A2M=51.0%`, `M2A=51.8%`, `hard_neg=89.2%`.
- Tabla A/B/C/D consolidada en single-seed: D > C > B > A.
- Gate 4.2 mantiene implementacion de codigo lista para continuar, pero screening sigue bloqueado hasta foundation lock definitivo.

### Ultima decision valida
- Se mantiene la secuencia acordada: foundation lock definitivo antes de screening Gate 4.2.
- Estado de lock: `Run D ep5` queda como foundation provisional; cierre final pendiente de desempate robusto `C5 vs D5`.

### Proximo paso unico recomendado
- Ejecutar desempate robusto `C5 vs D5` (reevaluacion multi-seed) y cerrar foundation lock definitivo.

### Bloqueantes / riesgos
- Iniciar screening Gate 4.2 sin lock definitivo rompe comparabilidad causal `D0 vs Dx`.
- Diferencia single-seed `D-C` en `S` es positiva pero acotada (`+1.6pp`), por lo que conviene cierre robusto antes de promover.

### Evidencia y archivos clave
- `data/bias_control_medium/training_outputs/bloqueA_runD/final_results.json`
- `data/bias_control_medium/training_outputs/bloqueA_runD/eval_per_epoch/eval_epoch5.json`
- `data/bias_control_medium/training_outputs/bloqueA_runC/final_results.json`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/00_TRONCAL/Proyecto_Estado_Actual.md`

---

## 2026-02-12 03:16 (UTC) - Handoff

### Metadata
- as_of_commit: b50e446
- collab_mode: off

### Estado real verificado
- Run B cerrado con mejor epoch 3 (`S=43.2%`, `A2M=43.2%`, `M2A=43.4%`, `hard_neg=85.2%`).
- Run C en ejecucion, con evaluacion cerrada al menos hasta epoch 2 (`S=35.0%`, `hard_neg=79.6%`) y checkpoint epoch 3 guardado.
- Gate 4.2 sigue sin screening activo; implementacion de codigo aun pendiente.

### Ultima decision valida
- Secuencia acordada: cerrar Run C -> comparativa A/B/C -> Run D condicional (DEC-007) -> foundation lock definitivo -> screening Gate 4.2.
- Gate 4.2 codigo puede avanzar en paralelo; Gate2R-lite queda en backlog post Gate 4.2 (higiene, no bloqueante).

### Proximo paso unico recomendado
- Cerrar Run C y resolver foundation lock A/B/C(/D) antes de habilitar screening de Gate 4.2.

### Bloqueantes / riesgos
- Si se corre screening Gate 4.2 sin foundation lock, se rompe comparabilidad causal.
- `experiments/bias_control/bloqueA_training.py` mantiene cambios locales sin commit y requiere auditoria antes de relanzes.

### Evidencia y archivos clave
- `data/bias_control_medium/training_outputs/bloqueA_runB/eval_per_epoch/eval_epoch3.json`
- `data/bias_control_medium/training_outputs/bloqueA_runC/eval_per_epoch/eval_epoch2.json`
- `data/bias_control_medium/training_outputs/bloqueA_runC_log.txt`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/00_TRONCAL/Proyecto_Estado_Actual.md`

---

## 2026-02-12 00:00 (UTC) - Handoff inicial

### Metadata
- as_of_commit: 593a11d
- collab_mode: off

### Estado real verificado
- Bloque A v1.1 activo en BIAS_CONTROL, con S0 y Run A cerrados, Run B cerrado y Run C en curso (evaluado al menos hasta epoch 2).
- El plan vigente de Gate 4.2 ratio-centrico esta consolidado en:
  - `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/06_GATE_4_2_RATIO_CENTRICO/plan_gate_4.2.md`
- El repositorio usa `COLLAB OFF` por defecto salvo activacion explicita del usuario.

### Ultima decision valida
- Gate 4.2 mantiene protocolo canonico y guardrails anti-variable-fantasma; su implementacion de codigo puede correr en paralelo a decisiones de foundation lock segun DEC-007 (sin bloquear trabajo no-GPU).

### Proximo paso unico recomendado
- Cerrar Run C y formalizar foundation lock A/B/C; en paralelo, mantener habilitada implementacion de Gate 4.2 sin ejecutar screening hasta tener foundation definitivo.

### Bloqueantes / riesgos
- Desalineacion temporal entre estado documental troncal y estado experimental real.
- Cambios locales sin commit en scripts de entrenamiento pueden afectar reproducibilidad si no se auditan antes de relanzes.

### Evidencia y archivos clave
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`
- `Documents/00_TRONCAL/Proyecto_Estado_Actual.md`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/06_GATE_4_2_RATIO_CENTRICO/plan_gate_4.2.md`
- `data/bias_control_medium/training_outputs/bloqueA_runA/`
- `data/bias_control_medium/training_outputs/bloqueA_runB/`
- `data/bias_control_medium/training_outputs/bloqueA_runC/`
