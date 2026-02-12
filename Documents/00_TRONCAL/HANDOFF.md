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
  - `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/06_GATE_4_2_RATIO_CENTRICO/PLANES/plan_gate_4.2.md`
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
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/06_GATE_4_2_RATIO_CENTRICO/PLANES/plan_gate_4.2.md`
- `data/bias_control_medium/training_outputs/bloqueA_runA/`
- `data/bias_control_medium/training_outputs/bloqueA_runB/`
- `data/bias_control_medium/training_outputs/bloqueA_runC/`
