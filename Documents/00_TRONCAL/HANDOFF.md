# HANDOFF Operativo - Proyecto Phideus

---

## Proposito

Este documento es el puente de continuidad entre sesiones, agentes e instancias.
No reemplaza roadmaps ni decisiones formales: sintetiza estado operativo real y proximo paso ejecutable.

---

## Precedencia de fuentes

1. `COLLAB/STATUS.md`: snapshot operativo "ahora" del ciclo collab.
2. `COLLAB/DECISIONS.md`: decisiones formales vigentes del protocolo.
3. `Documents/00_TRONCAL/HANDOFF.md` (este archivo): continuidad entre sesiones/instancias.
4. `COLLAB/HANDOFFS.md`: historial de traspasos entre agentes.

Regla: si hay conflicto, prevalecen `STATUS.md` y `DECISIONS.md` sobre este documento.

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
