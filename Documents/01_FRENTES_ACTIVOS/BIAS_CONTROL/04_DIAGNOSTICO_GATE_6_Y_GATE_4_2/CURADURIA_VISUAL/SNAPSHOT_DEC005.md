# SNAPSHOT DEC-005 — Decisiones Post-Diagnóstico

> Fecha: 2026-02-11
> Contexto: cierre de ejecución diagnóstica Gate 6 Phase 1 + Gate 4.2 Pre-Red (H4.2-6)
> Decisión anterior: DEC-004-A (cierre Gate 4.1)

---

## Resumen ejecutivo

DEC-005 fue un ciclo **diagnostic-only** (sin training) que ejecutó 5 scripts sobre los checkpoints existentes para responder dos preguntas:

1. **¿Por qué el fine-tuning degrada A2M?** → Porque el audio encoder está congelado. Solo el lado MIDI cambia.
2. **¿Se pueden extraer ratios armónicos desde audio real?** → No. AUC = 0.50 (chance level).

---

## Qué cayó

### H4.2-6: Dual-Domain Harmonic Ratios — ELIMINADA

**Hipótesis original:** Extraer ratios armónicos desde CQT del audio y compararlos con los del MIDI para obtener señal discriminativa cross-modal.

**Resultado:**

| Fase | AUC | delta_sim | Umbral GO | Veredicto |
|------|----:|----------:|----------:|-----------|
| P0 Oracle (MIDI sintetizado) | 0.559 | +0.034 | >= 0.80 | **NO-GO** |
| P1 Real (audio MAESTRO) | 0.502 | -0.004 | >= 0.70 | **NO-GO** |

**Por qué cayó:** El extractor CQT no puede separar las frecuencias fundamentales de los armónicos en audio polifónico de piano. Los histogramas de ratios resultan casi idénticos para pares alineados y random. Incluso bajo condiciones oracle (audio sintetizado limpio desde MIDI), el AUC apenas supera chance.

**Implicación:** No tiene sentido entrenar un modelo con ratios CQT como feature auxiliar. La señal no existe en la entrada.

**Fuente:** artefacto original `h426_prered_results.json` (no preservado hoy en `data/`; ver también la curaduría visual local y `INFORME_DEC005_DIAGNOSTICO_COMPLETO.md`).

---

## Qué descubrimos

### Hallazgo crítico: Audio encoder completamente congelado

El análisis de drift layer-by-layer reveló que los checkpoints `_base.pt` (RB0, RA5, R1) tienen **exactamente cero cambio** en:
- `audio_encoder.feature_extractor.*` (CNN) — 3.16M parámetros
- `audio_encoder.transformer.*` — 50.4M parámetros
- `audio_encoder.pos_embedding` — 6.14M parámetros

**Total: 59.7M parámetros del audio encoder no se mueven ni un epsilon.**

Solo cambian:
- `audio_projection.*` — 11-14% drift
- `midi_encoder.*` — 3-12% drift
- `midi_projection.*` — 11-13% drift

**Consecuencia directa:** El fine-tuning mueve el espacio MIDI pero el espacio audio queda fijo. Los puentes cross-modal se alargan (Gate2: 3.27 → R1: 4.68, +43%) y la separación cae (0.479 → 0.395, -18%).

**Fuente:** artefacto original `layer_drift.json` (no preservado hoy en `data/`; ver también la curaduría visual local y `INFORME_DEC005_DIAGNOSTICO_COMPLETO.md`).

### Hallazgo secundario: Degradación uniforme, no selectiva

Todos los checkpoints fine-tuned degradan de forma similar:
- La separación cae ~17% en promedio
- Los puentes se alargan ~40% en promedio
- El hubness no empeora dramáticamente (skewness similar)
- La correlación entre pérdida A2M y ganancia M2A por pieza es moderada (~0.4)

Esto sugiere que el problema no es de un run específico sino del **régimen de training** completo.

---

## Qué sobrevive

### H4.2-2: Adapter/Unfreezing controlado — PRIORIDAD #1

**Hipótesis:** Si desbloqueamos selectivamente capas del audio encoder (o insertamos adapters ligeros), el modelo puede adaptar ambos lados del espacio.

**Justificación post-DEC-005:** El hallazgo de audio encoder congelado hace esta hipótesis la candidata más directa. Si el problema es que audio no se mueve, la solución obvia es permitir que se mueva (de forma controlada para evitar catastrophic forgetting).

**Variantes a explorar:**
1. Unfreezing de las últimas N capas del audio transformer
2. LoRA/adapter modules en el audio transformer
3. Unfreezing gradual (curriculum)

### H4.2-1: Audio-only pre-training — PRIORIDAD #2

**Hipótesis:** Pre-entrenar el audio encoder en una tarea de auto-supervisión (masked prediction, contrastive) antes del fine-tuning cross-modal.

**Justificación:** Si el audio encoder está congelado, quizás necesita una fase de warmup que lo prepare para adaptarse al espacio cross-modal.

### S-Control: Control experimental obligatorio

Cualquier nuevo training debe incluir:
- **S-control:** Run de control sin la modificación (solo el régimen base) para confirmar que la mejora viene de la intervención, no del random seed o de más epochs.

---

## Hipótesis descartadas o en backlog

| Hipótesis | Estado | Razón |
|-----------|--------|-------|
| H4.2-6 (dual-domain ratios) | **ELIMINADA** | NO-GO en P0 y P1 |
| H4.2-4 (DANN revisitado) | **DESCARTADA** | Gate 3 ya cerró DANN; no hay evidencia nueva |
| H4.2-5 (curriculum learning) | **BACKLOG** | No prioritaria sin resolver el audio freeze |

---

## Tabla de decisión (DEC-005)

La matriz pre-definida en el plan DEC-005:

| Gate 6: drift asimétrico? | H4.2-6 P1: GO? | Siguiente paso |
|:-------------------------:|:---------------:|----------------|
| Sí | Sí | H4.2-6 training + H4.2-2 adapter + S-control |
| **Sí** | **No** | **H4.2-2 adapter + H4.2-1 audio-only + S-control** |
| No | Sí | H4.2-6 training + S-control |
| No | No | Solo S-control, re-evaluar rama 4.x |

**Resultado:** Fila 2 — Drift asimétrico confirmado (audio frozen), H4.2-6 NO-GO.

**Acción:** H4.2-2 adapter + H4.2-1 audio-only + S-control. Requiere DEC nueva (DEC-005 fue diagnostic-only).

---

## Próximos pasos concretos

1. **DEC-006:** Diseñar protocolo de training con adapter/unfreezing. Definir:
   - Cuántas capas desbloquear (o qué tipo de adapter)
   - Learning rate diferenciado (encoder vs projection)
   - Epochs, early stopping, métricas de monitoreo
   - S-control obligatorio

2. **Gate 4.2 training:** Ejecutar primer run con adapter. Comparar contra Gate 2 baseline.

3. **NO hacer:**
   - No repetir variantes de ratio auxiliary (ya cerrado)
   - No invertir en extracción CQT (ya NO-GO)
   - No correr más diagnósticos (DEC-005 dio evidencia suficiente)

---

## Referencias

- [INDEX_VISUAL.md](INDEX_VISUAL.md) — Navegación completa con todas las métricas y links
- [INFORME_DEC005_DIAGNOSTICO_COMPLETO.md](../INFORME_DEC005_DIAGNOSTICO_COMPLETO.md) — Informe detallado (1067 líneas)
- [INFORME_PLAN_CURADURIA_VISUAL_DEC005_CODEX.md](../INFORME_PLAN_CURADURIA_VISUAL_DEC005_CODEX.md) — Plan de Codex + revisión Claude

---

> Generado: 2026-02-11 | DEC-005 | Claude + Codex
