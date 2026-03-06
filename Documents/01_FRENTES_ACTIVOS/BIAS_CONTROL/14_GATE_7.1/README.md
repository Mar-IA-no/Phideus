# Gate 7.1 — MERT-330M Frozen Cross-Modal Probe

**Estado**: plan v2 formalizado, implementación pendiente.

## Propósito

Gate 7.1 no intenta repetir Gate 7. Su objetivo es responder una pregunta distinta: si el backbone de audio ya es fuerte, ¿la ventaja de `A4` sobrevive en retrieval cross-modal o desaparece?

La lectura correcta es prudente: sigue siendo un piloto decisional para Escalón 1, no un aislamiento causal puro. Cambian simultáneamente backbone, co-adaptación y régimen de pretraining.

## Diseño vigente

### Fase 7.1a — `D0` pilot

- `MERT-330M` congelado como encoder de audio.
- `MIDI encoder + projections` entrenados desde cero con VICReg.
- Objetivo: validar infraestructura, throughput y estabilidad metodológica antes de abrir cualquier claim sobre `A4`.

### Fase 7.1b — `a4r-mert`

- Solo se abre si `7.1a` confirma que el setup aprende y que el costo es razonable.
- No es un swap trivial de flag: requiere una variante nueva del mecanismo `a4r` adaptada a `MERTEncoder`.

## Guardrails

- No leer `Gate 7` como prueba de que MERT-330M “ya tiene A4” en sentido fuerte; el probe cerró solo sobre la envolvente espectral segment-level.
- No comparar `ΔA4` de `7.1b` como si fuera idéntico al `+5.5pp` multi-seed de Gate 5B.
- `1` seed alcanza para pilot y go/no-go, no para claim estadístico fuerte.
- El primer paso técnico obligatorio es corregir el `model.train()` leak y forzar carga explícita de MERT antes de los anti-ghost checks.

## Documento de trabajo

- `Plan_implementacion.md`: plan v2 completo con fases, riesgos, checks y criterios de lectura.
