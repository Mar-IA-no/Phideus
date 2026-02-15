# Gate 4.3 - Ratio Re-Centrico (Bifurcado)

Gate 4.3 queda definido como bloque causal para responder cuatro preguntas:

1. Que aporta la inyeccion de ratios del lado MIDI (paradigma temperado).
2. Que aporta la inyeccion de ratios del lado Audio (paradigma de armonia natural).
3. Que mecanismo de inyeccion funciona mejor: concat vs cross-attention.
4. Si la inyeccion dual (MIDI+Audio) y/o cross-modal suma senal.

## Estructura

- `README.md`: alcance y estado de etapa.
- `INFORME_GATE_4_3_RATIO_RE_CENTRICO.md`: marco metodologico.
- `plan_gate_4.3.md`: plan operativo de ejecucion.

## Documento eje

- `INFORME_GATE_4_3_RATIO_RE_CENTRICO.md`
- `plan_gate_4.3.md`

## Nota de comparabilidad

Todos los brazos de Gate 4.3 se corren fresh desde `foundation_locked_e25.pt` para mantener comparabilidad estricta entre scheduler/LR.

## Roadmap por fases

### Fase 0 — COMPLETE

Baselines + concat. Output: `gate43_20260214_1000/`

| Arm | Mecanismo | Best S | Best ep | MRR_avg | hard_neg |
|-----|-----------|--------|---------|---------|----------|
| D0 | baseline | 60.2% | e3 | 0.280 | 90.6% |
| D4 | MIDI concat | 63.6% | e5 | 0.313 | 91.2% |
| A4 | Audio concat | 63.6% | e5 | 0.297 | 92.4% |

### Fase 1 — RUNNING (~9h, lanzada 2026-02-14 16:30 UTC)

Concat restante + cross-attention audio. Brazos: A7, A4x, A7x.

### Fase 2 — D4x (cross-attention MIDI)

Codigo implementado y verificado en CPU. Pendiente pilot GPU y run 5ep.
Completa la matriz mecanismo x descriptor:

|  | Concat | Cross-attention |
|---|--------|----------------|
| MIDI intervals (D4) | D4 (Fase 0) | D4x (Fase 2) |
| Audio log-freq (A4) | A4 (Fase 0) | A4x (Fase 1) |
| Audio attractor (A7) | A7 (Fase 1) | A7x (Fase 1) |

### Fase 3 — Duales same-modality

Con ganadores de Fases 0-2 (concat o cross-att por descriptor). 2 brazos.

### Fase 4 — Cross-modal injection

Inyectar descriptores de un dominio en el encoder del otro:

| Brazo | Audio encoder recibe | MIDI encoder recibe |
|-------|---------------------|---------------------|
| CM-a | — | Audio desc (A_best) |
| CM-m | MIDI desc (D_best) | — |
| CM-bi | MIDI desc (D_best) | Audio desc (A_best) |

## Todos los brazos definidos

| Brazo | Lado | Descriptor | Mecanismo | Params nuevos | Fase | Status |
|-------|------|-----------|-----------|---------------|------|--------|
| D0 | — | — | baseline | 0 | 0 | COMPLETE |
| D4 | MIDI | intervals (4d) | concat | ~267K | 0 | COMPLETE |
| A4 | Audio | log-freq deltas (8d) | concat | ~1.06M | 0 | COMPLETE |
| A7 | Audio | rational attractor (12d) | concat | ~1.06M | 1 | RUNNING |
| A4x | Audio | log-freq deltas (8d) | cross-attn | ~4.2M | 1 | RUNNING |
| A7x | Audio | rational attractor (12d) | cross-attn | ~4.2M | 1 | RUNNING |
| D4x | MIDI | intervals (4d) | cross-attn | ~1.05M | 2 | CPU VERIFIED |
| Dual1 | Ambos | ganadores | ganador | TBD | 3 | PENDING |
| Dual2 | Ambos | ganadores | ganador | TBD | 3 | PENDING |
| CM-a | Cross | audio->MIDI | ganador | TBD | 4 | CONCEPTO |
| CM-m | Cross | MIDI->audio | ganador | TBD | 4 | CONCEPTO |
| CM-bi | Cross | bidireccional | ganador | TBD | 4 | CONCEPTO |
