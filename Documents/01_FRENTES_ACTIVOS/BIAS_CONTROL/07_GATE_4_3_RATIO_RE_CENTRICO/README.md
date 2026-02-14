# Gate 4.3 - Ratio Re-Centrico (Bifurcado)

Gate 4.3 queda definido como bloque causal corto para responder tres preguntas:

1. Que aporta la inyeccion de ratios del lado MIDI (paradigma temperado).
2. Que aporta la inyeccion de ratios del lado Audio (paradigma de armonia natural).
3. Si la inyeccion dual (MIDI+Audio) suma senal por encima de cada lado por separado.

## Estructura

- `PLANES/`: protocolo operativo de ejecucion (brazos, criterios y orden).
- `EVIDENCIAS/`: logs, tablas y snapshots de evaluacion por epoch.
- `RESULTADOS/`: consolidaciones comparativas entre brazos.
- `DECISIONES/`: cierres GO/NO-GO para paso a Gate 4.4.

## Documento eje

- `INFORME_GATE_4_3_RATIO_RE_CENTRICO.md`
- `PLANES/plan_gate_4.3.md`

## Nota de comparabilidad

Todos los brazos de Gate 4.3 se corren fresh desde `foundation_locked_e25.pt` para mantener comparabilidad estricta entre scheduler/LR.

Estado de arranque (2026-02-14, post cierre Gate 4.2):

- Precondicion cumplida: `D4 8ep` confirma mejora sostenida (`S best=64.2%`, `hard_neg=91.6%`).
- Siguiente paso operativo: pilotos `a4`, `a7`, `d4a4`, `d4a7` (1 epoca / 100 batches) antes del barrido 5ep.
