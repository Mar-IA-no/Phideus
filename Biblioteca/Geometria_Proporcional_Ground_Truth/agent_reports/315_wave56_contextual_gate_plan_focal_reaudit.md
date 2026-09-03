# R315 — reauditoría focal del plan de la Ola 56

> **Agente:** `01a0672b-90d1-7ca0-bb09-992b7ce7624d` (`Kepler`)
> **Fecha:** 2026-09-03
> **Veredicto:** `REVISE`
> **Preservación:** informe del subagente, verbatim.

## Findings

1. **ALTO: el shuffled conserva el marginal no ponderado, pero no el marginal efectivo del learner.** Cada token recibe igual peso total (`plan:29`), mientras el shuffle permuta targets por política entre tokens con distinto número de desacuerdos (`plan:203`). Por tanto, una fila pasa de peso `1/d_t` a `1/d_t'`: el multiconjunto permanece, pero cambia la distribución ponderada optimizada. En `dev_fit` primario de Ola 55, `d_t` varía entre `1` y `14`, confirmando que los pesos no son intercambiables. Debe permutarse dentro de estratos de igual peso efectivo o redefinir explícitamente weighting/null para preservar la medida ponderada.

2. **ALTO: Stage 0 aún deja decisiones capaces de cambiar la familia seleccionada.** No se fija si fit, scoring OOF, restricciones y elección en `dev_eval` operan sobre los `768` tokens in-catalog o sobre la población primaria; esa diferencia fue materialmente selector-sensitive en Ola 55 (`plan:23`, `plan:52`, `plan:59`). Tampoco se define clase de escalador, columnas escaladas ni tratamiento de columnas constantes (`plan:33`). Preservar luego el schema no sustituye congelarlo antes de ejecutar Stage 0.

3. **MEDIO: la resolución de degeneraciones no cubre `dev_eval`.** Los mínimos sólo protegen `dev_fit`, folds y logística (`plan:76`), aunque `dev_eval` decide la familia prospectiva (`plan:62`). Falta un conteo esperado/mínimo de población y desacuerdos, con `NOT_EVALUABLE` ante incumplimiento.

## Resoluciones R314

- `1 shuffled`: **incompleta** por el marginal ponderado.
- `2 Stage 0`: **incompleta** por población y escalado no congelados.
- `3 cronología/segunda auditoría`: **cerrada**; está explícita antes de claves y como máquina física de tres fases (`plan:86`).
- `4 selector_sensitive`: **cerrada** (`plan:177`).
- `5 token×política`: **cerrada**, con claim restringido y sensibilidad separada (`plan:143`).
- `6 degeneraciones/artefactos`: **parcial**, por `dev_eval`.

## Veredicto

`REVISE`. No se editaron archivos.
