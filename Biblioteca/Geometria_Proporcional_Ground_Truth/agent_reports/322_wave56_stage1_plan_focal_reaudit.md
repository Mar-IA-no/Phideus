# R322 — Reauditoría focal del plan prospectivo Stage 1 de Ola 56

> **Dictamen:** `REVISE`
> **Alcance:** auditoría estática, independiente y de sólo lectura.

## Findings

### Alto 1 — escrow sin barrera de durabilidad completa

Faltaban `fsync` del directorio después del rename, publicación atómica del
freeze público y reapertura verificable de ambos artefactos antes del
generador. El escrow puede seguir siendo la autoridad de recovery, pero toda la
secuencia debe ser bloqueante pre-draw.

### Alto 2 — contratos incompatibles para recovery post-label

El plan padre permitía un delta de código con las mismas claves, mientras el
plan de implementación exigía el mismo HEAD e invalidaba el draw si era
necesario cambiar código. Debe prevalecer esta última regla: después de abrir
labels sólo se reanuda el ejecutable congelado.

### Medio 1 — schema prospectivo contradictorio

El plan padre enumeraba `disagreement` como feature Stage 1, creando dieciocho
columnas, aunque Stage 0 y su cierre fijan diecisiete. `disagreement` debe ser
máscara de población/override, no columna.

### Medio 2 — eje shuffled incompleto

Además de mappings y targets, los arrays deben preservar por `shuffle_id` los
scores, thresholds, acciones, overrides y métricas token×política de cada una
de las cinco réplicas antes de guardar su agregado.

## Estado de R321

La barrera OS, máquina transaccional, replay exacto, semántica de
`NOT_EVALUABLE` y enumeración de brazos quedaron resueltos. Persistían los
cuatro contratos anteriores; el plan no estaba aún listo para implementación.
