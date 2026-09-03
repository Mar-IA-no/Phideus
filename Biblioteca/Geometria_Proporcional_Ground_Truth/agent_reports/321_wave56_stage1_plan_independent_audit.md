# R321 — Auditoría independiente del plan prospectivo Stage 1 de Ola 56

> **Dictamen:** `REVISE`
> **Alcance:** auditoría estática, de sólo lectura y con contexto mínimo.
> **Resultado:** seis findings materiales: tres altos y tres medios.

## Findings

### Alto 1 — la separación temporal no era una barrera física

El generador escribe la verdad de `train`, `val` y `lockbox` bajo el mismo árbol
antes de inferencia. Comprobar que no existen labels o bundles futuros detecta
materializaciones, pero no impide que un proceso abra directamente la verdad
sellada. Se requiere un materializador split-scoped separado, permisos o
aislamiento equivalente, receipts y tests negativos de acceso.

### Alto 2 — faltaba freeze durable y escrow atómico antes del draw

El plan escribía `preparation_freeze.json` después de inferencia. Un crash
durante generación podía dejar una parte de las claves sin un binding durable a
commit, config y fuentes. Se requiere persistir atómicamente las tres claves y
su freeze completo antes de invocar el generador.

### Alto 3 — idempotencia y recovery dejaban flexibilidad post-hoc

No se definían estados recuperables entre apertura de oracle, construcción de
bundle, fit y freeze. Permitir un delta de código que sólo conservara el
estimando podía alterar estimador, selector o bordes después de observar
labels. Se requiere una máquina de estados transaccional, validación de hashes y
ejecutable congelado; cualquier excepción debe pasar auditoría independiente
sin uso selectivo de verdad abierta.

### Medio 1 — replay y RNG incompletamente especificados

No se distinguían artefactos exactos de runtime/paths ni se congelaban seed,
generador y orden de tokens para bootstrap. Se requiere una matriz explícita de
comparación y RNG completo.

### Medio 2 — semántica incorrecta de `NOT_EVALUABLE` shuffled

Una fracción movible insuficiente debe invalidar sólo el control shuffled y la
condición prospectiva que depende de él, no toda la fase.

### Medio 3 — brazos y arrays no estaban enumerados

El plan debía nombrar los siete brazos, incluidos `pure_joint_full` y
`oracle_positive_gain`, y fijar el contenido mínimo de `result_arrays.npz` por
split y brazo.

## Verificaciones sin finding

`ridge_contextual(alpha=1.0)` y `ridge_advantage_only(alpha=100.0)` coinciden
con Stage 0. La grilla de cuantiles, los cinco shuffles estratificados y la
selección separada en `gate_select` son consistentes con el plan padre. El
generador soporta `train`, `val` y `lockbox`, y el oracle materialización por
split; la arquitectura es implementable una vez resueltos los contratos
anteriores.
