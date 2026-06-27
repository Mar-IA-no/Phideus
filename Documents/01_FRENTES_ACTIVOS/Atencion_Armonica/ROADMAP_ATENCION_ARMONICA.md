# Roadmap — Atención Armónica

> Documento estructural del frente. Resume la pregunta científica, el estado metodológico actual y la secuencia de fases prevista sin confundir incubación local con propagación canónica al troncal.

## §1 Identidad del frente

**Qué es.** Un frente piloto que pregunta si la operación central del patrón AlphaFold-like en este dominio no está en un transformer genérico con features armónicas inyectadas, sino en una representación explícita de pares cuya actualización triangular propaga consistencia global sobre pertenencia a una misma fuente armónica.

**Qué no es.** No es una validación general de “armonía natural” en audio. No es todavía un frente troncal del programa. No es un reemplazo inmediato del modo descriptor-guided vigente en Phideus. Es un experimento de falsación barato y metodológicamente exigente para decidir si vale la pena escalar un programa arquitectónico nuevo.

## §2 Origen del frente

El frente nace de una conversación sobre AlphaFold y de una crítica precisa a una traslación demasiado rápida de su intuición. La versión trivial de “triangle update sobre diferencias en log-frecuencia” no imponía ninguna restricción real: en `R`, las diferencias de logs son identidades algebraicas gratis. La no trivialidad relevante apareció en otro lugar: la **transitividad de la pertenencia a una misma fuente/fundamental** cuando el pairwise local es ambiguo.

La analogía útil quedó así:

- en AlphaFold: representación de pares + restricción global no trivial sobre geometría;
- aquí: relación `same-source` entre parciales + consistencia global de una partición.

## §3 Estado actual

### Estado experimental

- `pairformer.py`, `harness.py`, `1_train_grouping.py` y `1_report.py` ya fueron escritos y auditados por capas.
- Los 6 modelos del frente quedaron definidos y verificados por shapes, simetría y contraste.
- El cuello ya no está en la arquitectura ni en el training loop, sino en el **diseño del dataset**.

### Estado metodológico

El frente ya atravesó tres iteraciones de diseño:

1. `v1`: parciales exactos armónicos. Falló por feature-triviality (`ratio_residual` y `common_f0_residual` casi oraculares).
2. `v2`: inarmonicidad `beta>0` + dropout. Falló por leak de amplitud (`amp=1/n`).
3. `v2.1`: inarmonicidad + amplitud randomizada + gate explícito antes de GPU. Es la versión vigente.

### Estado documental

La carpeta local del frente ya preserva:

- `PLAN_FASE_0_v1_superseded.md` como registro del primer diseño.
- este roadmap y el plan `v2.1` como estado metodológico actual.

No hay propagación al troncal hasta resultado real.

## §4 Fase 0: experimento decisivo

La `Fase 0` no busca construir la arquitectura final del programa. Busca contestar una sola pregunta: si una representación persistente de pares con propagación triangular aporta sobre un baseline token-only cuando ambos reciben la misma evidencia armónica local.

### Tarea

Agrupamiento armónico en mezclas polifónicas sintéticas: dado un conjunto de picos/parciales, predecir la matriz `N x N` de relación de equivalencia `same-source`.

### Modelos congelados

- `A-naive`
- `A-rich`
- `B`
- `B-local`
- `B-minus`
- `B-shuffle`

### Contrastes

- primario: `B vs A-rich`
- transitividad pura: `B vs B-local`
- secundario: `B vs B-minus`
- lateral: `A-rich vs A-naive`

## §5 Gate de validez del dataset

El aprendizaje principal del frente es que el dataset debe ser auditado antes del training. El gate vigente exige dos cosas:

### Headroom

Ninguna feature cerrada ni probe per-par debe resolver la tarea al techo. Para eso se miden:

- single-feature AUC,
- `LogReg`,
- `PairMLP` con el mismo tipo de inputs que recibe `A-rich`.

### Solvabilidad

La tarea no debe volverse imposible al romper los leaks. Por eso se exige además un `oracle_privileged_upper_bound` que muestre recuperabilidad global mínima.

Si una combo del sweep no cumple ambas condiciones, no va a GPU.

## §6 v2.1 vigente

La formulación actual del generador trabaja con:

- `beta>0` per-source,
- amplitud randomizada per-source,
- dropout de parciales,
- `calibration_pool` y `final_pool` separados,
- desempate del sweep por peor celda decisiva.

Su objetivo no es optimizar performance todavía, sino conseguir un problema donde:

1. `A-rich` no llegue al techo por evidencia per-par cerrada;
2. la estructura siga siendo recuperable globalmente;
3. el contraste `B vs A-rich` vuelva a tener significado causal.

## §7 Fases siguientes

Solo si `Fase 0` entrega un resultado interpretable tiene sentido abrir fases posteriores:

- `Fase 1`: picos detectados (`CQT`) en lugar de parciales exactos.
- `Fase 2`: mezclas con estructura temporal/onsets.
- `Fase 3`: integración con un trunk audio real y eventual backbone foundation.

Esas fases hoy no están habilitadas. La condición previa sigue siendo la misma: que `Fase 0` genere un contraste con headroom real.
