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
- El cuello ya no está en la arquitectura, en el training loop ni en el diseño base del dataset: el `final_pool` `v2.1` quedó congelado con gate `PASS` y la `Fase 0` cerró `54/54`.
- La atribución principal ya quedó leída: el pair-state es el salto grande; el `triangle` no domina `IID`, pero sí mejora `OOD-poly` en `AUC/AP` frente a `B-local`, con calibración `ARI@τ_val` todavía pendiente.

### Estado metodológico

El frente ya atravesó tres iteraciones de diseño:

1. `v1`: parciales exactos armónicos. Falló por feature-triviality (`ratio_residual` y `common_f0_residual` casi oraculares).
2. `v2`: inarmonicidad `beta>0` + dropout. Falló por leak de amplitud (`amp=1/n`).
3. `v2.1`: inarmonicidad + amplitud randomizada + gate explícito antes de GPU. Es la versión vigente y ya pasó sweep + gate final.

La combo congelada de `v2.1` quedó fijada en:

- `beta-center = 1e-3`
- `alpha-range = [0.5, 1.5]`
- `sigma_amp = 0.5`
- `p_drop = 0.3`

El `final_pool` asociado pasó el gate con `PairMLP` en la banda `0.81-0.84` y `oracle_priv = 1.0` en las celdas decisivas.

### Estado documental

La carpeta local del frente ya preserva:

- `PLAN_FASE_0_v1_superseded.md` como registro del primer diseño.
- este roadmap y el plan `v2.1` como estado metodológico actual.

La propagación al troncal queda permitida solo en forma acotada: `Fase 0` cerrada, `GO` condicionado y caveat de calibración explícito.

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

### Orden de lectura congelado

El orden interpretativo del frente quedó congelado así y ya fue aplicado al cierre:

1. `B vs B-local`
2. `B-shuffle`
3. `B vs A-rich`
4. `B vs B-minus`

La razón es metodológica. `B > A-rich` solo prueba que la maquinaria de pares completa aporta sobre un baseline token-only fuerte. No identifica todavía si el aporte específico viene de la transitividad triangular o de otro aspecto del pair-state.

## §5 Gate de validez del dataset

El aprendizaje principal del frente es que el dataset debe ser auditado antes del training. El gate vigente exige dos cosas:

### Headroom

Ninguna feature cerrada ni probe per-par debe resolver la tarea al techo. Para eso se miden:

- single-feature AUC,
- `LogReg`,
- `PairMLP` con el mismo tipo de inputs que recibe `A-rich`.

### Solvabilidad

La tarea no debe volverse imposible al romper los leaks. Por eso se exige además un `oracle_privileged_upper_bound` que muestre recuperabilidad global mínima.

Si una combo del sweep no cumple ambas condiciones, no va a GPU. En `v2.1`, esa discusión ya quedó cerrada: el `final_pool` pasó.

### Caveat estructural que queda después del gate

El gate dejó un caveat explícito y persistente: `oracle_unpriv_f0only` colapsa a valores muy bajos. Eso no invalida el frente, pero sí delimita con precisión lo que se puede inferir:

- el gate certifica que el dataset **no es feature-trivial**;
- el gate no certifica por sí solo que una aproximación simple no-supervisada pueda resolverlo;
- por eso el smoke supervisado de `A-rich` y los contrastes del training completo siguen siendo necesarios.

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

Ese objetivo ya se cumplió en el sentido mínimo requerido por el frente:

- el sweep mostró `headroom`;
- el `final_pool` pasó el gate;
- el smoke de `A-rich` mostró aprendibilidad supervisada;
- el training real ya empezó a producir una separación parcial `B > A-rich`.

Lo que queda abierto ya no es la validez del dataset ni la atribución básica, sino la siguiente frontera: calibrar el paso de ranking de pares a clustering y validar el sesgo del `triangle` fuera del sintético.

## §7 Fases siguientes

Como `Fase 0` entregó un resultado interpretable, tiene sentido abrir fases posteriores con alcance acotado:

- `Fase 1`: picos detectados (`CQT`) en lugar de parciales exactos.
- `Fase 2`: mezclas con estructura temporal/onsets.
- `Fase 3`: integración con un trunk audio real y eventual backbone foundation.

Esas fases quedan habilitadas como **GO acotado**, no como escalado irrestricto. La condición siguiente es resolver calibración de `τ` y comprobar si la ventaja `OOD-poly` del `triangle` sobrevive cuando los picos dejan de ser parciales exactos.
