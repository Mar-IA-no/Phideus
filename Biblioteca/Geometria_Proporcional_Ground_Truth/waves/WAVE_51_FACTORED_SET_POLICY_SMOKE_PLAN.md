# Ola 51 — plan de smoke CPU para lector factorizado de conjunto y elección

> **Estado:** `FROZEN-DEVELOPMENT-SMOKE / OPENED-HISTORICAL-DATA / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-03
> **Antecedente:** `WAVE_50_PROSPECTIVE_CLOSED.md`

## Pregunta

La Ola 50 mostró que una salida sigmoid conserva mejor las familias todavía
compatibles que un softmax exclusivo, pero no cumplió simultáneamente el margen
preregistrado de top-1 y pagó un exceso leve de ancho. Este smoke pregunta si
esa tensión puede separarse arquitectónicamente: una cabeza representa el
conjunto compatible y otra elige un miembro sin reescribir ese conjunto.

El benchmark no contiene una utilidad o preferencia verdadera entre miembros
compatibles. Por eso la segunda salida se denomina **cabeza de elección bajo
supervisión parcial**, no política semántica. Sólo se evalúa si el desacople de
objetivos es mecánicamente útil; no si la elección expresa una acción correcta
en el mundo.

## Régimen de evidencia

Se reutilizan únicamente `train` y `val` ya abiertos de la Ola 50. El smoke no
lee el lockbox, no produce evidencia confirmatoria y no puede promover una
arquitectura. Si el mecanismo supera sus controles, una ola posterior deberá
usar paquete y lockbox frescos, con protocolo prospectivo propio.

## Arquitectura matched

Todos los brazos usan exactamente el mismo `DualHeadDeepSet`:

- encoder DeepSets común: MLP por punto, pooling `mean + max` y MLP de set;
- `set_head`: cuatro logits de compatibilidad independiente;
- `choice_head`: cuatro logits para elegir un miembro compatible;
- mismo estado inicial por seed, mismos batches, mismo orden de puntos, mismo
  número de parámetros y mismo total de pasos de optimización.

La cabeza no usada por un brazo permanece presente para igualar parámetros.
Los cuatro brazos son:

1. `softmax_only`: entrena encoder + `choice_head` con partial-label loss;
2. `sigmoid_only`: entrena encoder + `set_head` con BCE multi-label;
3. `joint_multitask`: entrena ambas cabezas y el encoder con
   `BCE(set) + partial_label(choice)`;
4. `factored_frozen`: fase A idéntica a `sigmoid_only`; después congela encoder
   y `set_head` y entrena sólo `choice_head` con partial-label loss.

Presupuesto primario: `50` épocas de fase A + `10` de fase B. Los controles
`softmax_only`, `sigmoid_only` y `joint_multitask` reciben `60` épocas sobre su
objetivo para igualar cantidad de batches/backprop. No hay early stopping ni
selección de checkpoint.

## Lectura

La salida de conjunto de `factored_frozen` es la congelada al final de la fase
A. Debe ser array-exact con `sigmoid_only@50`; `sigmoid_only@60` queda además
como control de presupuesto. La salida de elección no cambia el threshold ni
los miembros del conjunto. Para `softmax_only`, las métricas de conjunto se
derivan de `softmax(choice_head)`; para los otros brazos, de
`sigmoid(set_head)`. Se reportan:

- set recall, complete coverage, incompatibilidad, ancho y exact-set desde
  `set_head`;
- top-1 compatible de `choice_head` sin gate;
- top-1 compatible restringido al conjunto predicho; si el conjunto está
  vacío, fallback al argmax global de `choice_head`;
- macro-AUC/AP de membresía como diagnóstico threshold-free.

Cada cabeza de conjunto selecciona su threshold sólo en `val_threshold`; todas
las métricas se leen en el `val_monitor` disjunto. El sistema primario promedia
logits crudos sobre seeds antes de calcular métricas.

## Criterio diagnóstico predeclarado

El mecanismo queda **prometedor para un protocolo prospectivo posterior** sólo
si, en `NEAR_RIVAL` del monitor:

1. `factored_frozen` es array-exact con `sigmoid_only@50` en logits de conjunto,
   y queda dentro de `0.01` de `sigmoid_only@60` en set recall sin aumentar
   ancho ni incompatibilidad más de `0.02`;
2. mejora top-1 compatible restringido al conjunto en al menos `0.02` frente al
   argmax de `sigmoid_only`;
3. no queda por debajo de `softmax_only` en top-1 por más de `0.01`;
4. `joint_multitask` permite distinguir si el beneficio requiere separación
   temporal o sólo dos pérdidas simultáneas.

Estos umbrales organizan el smoke y no son un GO científico. Si fallan, se
conserva el resultado negativo y no se escala por inercia.

## Controles e invariantes

- tres seeds de desarrollo, con inicialización y batches compartidos;
- permutación de puntos: delta absoluto de logits dentro de la tolerancia de
  Ola 50;
- parámetros idénticos entre brazos;
- control `factored_frozen_shuffled_choice` sobre subconjunto matched: fase A
  conserva targets verdaderos y sólo la fase B recibe un derangement de la
  supervisión de elección;
- encoder y `set_head` de `factored_frozen` byte-identical antes y después de
  fase B;
- ningún path de lockbox en el inventario de lecturas;
- checkpoints `last_epoch`, logits crudos, normalizador, split manifest, config,
  métricas por token y runtime preservados.

## Alcance y próximo discriminante

Un positivo sólo autoriza diseñar el contraste prospectivo fresh-package. No
demuestra una política útil, una geometría natural ni superioridad sobre EIV.
Para eso la cabeza de decisión deberá recibir una autoridad adicional
explícita: utilidad, costo, contexto o nueva medición. El GO/NO-GO y cualquier
promoción permanecen bajo decisión del usuario.
