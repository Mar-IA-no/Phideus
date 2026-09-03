# Ola 51 — cierre del smoke de conjunto y elección factorizados

> **Estado:** `EXECUTED / REPLAYED / DEVELOPMENT-NEGATIVE / NO-GO-NOGO`
> **Fecha:** 2026-09-03
> **Código ejecutado:** `8d6058a6e04fcadafad7ed7b61209974926948ea`
> **Régimen:** CPU, train/val históricos ya abiertos, lockbox no leído

## Qué se puso a prueba

La Ola 50 había mostrado una tensión: una salida sigmoid retenía mejor las
familias compatibles, pero no resolvía al mismo tiempo la elección top-1 ni el
costo de ancho. La Ola 51 ensayó una separación arquitectónica explícita. Un
encoder DeepSets común alimentó una cabeza de pertenencia al conjunto y otra de
elección bajo supervisión parcial. El brazo principal aprendió primero el
conjunto durante 50 épocas, congeló encoder y cabeza de conjunto, y entrenó la
cabeza de elección durante 10 épocas adicionales.

La comparación incluyó softmax exclusivo, sigmoid, multitask simultáneo, un
brazo staged con encoder no congelado y controles true/shuffled sobre un
subconjunto matched. Todos los brazos principales compartieron arquitectura,
estado inicial, batches, parámetros (`13.384`) y cantidad de backprops. El
protocolo fue auditado antes de la corrida y quedó anclado por hash a los cinco
inputs train/val y a los manifiestos canónicos de Ola 50.

## Resultado diagnóstico

En los `384` pair tokens `NEAR_RIVAL` de `val_monitor`, el patrón
predeclarado fue falso:

| Salida | Set recall | Ancho | Any incompatible | Top-1 gated |
|---|---:|---:|---:|---:|
| `softmax_only` | 0.384 | 1.016 | 0.120 | **0.885** |
| `sigmoid_only@50` | 0.808 | 2.727 | 0.247 | 0.862 |
| `sigmoid_only@60` | **0.865** | 2.984 | 0.331 | 0.867 |
| `joint_multitask` | 0.755 | 2.570 | 0.203 | 0.875 |
| `staged_unfrozen` | 0.818 | 2.792 | 0.396 | 0.865 |
| `factored_frozen` | 0.808 | 2.727 | 0.247 | 0.867 |

El brazo factorizado preservó exactamente la salida set-valued de época 50,
como exigía el diseño, pero perdió `0.0577` de recall frente al sigmoid que usó
las 60 épocas para seguir aprendiendo el conjunto. Su top-1 gated no mejoró al
sigmoid (`0.867` en ambos) y quedó `0.0182` por debajo del softmax, más allá del
margen de no inferioridad `-0.01`.

Los comparadores de mecanismo tampoco sostuvieron la receta. `staged_unfrozen
− joint_multitask = -0.0104` y `factored_frozen − staged_unfrozen = +0.0026` en
top-1 gated. No apareció una contribución material ni del staging ni del
congelamiento bajo los márgenes predeclarados.

## Qué sí aprendió la cabeza de elección

El control matched evita confundir un resultado negativo con incapacidad total
de la segunda cabeza. Con targets verdaderos, la cabeza superó al derangement en
`+0.1146` de top-1 libre y `+0.0208` de top-1 gated. La señal existe y el último
valor supera apenas el mínimo `+0.02`. Sin embargo, el conjunto predicho
restringe casi toda esa ventaja: aprender algo acerca de qué miembro es
compatible no bastó para mejorar la decisión final ni compensar el aprendizaje
de conjunto perdido por congelar temprano.

## Implicación arquitectónica

La separación conceptual entre **conjunto identificado** y **decisión** sigue
siendo necesaria, pero esta implementación simple no queda respaldada como
receta para un protocolo prospectivo. El problema no se resuelve agregando una
cabeza y congelando el espacio común: sin una autoridad de utilidad, costo o
contexto, la cabeza de elección sólo redistribuye preferencia entre miembros ya
compatibles, mientras el congelamiento interrumpe el aprendizaje del conjunto.

El próximo discriminante no debería repetir el mismo two-stage con más épocas.
Las alternativas recuperables son: una decisión condicionada por autoridad
externa real; una optimización multiobjetivo que preserve explícitamente el
frente de Pareto entre cobertura, ancho y elección; o una réplica bajo generador
o aparato independiente antes de atribuir generalidad a la señal set-valued de
Ola 50. Ninguna alternativa queda promovida por este smoke.

## Integridad y replay

- corrida primaria: `46.35 s`, RSS máximo `958 536 KiB`, CUDA deshabilitada;
- réplica: `47.92 s`, RSS máximo `958 572 KiB`, CUDA deshabilitada;
- `57/57` NPZ array-exact;
- `24/24` estados de modelo exactos;
- `8/8` archivos de métricas por token byte-exact;
- split y mapping shuffled byte-exact;
- summaries semánticamente iguales al excluir tiempo, RSS y hashes derivados;
- `92/92` artefactos primarios y `3/3` piezas finales verificados contra sus
  manifiestos.

`artifact_hash` fue idéntico en ambas corridas:
`fbd868cf781f6e09733515573586bd7306deac9c564b7ce4b504fd30ceb8e0a0`.
La auditoría independiente preejecución quedó preservada en
`agent_reports/287_wave51_factored_smoke_independent_audit.md`.

## Alcance

Esta es evidencia de desarrollo sobre datos históricos abiertos. No leyó el
lockbox, no replica con generador independiente, no valida una política
semántica, una geometría natural o una PPU, y no decide GO/NO-GO.
