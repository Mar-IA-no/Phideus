# Ola 50 — cierre prospectivo del contraste neuronal set-valued

> **Estado:** `PROSPECTIVE-INTERNAL / EXACT-REPLAY-PASS / JOINT-PATTERN-FALSE / TECHNICAL-RECOVERY-DISCLOSED / USER-GO-NOGO`
> **Fecha:** 2026-09-03
> **Código ejecutado:** `78e9377693bbe8b105ea5b356aac431fb4cc38a4`
> **Antecedentes:** `WAVE_49_CLASSICAL_BENCHMARK_CLOSED.md`, `WAVE_50_MATCHED_NEURAL_SET_OUTPUT_PLAN.md`

## Pregunta adjudicada

La Ola 50 probó una decisión arquitectónica estrecha. Con el mismo DeepSets,
los mismos cuatro logits, inicialización, batches, presupuesto y target
multi-hot, comparó dos paquetes de salida y entrenamiento:

- `softmax_partial`: una distribución exclusiva sobre cuatro familias, entrenada
  con partial-label loss;
- `sigmoid_set`: cuatro decisiones compatibles independientes, entrenadas con
  BCE multi-label.

No fue una comparación entre encoders ni una prueba de una PPU completa. El
estimando fue si un lector que no impone exclusividad conserva mejor un conjunto
de relaciones observacionalmente compatibles sin perder más de tres puntos en
la decisión top-1 común.

## Resultado confirmatorio

La unidad de análisis fue `pair_token`; las vistas correlacionadas se redujeron
antes del bootstrap. El sistema primario promedió logits crudos de cinco seeds y
la población confirmatoria quedó fijada de antemano en los `384` tokens
`NEAR_RIVAL` del lockbox fresco.

| Diferencia `sigmoid - softmax` | Estimación | IC bilateral 97,5% | Condición preregistrada |
|---|---:|---:|---|
| recall del conjunto compatible | `+0.1148` | `[+0.0693, +0.1582]` | cumple: límite inferior `> +0.03` |
| top-1 compatible | `-0.0013` | `[-0.03125, +0.02865]` | no cumple: límite inferior no supera `-0.03` |

El patrón evidencial conjunto es **falso**. La mejora de recall es grande,
consistente en las cinco seeds y compatible con la hipótesis de que el simplex
exclusivo descarta alternativas legítimas. La no inferioridad top-1, en cambio,
quedó a `0.00125` del margen por su límite inferior. El punto estimado es casi
nulo, pero el protocolo adjudica intervalos, no impresiones sobre el punto.

La salida sigmoid también excedió levemente el límite de transferencia de ancho
calibrado en val: `mean_width - mean_target_cardinality = 0.2565`, frente al
máximo `0.25`. La tasa de fixtures con alguna selección incompatible fue
`0.3587`, dentro del máximo `0.40`. Este chequeo era report-only por
preregistración: no invalida la corrida, pero impide presentar la ganancia de
recall como gratuita.

## Lectura de controles

Los controles delimitan el mecanismo, pero no convierten el resultado en una
prueba causal más amplia.

- En `sigmoid_set`, quitar EIV elevó el recall `NEAR_RIVAL` de `0.8644` a
  `0.9505`, pero también amplió el ancho de `2.9766` a `3.3932` y la tasa de
  selección incompatible de `0.3320` a `0.4896`. La incertidumbre observacional
  opera como restricción de riesgo y ancho; no es una fuente monotónica de
  accuracy.
- `shuffled_target` debe compararse sólo con `true_target_control`, porque ambos
  usan el mismo subconjunto balanceado. En sigmoid, barajar baja recall
  `0.4505 -> 0.3984`, top-1 `0.7695 -> 0.7292` y cobertura completa
  `0.1940 -> 0.0859`. Hay señal del target, aunque el control reducido no
  autoriza compararlo directamente con `main`.
- En softmax, el barajado puede aumentar recall expandiendo el conjunto, pero
  empeora top-1, incompatibilidad y ancho frente al control verdadero. Ninguna
  métrica aislada basta para declarar aprendizaje relacional.
- La referencia clásica `catalog_eiv` conserva una frontera importante. En
  `ALL_ELIGIBLE` alcanza recall `0.9303`, incompatibilidad `0.2090` y ancho
  `2.4824`; el sigmoid neuronal obtiene `0.8624`, `0.3587` y `2.7083`. La ola
  discrimina una interfaz neuronal; no demuestra superioridad frente al método
  clásico ni una nueva geometría.

## Integridad y recuperación técnica

La generación, entrenamiento e inferencia permanecieron separados. Training no
montó el visible del lockbox; inferencia se ejecutó como proceso sin capacidad
de fit; predicciones, checkpoints, normalizador y thresholds se congelaron antes
de materializar el oracle. Las diez mutaciones protocolarias fueron rechazadas
y `23/23` tests focales de Ola 50 pasan en el cierre.

La corrida canónica requirió una recuperación técnica porque dos defectos del
checker/evaluador aparecieron después de congelar predicciones: el conjunto
compatible se comparaba como lista ordenada y la evaluación intentaba alinear el
inventario visible completo con el subconjunto etiquetado. Durante el segundo
incidente se observó un preview del resultado. No se generó otro lockbox ni se
retunearon modelo, target, umbral o criterio: una autorización versionada ligó
la recuperación al intento exacto y permitió sólo los dos deltas de código de
alineación. Antes de reabrir el oracle, la recuperación verificó `80`
checkpoints por semántica, `161` NPZ arreglo por arreglo, `29` archivos byte a
byte y tres receipts runtime bajo normalización declarada.

El replay final independiente repitió el resultado y verificó `80` checkpoints,
`161` NPZ, `73` archivos byte a byte, cuatro manifests validados y dos manifests
comparados semánticamente. “Replay exacto” significa aquí exactitud
semántica/por-array/por-byte bajo exclusión explícita de timestamps del baseline
clásico y paths de comandos dependientes del directorio de ejecución; no
significa identidad textual de esos campos runtime.

## Implicancia arquitectónica

La evidencia favorece conservar dos operaciones que una clasificación cerrada
mezcla: **representar el conjunto identificado** y **tomar una decisión dentro
de él**. Una arquitectura proporcional candidata puede usar una cabeza
set-valued independiente para mantener relaciones todavía compatibles y delegar
la selección puntual a una política posterior, informada por contexto, costo o
una nueva medición. Esta separación es más coherente con el ground truth
estratificado de las olas previas que forzar una familia única dentro del
embedding.

La Ola 50 no promueve esa candidata. El joint pattern no se cumplió, la salida
sigmoid pagó ancho, el clásico EIV siguió siendo más fuerte y el paquete procede
del mismo generador que la Ola 49. Los próximos discriminantes útiles son:

1. replicar bajo un generador o aparato independiente antes de hablar de
   transferencia;
2. aumentar evidencia independiente para estimar con precisión el margen top-1,
   sin cambiar retrospectivamente el `-0.03` de esta ola;
3. comparar una arquitectura de dos etapas —conjunto identificado y política de
   decisión— contra ambos brazos matched, preservando costo y parámetros;
4. exigir que el lector neuronal alcance la frontera clásica EIV en recall,
   riesgo y ancho, no sólo que supere al softmax.

La elección entre estos caminos y cualquier `GO/NO-GO` siguen perteneciendo al
usuario.

## Artefactos

- corrida canónica: `data/geometria_proporcional/wave50_prospective_v1/`;
- replay: `data/geometria_proporcional/wave50_prospective_v1_replay/`;
- resumen numérico: `prospective_summary.json` dentro de cada corrida;
- comparación de replay: `exact_replay_comparison.json` dentro del replay;
- auditoría independiente final: `agent_reports/286_wave50_prospective_final_independent_audit.md`.
