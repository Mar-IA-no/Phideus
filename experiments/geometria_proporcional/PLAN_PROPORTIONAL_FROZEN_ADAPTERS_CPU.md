# Plan CPU — adaptadores congelados por semántica de executor

**Fecha:** 2026-09-03  
**Estado:** diseño previo a implementación  
**Régimen:** seguimiento sintético exploratorio; test ya abierto; CPU-only  
**Autoridad:** produce evidencia; no promueve arquitectura ni decide GO/NO-GO

## Pregunta

¿El estado relacional de los checkpoints existentes contiene señal suficiente
para aprender interfaces estrechas compatibles con cada executor, o la
corrección requiere modificar encoder y mixer?

El diagnóstico anterior separó dos semánticas. WLS aprovecha una importancia
localizada por arista; IRLS funciona mejor sin ese peso exógeno y debe recibir
una relación que no desorganice su robustez residual. Este seguimiento congela
el tronco completo y reoptimiza una sola head por vez.

## Preflight de factibilidad ya verificado

Los dieciséis checkpoints `.npz` separan `edge_encoder`, `path_mlp`,
`path_score`, `update_gate`, `update_value`, `norm`, `correction_head` y
`reliability_head`, junto con metadata y estado de optimizador. Los modelos de
path mixing tienen `50.435` parámetros entrenables originales. La head de
confiabilidad contiene `4.225` y la de corrección `4.224`: alrededor de `8,38%`
del modelo cada una.

Una carga CPU real de `raw_typed|seed=104729` confirmó que:

- al habilitar sólo `reliability_head`, únicamente ese grupo recibe gradiente;
- al habilitar sólo `correction_head`, únicamente ese grupo recibe gradiente;
- ningún parámetro congelado conserva gradiente;
- CUDA permanece invisible.

El formato permite, por tanto, aislar las heads sin reconstruir el training
completo. El estado de AdamW histórico no se reutiliza: el objetivo y el
conjunto de parámetros cambian, de modo que cada adaptador recibe optimizador
nuevo y trazable.

## Universo y brazos

Se reutilizan sin regenerar decisiones los splits, lineages, elegibilidad y dos
seeds del smoke neuronal. Entran los cuatro brazos primarios:

- `RAW-GENERIC`;
- `RAW-TYPED`;
- `CLOSURE-GENERIC`;
- `CLOSURE-TYPED`.

Cada checkpoint origina dos adaptadores:

1. **WLS-weight adapter.** Congela encoder, mixer, normalización y
   `correction_head`; usa siempre la relación observada y reoptimiza sólo
   `reliability_head` mediante MSE de potenciales módulo gauge a través del WLS
   diferenciable.
2. **IRLS-relation adapter.** Congela encoder, mixer, normalización y
   `reliability_head`; reoptimiza sólo `correction_head` mediante MSE contra la
   relación limpia más el término de cierre ya autorizado. En evaluación usa
   peso unitario y el Huber-IRLS externo congelado.

La segunda rama no se denomina entrenamiento diferenciable por IRLS. El solver
robusto vigente es externo y NumPy; sólo adjudica utilidad downstream. La head
se entrena como denoiser relacional sintético y se evalúa después con IRLS.

## Contratos de comparación

Por brazo, seed y slice se preservan al menos:

- WLS estático `observed|inherited-learned` frente a WLS con weight adapter;
- IRLS estático `observed|unit` frente a IRLS con relation adapter;
- paquete heredado `corrected|inherited-learned` como referencia histórica;
- outputs del checkpoint antes y después del refit;
- métricas de relación, WLS, IRLS, convergencia y condición.

La unidad sigue siendo el master. IID y grouped se informan por separado. El
promedio entre seeds precede al efecto pareado y el bootstrap reutiliza índices
preservados cuando la cardinalidad coincide. Cualquier fallo IRLS vuelve no
evaluable el estimando agregado afectado; no se promedian supervivientes.

## Entrenamiento y selección

- cinco épocas fijas;
- mismo orden determinista por seed y época del smoke;
- AdamW nuevo con `lr=1e-3`, `weight_decay=1e-4` y clipping `5.0`;
- CPU, un thread numérico, CUDA invisible;
- sin ajuste por test ni selección de época;
- validation se reporta como diagnóstico, no elige configuración;
- techo de `30 min` y `4 GiB` RSS para la corrida completa.

Cinco épocas recortan el costo sin convertir esta prueba en evidencia de techo.
Si el preflight de una combinación proyecta exceder el presupuesto, se detiene
y se rebasa el alcance antes de producir un artefacto oficial.

## Orden de lectura predeclarado

1. integridad del freeze y conteo de parámetros;
2. WLS adapter frente a su interfaz estática;
3. IRLS relation adapter frente a `observed|unit`;
4. interacción `GENERIC/TYPED × RAW/CLOSURE` por executor;
5. convergencia y fallos numéricos;
6. relation RMSE como mecanismo, no sustituto de utilidad downstream.

No se fija un umbral de promoción. Las lecturas posibles son:

- una head mejora downstream con tronco congelado: compatibilidad con la
  hipótesis de que el estado común ya contiene señal útil;
- mejora pre-solver sin mejora downstream: desajuste entre target de head y
  semántica del executor;
- ausencia de mejora: no prueba que el tronco sea inútil, porque el régimen es
  corto, de dos seeds y con test previamente abierto;
- efecto sólo en un slice: se reporta como tal, sin generalización.

## Artefactos y replay

La corrida debe preservar config resuelta, hashes de fuentes y checkpoints,
parámetros congelados/entrenables, checkpoints adapter `last_epoch`, historia
por época, estados raw por master/arista, fallos, efectos, índices bootstrap,
manifest, observación de runtime separada y un script de replay autocontenido.
El replay debe exigir commit limpio, verificar hashes y reproducir byte por
byte todos los artefactos deterministas.

## Cola explícita fuera de alcance

- diferenciación implícita o unrolling de IRLS;
- reentrenamiento de encoder o mixer;
- seeds adicionales;
- transferencia a dominio físico o Atención Armónica;
- freeze confirmatorio;
- toda ejecución GPU.

Estas tareas permanecen en cola hasta evidencia o autorización posterior.
