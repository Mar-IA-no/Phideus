# Plan CPU — pérdida relacional local frente a pérdida post-IRLS

**Fecha:** 2026-09-04  
**Estado:** protocolo implementado; corrida oficial pendiente  
**Régimen:** seguimiento sintético exploratorio; test ya abierto; CPU-only  
**Autoridad:** compara objetivos de una head congelada; no promueve arquitectura ni decide GO/NO-GO

## Pregunta

El refit head-only anterior redujo el error marginal de relación y empeoró
IRLS. El diagnóstico siguiente mostró que un unroll Torch de `K=64` reproduce el
executor robusto con fidelidad suficiente dentro de validation. La pregunta
ahora es estrecha: con el mismo estado relacional y la misma capacidad, ¿cambia
la utilidad downstream cuando la `correction_head` se optimiza por potenciales
posteriores a IRLS en lugar de por relaciones limpias arista a arista?

## Universo y freeze

Se reutilizan los cuatro brazos primarios y dos seeds del smoke:
`RAW/CLOSURE × GENERIC/TYPED`. Cada modelo parte del mismo checkpoint original;
encoder, mixer, normalización y `reliability_head` quedan congelados. Sólo se
entrenan los `4.224` parámetros de `correction_head`.

Train conserva las `496` vistas elegibles. Validation y test conservan las
vistas ya abiertas por experimentos anteriores; ninguna de ellas interviene en
ajuste, escala, selección de época o hiperparámetros. El resultado es
exploratorio y se informa por IID/grouped, sin reinterpretarlo como confirmación.

## Contraste causal de objetivos

Por checkpoint se crean dos copias idénticas:

1. **local_relation:** MSE de `y_tilde` contra la relación limpia más `0.05`
   de cierre L1, igual a la jurisdicción del adaptador anterior;
2. **post_irls:** MSE de `x_hat` contra el potencial verdadero en gauge de media
   cero, donde `x_hat` proviene del surrogate unit-base `K=64`, más el mismo
   término de cierre L1.

Los dos regímenes reciben los mismos inputs públicos y targets privados
autorizados por el simulador. Ninguno recibe máscara causal. El executor IRLS
canónico NumPy no participa en backprop y permanece como adjudicador externo.

## Control de escala de optimización

Cambiar de target cambia también unidades y magnitud del gradiente. Para no
confundir una dirección distinta con un learning rate efectivo arbitrario, se
usan los primeros cuatro batches del orden compartido de época cero, sin
actualizar parámetros. Se calcula la mediana de la norma L2 del gradiente total
de cada objetivo. La pérdida `post_irls` recibe un único multiplicador

```text
scale = median_grad_norm(local_relation) / median_grad_norm(post_irls).
```

El factor se fija por checkpoint, usa sólo train y debe ser finito dentro de
`[1e-3, 1e3]`. No se recalcula por época. Se preservan normas por batch, factor,
pérdidas sin escalar y gradientes pre-clipping. Esta igualdad sólo controla la
escala inicial; la trayectoria posterior puede separarse y se reporta como
parte del mecanismo, no como nuisance eliminado.

## Presupuesto igualado

- cinco épocas fijas y batch efectivo `64`;
- mismo orden de vistas por brazo, seed y época para ambos objetivos;
- AdamW nuevo, `lr=1e-3`, `weight_decay=1e-4`, clipping `5.0`;
- acumulación por vista antes de un único update de batch, para liberar el grafo
  de `K=64` sin cambiar el gradiente medio;
- ninguna selección de época ni tuning por validation/test;
- CPU, un thread numérico, `CUDA_VISIBLE_DEVICES=''`, techo de `20 min` y
  `4 GiB` por corrida.

El costo computacional no queda igualado entre objetivos: el unroll post-IRLS
es intrínsecamente más caro. Sí quedan igualados capacidad, ejemplos,
backprops, updates y schedule. Runtime se informa como costo del mecanismo.

## Evaluación y orden de lectura

Cada salida se evalúa con relación corregida, peso unitario y el IRLS canónico
convergido. Se preservan también `observed|unit`, el adaptador local histórico y
la salida heredada. El orden de lectura es:

1. integridad del freeze, igualdad inicial y control de escala;
2. convergencia del executor canónico;
3. `post_irls - local_relation` en RMSE de potenciales, por IID/grouped;
4. cada objetivo frente a `observed|unit`;
5. RMSE relacional como mecanismo secundario;
6. interacción `RAW/CLOSURE × GENERIC/TYPED` y consistencia entre seeds;
7. costo temporal y RAM.

El promedio entre seeds precede al delta por master y el bootstrap jerárquico
reutiliza los `2.000` índices preservados. Un fallo IRLS vuelve no evaluable la
celda agregada afectada; no se promedian sólo supervivientes.

## Interpretaciones admisibles

- una mejora post-IRLS consistente con igual escala inicial favorece la
  hipótesis de desajuste del target local;
- una mejora sólo IID o sólo en algunos brazos se informa en esos slices;
- mejor RMSE relacional sin mejora post-solver replica el desajuste previo;
- ausencia de mejora no establece un techo: son cinco épocas, dos seeds,
  tronco congelado y un surrogate unit-base específico;
- ninguna salida acredita geometría física, transporte a otro dominio,
  arquitectura promovida o GO/NO-GO.

## Artefactos y replay

Se preservan config resuelta, hashes de fuentes/checkpoints/crudos históricos,
contrato de freeze, control de escala, historia por época, checkpoints
`last_epoch`, relaciones por vista y arista, métricas por unidad, fallos,
bootstrap, serie de RAM, manifest, runtime separado y replay byte-exacto. Test
no se reabre para seleccionar: sólo se aplica el contraste congelado.

## Fuera de alcance y cola GPU

No se reentrenan encoder o mixer, no se añaden seeds, no se barre `K`, no se
ajustan lambdas y no se transfiere a datos físicos. Cualquier reentrenamiento
integral o extensión GPU queda en cola hasta que Mariano retire la suspensión
vigente.
