# Plan CPU — gate residual de identidad para corrección post-IRLS

**Fecha:** 2026-09-04  
**Estado:** corrida oficial y replay completados
**Régimen:** seguimiento exploratorio; validation IID y test previamente abierto  
**Autoridad:** prueba routing público dentro del banco; no promueve arquitectura ni decide GO/NO-GO

## Pregunta

La pérdida post-IRLS corrigió el desajuste del target local, pero dejó una
asimetría: empeora `observed|unit` en IID y mejora tres de cuatro brazos
grouped. En validation, aun con media adversa, entre `26,8%` y `28,3%` de las
vistas mejoran con corrección completa. El siguiente contraste pregunta si esa
heterogeneidad puede anticiparse antes del solver mediante observables públicos,
o si sólo aparece cuando el target privado adjudica el resultado.

El gate no genera otra corrección. Elige una intensidad residual

```text
y(alpha) = observed + alpha * (post_irls - observed),
alpha in {0, 0.25, 0.5, 0.75, 1}.
```

`alpha=0` copia la observación byte por byte y constituye abstención exacta;
`alpha=1` reproduce la head post-IRLS. Todos los niveles usan peso unitario y
el IRLS NumPy canónico.

## Fuentes y universo

La fuente primaria es
`data/geometria_proporcional/proportional_graph_irls_loss_contrast_v1/`,
incluidos manifest, config, `raw_eval`, métricas, bootstrap y checkpoints. El
runner verifica sus hashes y reconstruye las observaciones públicas desde el
config del smoke. No reentrena encoder, mixer o heads.

El manifest fuente queda congelado con SHA-256
`52c714901f633ba4d20d8c2a8702648800f3d5f8c77b9fd2c78092995d0dc8aa`.

Se conservan cuatro brazos, dos seeds, `127` vistas validation IID y `252`
masters test con vistas IID/grouped pareadas. Las decisiones se comparten entre
seeds: primero se promedian features y pérdidas por `arm × view`, y luego se
elige alpha. Seed nunca entra como feature.

## Frontera de información

El vector público congelado contiene quince magnitudes:

- nodos, aristas válidas, densidad y caminos válidos;
- media, desvío y máximo de varianza observacional;
- RMS y máximo absoluto de la relación observada;
- RMS y máximo absoluto de la corrección post-IRLS;
- RMS y mediana absoluta de cierre para observación y corrección.

No recibe `x_true`, relación limpia, máscara causal, mecanismo, split, IDs,
seed, métricas, convergencia ni outputs post-solver. Los targets privados sólo
construyen en validation el delta de RMSE de cada alpha frente a identidad; en
test aparecen después de congelar modelo y decisiones para puntuar.

Una mutación de targets y métricas test debe dejar byte-idénticos modelos,
alphas y decisiones. Cambiar etiquetas de mecanismo sin reordenar vistas debe
dejar las decisiones idénticas. El constructor de features rechaza campos
privados y valores no finitos.

## Brazos del gate

1. **identity:** `alpha=0` en toda vista.
2. **always-post:** `alpha=1` en toda vista.
3. **constant-validation:** un alpha global por brazo, elegido por RMSE medio
   validation después de promediar seeds; empates favorecen el menor alpha.
4. **correction-scale ridge:** predictor reducido que usa sólo RMS y máximo de
   corrección; controla cuánto aporta una heurística de intensidad.
5. **public ridge gate:** cuatro regresiones ridge predicen en validation el
   delta de cada alpha no nulo frente a identidad. La acción toma el mínimo
   predicho entre esos cuatro valores y cero; empates favorecen identidad y
   luego el menor alpha.
6. **shuffled-target ridge:** dieciséis réplicas del mismo gate después de
   permutar conjuntamente los cuatro targets dentro de estratos de número de
   nodos. Conservan features, capacidad, folds y distribuciones marginales;
   destruyen sólo la correspondencia pública con el beneficio.
7. **oracle per-view:** mejor alpha por target retenido, únicamente como cota
   diagnóstica no deployable.

## Ajuste y selección

Los folds son cinco particiones deterministas agrupadas por master. Para cada
familia de features, `lambda ∈ {0, 0.01, 0.1, 1, 10, 100}` se selecciona por
MSE out-of-fold conjunto sobre los cuatro deltas; el empate favorece mayor
regularización. El reporte validation de cada gate usa predicciones out-of-fold,
no el reajuste in-sample. Como esas mismas predicciones intervienen en elegir
`lambda`, su error es un diagnóstico de tuning y no una estimación independiente.
Elegido lambda, el modelo final se ajusta en toda validation y se aplica una
sola vez a test IID/grouped sin conocer mecanismo.

Las réplicas shuffled repiten selección de lambda y ajuste. No se selecciona
la réplica más favorable: sus resultados se promedian por vista antes del
contraste. El gate reducido y el completo comparten folds, lambdas, regla de
acción y presupuesto; difieren sólo en features.

## Evaluación

Por brazo y slice se preservan:

- RMSE absoluto y delta del gate frente a identity, always-post y constante;
- fracción de acciones por alpha y tasa exacta de identidad;
- regret frente al oracle per-view;
- correlación predicción–delta y MSE en validation OOF, test IID y grouped;
- efecto del gate completo frente al control reducido y al shuffle;
- convergencia, iteraciones, condición y fallos del IRLS;
- consistencia entre seeds antes del promedio.

El bootstrap reutiliza `2.000` índices por master. Cualquier no convergencia en
una seed vuelve no evaluable el estimando agregado afectado. No se promedian
supervivientes. Validation sólo contiene IID: un beneficio grouped en test es
transporte exploratorio, no validación independiente.

## Lecturas admisibles

- superar identity en IID y grouped, además de los controles, favorece una
  señal pública de competencia por vista;
- igualar identity porque validation elige abstención es un resultado válido:
  el gate evita daño, pero no extrae la mejora grouped;
- ganar grouped y perder IID conserva la disociación y no resuelve routing;
- superar always-post pero no identity muestra mitigación, no utilidad neta;
- no superar shuffles rechaza atribución a la correspondencia entre features y
  beneficio;
- ningún resultado autoriza transferencia física, arquitectura promovida,
  techo o GO/NO-GO.

## Artefactos, recursos y replay

Se guardan atestación de fuente, config resuelta, matriz de features, métricas
por alpha y seed, folds, modelos, predicciones OOF/test, decisiones, shuffles,
estados crudos por solver, efectos, bootstrap, muestras de RAM, manifest y
replay exacto. La corrida es CPU-only, un thread, `CUDA_VISIBLE_DEVICES=''`,
techo de `10 min` y `4 GiB`. Toda alternativa GPU permanece en cola.
