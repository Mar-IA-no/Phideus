# Plan CPU — gate residual sensible a localización topológica

**Fecha:** 2026-09-04
**Estado:** diseño inicial; implementación pendiente
**Régimen:** contraste prospectivo en tres realizaciones frescas
**Autoridad:** discrimina valor de observables topológicos públicos dentro del banco; no promueve arquitectura ni decide GO/NO-GO

## Pregunta

El gate de ventaja global reproduce beneficio grouped, pero no ordena seguridad
IID de manera estable. La auditoría R359 mostró que aumentar el número de
masters bajo un efecto fijo no basta: los candidatos con mejor proyección
mantienen mejora grouped y balanceada, mientras el punto IID cambia de signo
entre realizaciones. La diferencia generativa entre mecanismos no está en la
magnitud total de la corrupción —que se comparte dentro de cada master— sino en
dónde cae sobre el grafo.

El contraste siguiente pregunta si una descripción pública, invariante y de
baja dimensión de la localización de la corrección mejora el routing frente a
dos controles distintos:

1. el gate público vigente de quince features;
2. un gate de igual dimensión cuya localización se destruye permutando la
   corrección entre aristas, sin cambiar su multiconjunto de magnitudes.

No se usan masks causales, mecanismo, `x_true`, relación limpia, error de test
ni resultado del solver para construir acciones desplegables.

## Fuentes congeladas

El tronco neuronal y las heads post-IRLS permanecen congelados en
`proportional_graph_irls_loss_contrast_v1`. El config generativo, los cuatro
brazos `RAW/CLOSURE × GENERIC/TYPED`, las seeds neuronales
`104729/130363`, los alphas `{0,.25,.5,.75,1}` y el executor IRLS canónico no
cambian. La cadena R356–R359 funciona como motivación abierta, no como conjunto
de ajuste.

Se generan tres realizaciones nuevas y disjuntas:

- **calibration**, seed `2026090609`: ajusta únicamente los gates ridge;
- **policy selection**, seed `2026090617`: selecciona thresholds bajo la misma
  banda superior simultánea IID;
- **adjudication**, seed `2026090629`: permanece sin materializar hasta
  congelar modelos, grillas, thresholds, hashes y receipt.

Cada realización debe aportar al menos `220` masters IID/grouped completos. No
se cambian seeds por conteo ni outcome. Preflights y tests usan otras seeds.

## Features de localización

Para una vista, sea `delta_e = corrected_e - observed_e`, `a_e=|delta_e|` y
`B` la incidencia orientada pública. Las features base siguen siendo las quince
de R356. Se agregan nueve escala-invariantes, con convención cero explícita:

1. `correction_location_defined`: uno si `sum a_e > 0`, cero si no;
2. `correction_edge_entropy_normalized`: entropía de `a_e/sum a` dividida por
   `log |E|`;
3. `correction_node_mass_entropy_normalized`: entropía de la masa incidente
   `m=|B|^T a` dividida por `log |V|`;
4. `correction_node_mass_max_share`: `max(m)/sum(m)`;
5. `correction_node_mass_effective_fraction`:
   `1 / (|V| sum_v q_v^2)`, con `q=m/sum(m)`;
6. `correction_linegraph_product_ratio`: media de `a_e a_f` sobre pares de
   aristas adyacentes, dividida por `mean(a^2)`;
7. `correction_linegraph_smoothness`: media de `(a_e-a_f)^2` sobre esos pares,
   con el mismo denominador;
8. `correction_divergence_entropy_normalized`: entropía de
   `|B^T delta| / sum |B^T delta|`, dividida por `log |V|`;
9. `correction_divergence_max_share`: máximo de esa distribución.

Si la masa correspondiente es cero, todas sus estadísticas se fijan en cero;
el flag conserva la distinción entre ausencia y concentración. Los pares de
line graph son unordered. Las nueve features deben pasar tests de relabeling de
nodos, reordenamiento/reversión coherente de aristas, escala positiva de
`delta` y finitud en casos degenerados.

## Control topológico matched

El control conserva las quince features base verdaderas. Para las nueve nuevas,
permuta `delta` sobre las aristas de cada vista mediante una permutación
determinista derivada de `phase × arm × neural_seed × view_public_hash ×
replicate`. Preserva exactamente el multiconjunto firmado y absoluto de la
corrección, su norma, máximo, flag y entropía de aristas; destruye su asignación
a nodos y vecindades.

Se usan dieciséis réplicas. Cada una tiene la misma cantidad de inputs,
parámetros, outputs, folds, lambdas y targets que el gate topológico real. Se
promedian sus métricas por vista; nunca se elige la réplica más favorable. Un
segundo conjunto de dieciséis controles conserva features topológicas reales y
rota los targets por master dentro de estratos de tamaño, para detectar crédito
espurio de labels.

## Familias y ajuste

Todos los modelos son ridge lineales de cuatro salidas, con lambdas
`{0,.01,.1,1,10,100}` y cinco folds agrupados por master:

1. `correction_scale`: dos features de magnitud heredadas;
2. `public_base`: las quince features públicas heredadas;
3. `topology_augmented`: las quince base más las nueve de localización;
4. `topology_permuted`: dieciséis controles de veinticuatro features;
5. `target_shuffled_topology`: dieciséis controles de veinticuatro features
   con targets rotados.

Calibration ajusta cada familia sin abrir policy selection. También preserva
OOF predictions, error de predicción y acciones como diagnóstico de tuning, sin
atribuirles autoridad prospectiva.

## Selección segura

Las grillas de threshold se derivan exclusivamente de ventajas positivas en
calibration: `0`, cuantiles `{.25,.5,.6,.7,.8,.9,.95,.975,.99}` e identidad.
Policy selection aplica los modelos congelados y usa `2.000` bootstraps por
master. Para cada familia o réplica se calcula la banda superior simultánea IID
sobre thresholds no identidad. Un candidato es admisible sólo si su límite
superior es no positivo; identidad es admisible algebraicamente. Entre
admisibles se elige el mejor efecto balanceado, con empate hacia mayor
abstención.

Los controles reciben exactamente el mismo selector. Antes de adjudicación se
congelan modelos, permutaciones o seeds de derivación, grillas, thresholds,
config, commit, manifests y ausencia de la seed futura.

## Adjudicación

Sobre los masters frescos se reportan, por brazo y por mecanismo:

- RMSE y delta frente a identidad para unconstrained y safe de cada familia;
- `topology_augmented - public_base`;
- `topology_augmented - topology_permuted` promediado por réplica;
- `topology_augmented - target_shuffled_topology`;
- comparación con `correction_scale`;
- fracción de identidad y distribución de alphas;
- correlación entre ventaja y beneficio realizado;
- interacción grouped menos IID;
- margen oracle por vista, siempre privado y no desplegable.

El master es la unidad bootstrap; primero se promedian seeds y luego las dos
vistas al formar el estimando balanceado. Se usan los mismos `2.000` índices
para todas las políticas. Una no convergencia invalida el master afectado, sin
complete-case silencioso.

## Lecturas admisibles

- si topology safe interviene, conserva un límite superior IID no positivo en
  adjudicación y mejora grouped con límite superior negativo frente a
  identidad, public base y ambos controles, la localización aporta routing
  transportable bajo esta ley; los resultados se informan además por brazo y
  no se convierten en un GO/NO-GO agregado;
- si topology sólo supera al control target-shuffled, hay señal aprendible pero
  no atribución específica a la asignación topológica;
- si no supera topology-permuted, las nueve features no reciben crédito aunque
  el gate mejore identidad;
- si vuelve a seleccionar identidad, el firewall permanece informativo y no se
  relaja sobre adjudicación;
- un promedio balanceado favorable no sustituye seguridad IID;
- ningún resultado autoriza geometría física, interfaz universal, transferencia
  de dominio, promoción o GO/NO-GO.

## Artefactos y recursos

Cada fase preserva vistas, hashes públicos, features base/topológicas y de
control, targets sólo donde corresponden, predicciones, acciones, alphas,
estados forward/solver, modelos, folds, thresholds, bootstraps, effects,
receipt, manifest y replay. Calibration y selection deben poder inspeccionarse
sin que exista el directorio de adjudicación.

La ejecución usa `CUDA_VISIBLE_DEVICES=''`, un thread, máximo `12 min` y
`4 GiB` por fase. Todo trabajo GPU continúa en cola hasta que Mariano revoque
explícitamente la suspensión.
