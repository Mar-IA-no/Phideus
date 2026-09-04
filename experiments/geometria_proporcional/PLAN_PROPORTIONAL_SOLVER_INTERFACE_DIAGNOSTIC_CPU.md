# Plan CPU — diagnóstico de interfaz y semántica de confiabilidad

**Estado:** auditado internamente contra fuentes; habilitado para implementación CPU  
**Fecha:** 2026-09-03  
**Régimen:** seguimiento exploratorio post hoc con estados congelados; CPU-only  
**Autoridad de promoción y GO/NO-GO:** usuario

## 1. Problema que queda abierto

El desentrelazado mostró que la pareja `relación corregida + peso aprendido` no
forma un contrato solver-agnóstico. Bajo WLS, el peso aprendido sobre la
observación aporta la mayor mejora y la corrección devuelve parte de ella; bajo
IRLS, la corrección perjudica principalmente IID y el peso aprendido perjudica
grouped. Los controles reproducen esa inversión, de modo que no alcanza con
atribuirla al mixer tipado.

El siguiente contraste pregunta qué semántica funcional tiene la confiabilidad
aprendida cuando entra en cada executor:

1. ¿su utilidad depende de dónde ubica los pesos o sólo de cuán concentrada es
   su distribución?;
2. ¿IRLS vuelve a suprimir las mismas aristas y produce una doble
   reponderación que reduce masa efectiva?;
3. ¿un temperado monotónico, que conserva el ranking de la head pero reduce su
   intensidad, se selecciona en validation y transporta a test IID/grouped?;
4. ¿observables públicos pre-solver permiten anticipar cuándo usar peso
   aprendido, o el patrón de validation IID no transporta al mecanismo
   grouped?

La prueba no reentrena el encoder, el mixer ni las heads. Sí ejecuta nuevas
combinaciones de pesos mediante los mismos WLS e IRLS congelados. Como la
pregunta nació después de abrir el test del smoke y del desentrelazado, todo el
resultado es exploratorio. Un freeze previo limita grados de libertad desde
este punto, pero no restaura un estatuto confirmatorio perdido.

## 2. Fuentes y atestación

Las únicas fuentes admitidas son:

- `data/geometria_proporcional/proportional_graph_neural_smoke_v1/` para
  observaciones canónicas, outputs neuronales y targets de scoring;
- `data/geometria_proporcional/proportional_graph_solver_disentanglement_v1/`
  para celdas base, efectos, pesos finales y lineage de la recombinación.

El modo oficial exige:

- manifest del smoke con SHA-256
  `7e982a92bd366a4c22fe95c0cc9c8fd5f7a1773bd78c4e8e0781b5c2259624e1`;
- manifest del desentrelazado con SHA-256
  `17d09bdee0100ebc1d5a7ee4f2f3c826e51b6e04ef4ecbc33c8f290fb5cf0347`;
- commit fuente registrado por el desentrelazado
  `0731bc054fa9c716483ac3d09b12fb922446a784`, más commit y worktree limpio
  propios del nuevo runner antes de su primera ejecución oficial;
- hashes y tamaños válidos para los dieciséis raws neuronales, el control
  `observed_unweighted`, `per_view_metrics.npz`, los dieciséis estados
  `raw_solver`, ambos configs resueltos y ambos índices bootstrap;
- igualdad exacta de brazos, seeds, vistas, masters, splits, mecanismos,
  offsets, observación pública, relaciones corregidas y confiabilidad entre
  las dos fuentes allí donde se solapan;
- para `closure_typed_path_shuffle`, validación interna completa de sus paths
  intervenidos sin exigir igualdad con los paths canónicos no barajados;
- coincidencia byte-exacta de los índices bootstrap copiados por el
  desentrelazado;
- hashes del solver y de los constructores públicos iguales a los atestados.

El runner abre ambas fuentes sólo para lectura, rechaza outputs iguales,
ancestros o descendientes de cualquier fuente, aborta si el destino existe y
no ofrece `--force`. El replay escribe siempre en otro destino.

## 3. Frontera entre inputs públicos, autoridad privada y diagnósticos

La whitelist de inputs pre-solver contiene únicamente:

- `n_nodes`, `edge_index`, `edge_valid` y `edge_variance`;
- `observed_log_ratio`, `path_index`, `path_sign` y `path_valid`;
- `corrected_log_ratio` y `reliability` producidos por el modelo;
- identidad del brazo y del solver como contexto contractual.

Quedan prohibidos como features del selector y de cualquier transformación de
peso:

- `clean_log_ratio`, `x_true` y `causal_corruption_mask`;
- `mechanism`, `split`, seed, IDs persistentes o lineage;
- RMSE de relación o cociente, convergencia observada en test y toda salida
  post-solver de la misma vista.

Los targets privados pueden usarse después del solve para medir error y, en
validation, para ajustar el selector o elegir hiperparámetros. La máscara
causal sólo puede aparecer en un diagnóstico secundario de alineación. Los
pesos finales de IRLS sólo pueden aparecer después de la decisión como
diagnóstico de doble reponderación; nunca vuelven al selector.

Tests espía deben demostrar que ni el constructor de features ni las familias
de peso aceptan campos privados. El mecanismo puede usarse para estratificar
resultados una vez congeladas las decisiones, no como input.

Una mutación focal de todas las métricas y targets test debe dejar idénticos
`static_selection.json`, `temperature_selection.json`, folds, coeficientes y
decisiones validation. Una mutación de `mechanism` que conserve el orden de
vistas también debe dejar idénticas las decisiones por vista. Cualquier cambio
indica leakage y bloquea la corrida.

## 4. Universo y unidad analítica

Se conservan los ocho brazos, dos seeds y `631` vistas por raw: `127` de
validation y `504` de test. Validation contiene únicamente el régimen IID;
test conserva `252` masters con vistas IID/grouped pareadas. Esta asimetría es
parte de la pregunta de transporte, no un detalle que pueda balancearse usando
test.

Para selección e inferencia, primero se promedian los dos seeds dentro de
`arm × view × variante`; luego se forman deltas por vista y se agregan por
master antes de comparar alternativas. El runner verifica que validation tenga
exactamente una vista por master; si no la tiene, mantiene igualmente la media
por master como unidad de selección. IID y grouped se parean por master. Las
ocho familias de brazos se informan completas; los cuatro brazos factoriales
son la lectura primaria y los cuatro controles son diagnóstico de atribución.

## 5. Bloque A — selector estático por solver

Para cada `arm × solver`, validation elige una de las cuatro celdas existentes:

```text
observed|unit
observed|learned
corrected|unit
corrected|learned
```

La regla minimiza el RMSE medio de cociente después de promediar seeds. Una
celda IRLS sólo es elegible si converge en las `127 × 2` evaluaciones de
validation. Los empates dentro de `1e-12` se resuelven por el orden anterior,
que privilegia la interfaz más simple. La elección se congela antes de leer
las métricas test del nuevo runner y se transporta sin conocer mecanismo.

En test se reportan el nivel seleccionado y su delta frente a
`observed|unit`, `observed|learned` y `corrected|learned`. Una no convergencia
en cualquiera de los dos seeds vuelve no evaluable el slice estricto. Un
rescate que reintenta `observed|unit` puede mostrarse sólo como política
operacional secundaria, con tasa de retry y cómputo adicional explícitos.

Este bloque establece un baseline de selección, no una arquitectura nueva. Si
elige siempre una celda obvia, el resultado es precisamente que el routing
global no añade información.

## 6. Bloque B — familias congeladas de confiabilidad

La lectura primaria fija la relación observada para aislar el peso. Una
lectura secundaria repite sólo la familia de temperado sobre la relación
corregida, sin mezclar ambos resultados.

### 6.1 Temperado que conserva ranking

Para cada peso aprendido válido `w_e`, se define:

```text
w_e(alpha) = normalize_mean(exp(alpha * log(max(w_e, 0.001))))
alpha in [0.0, 0.25, 0.5, 0.75, 1.0]
```

La normalización usa sólo aristas válidas y deja peso cero en padding.
`alpha=0` se materializa copiando el peso unitario y `alpha=1` copiando la
confiabilidad fuente; la fórmula se usa sólo en los niveles intermedios. Por
`arm × solver × relación`, validation
elige un único `alpha` minimizando RMSE medio después de promediar seeds. Se
excluye un `alpha` IRLS si tiene cualquier no convergencia en validation. El
orden de desempate es `0.0, 0.25, 0.5, 0.75, 1.0`, para no preferir intensidad
sin evidencia.

La selección se transporta intacta a test IID/grouped. Se informan los cinco
niveles, el alpha elegido, su delta frente a unidad y learned, y el cambio
grouped-minus-IID. No se interpola ni se amplía la grilla después de ver test.

### 6.2 Shuffles de ubicación con distribución preservada

Ocho permutaciones deterministas por `arm × seed × view` reasignan los pesos
aprendidos entre aristas válidas de la misma vista. Preservan exactamente el
multiconjunto, media, cuantiles y masa total; cambian únicamente qué arista
recibe cada valor. Las seeds de shuffle se derivan por SHA-256 de una seed raíz,
el identificador local de vista, brazo, training seed y réplica. Nunca usan
target, mecanismo ni métricas.

Cada réplica se resuelve con WLS e IRLS sobre la relación observada. Las ocho
réplicas se promedian por vista antes de promediar training seeds. El contraste
`learned - shuffled` mide si la ubicación aprendida aporta frente a la misma
concentración marginal. `shuffled - unit` muestra si la distribución por sí
sola altera al solver.

No se interpreta un shuffle como modelo deployable. Es un control de
localización y toda inferencia queda dentro del banco sintético.

Los endpoints `alpha=0/1` deben reproducir las cuatro celdas atestadas del
desentrelazado —relación observada/corregida y solver WLS/IRLS— en `x_hat`,
pesos finales, convergencia, iteraciones, rango, condición y métricas. Los
discretos coinciden exactamente y los floats a `1e-12`; una divergencia aborta.

## 7. Bloque C — diagnóstico pre-solver de transporte

Se construye un vector fijo de observables públicos por `arm × view`,
promediando entre seeds las features dependientes del modelo:

- número de nodos, aristas válidas, densidad y paths válidos;
- media, desvío y máximo de `edge_variance`;
- RMS y máximo absoluto de la observación;
- RMS y máximo absoluto de `corrected-observed`;
- RMS y mediana absoluta del cierre público sobre paths para observación y
  corrección, calculado como la suma de las tres relaciones indexadas por
  `path_index` multiplicadas por sus signos de `path_sign`, después de validar
  offsets, índices, máscaras y orientación;
- media, desvío, mínimo, mediana, máximo, entropía normalizada y tamaño efectivo
  de la confiabilidad;
- correlación de Pearson entre confiabilidad y magnitud de corrección, con cero
  cuando alguna varianza sea nula.

El schema y el orden quedan congelados; no hay selección de features. Los
valores no finitos abortan.

Para cada `arm × solver`, una regresión ridge predice en validation el delta
`Q(observed,learned)-Q(observed,unit)`. La estandarización se ajusta sólo sobre
validation. `lambda ∈ [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]` se selecciona por
cinco folds deterministas agrupados por master; el empate favorece el lambda
mayor. El modelo se reajusta sobre toda validation y en test elige learned sólo
si el delta predicho es menor que cero. No recibe mecanismo ni se recalibra.

Se reportan:

- correlación predicción–delta y error absoluto en validation, test IID y test
  grouped;
- fracción de vistas enviadas a learned;
- RMSE de la política frente a unit, learned y selector estático;
- regret frente al mejor de `unit/learned` por vista, sólo como cota oracle
  diagnóstica;
- tasa de fallos IRLS y una variante de rescate a unit claramente separada.

Como validation sólo contiene IID, este bloque prueba transporte de una señal
pre-solver a un cambio de dependencia. No puede presentarse como evaluación
independiente: test ya fue observado por experimentos anteriores, y la familia
de features fue motivada por ese resultado.

## 8. Diagnóstico de doble reponderación

Después del solve, y fuera de todo routing, se comparan por vista:

- tamaño efectivo del peso base learned, del multiplicador robusto efectivo
  inducido por IRLS con base unit y del peso final con base learned;
- correlación y solapamiento del decil inferior entre confiabilidad aprendida y
  peso IRLS final con base unit;
- masa relativa asignada a aristas causalmente alteradas, usando la máscara
  privada sólo para adjudicación;
- cambio en iteraciones, condición, residual ponderado y RMSE de cociente;
- asociación entre pérdida de tamaño efectivo y
  `Q(observed,learned,IRLS)-Q(observed,unit,IRLS)`.

La comparación debe distinguir peso base, multiplicador robusto efectivo y
peso final. Como el solver preserva el estado reponderado luego de damping y
normalización, el cociente `final/base` sólo identifica un multiplicador
relativo hasta escala global; se normaliza por su media válida y se valida
reconstruyendo `normalize(base*multiplier)` a `1e-12`. No se lo presenta como
el candidato Huber instantáneo de la última iteración.

Una lectura compatible con «doble supresión» exige simultáneamente mayor
solapamiento de colas, caída de tamaño efectivo y asociación con degradación
IRLS. Ninguno de esos diagnósticos por separado establece mecanismo causal.

## 9. Estimandos, bootstrap y fallos

El RMSE de cociente sigue siendo la métrica primaria. Cada contraste usa signo
`variante - referencia`, donde negativo favorece la variante. Se preservan los
`2.000` índices bootstrap por master de la fuente para medias e intervalos
marginales del 95% en test IID, grouped y grouped-minus-IID.

No se reportan p-values ni corrección que sugiera confirmación. Los intervalos
son descriptivos y marginales. Los resultados por brazo preceden a cualquier
promedio de familia; no se crea un score agregado entre brazos.

El promedio sobre las ocho réplicas de shuffle no convierte la aleatoriedad de
permutación en incertidumbre muestral. Se conserva y reporta por separado la
dispersión entre réplicas; el bootstrap por master sólo describe la variación
entre masters del estimando promediado sobre esa familia congelada.

La política estricta de IRLS se conserva: cualquier no convergencia requerida
por un estimando vuelve no evaluable el slice completo. Los supervivientes
finitos, rescates y promedios de shuffle se etiquetan como diagnósticos y nunca
reemplazan el estimando estricto. Excepciones estructurales, rank inválido,
arrays malformados o divergencia respecto de celdas ya atestadas abortan la
corrida.

## 10. Patrones de lectura predeclarados

- **Ubicación informativa:** `learned` mejora frente a `shuffled`, mientras
  `shuffled` se acerca a `unit`; la ventaja reside en la asignación por arista.
- **Concentración dominante:** `learned` y `shuffled` se parecen entre sí y
  ambos difieren de `unit`; la distribución marginal domina la ubicación.
- **Intensidad descalibrada:** un `alpha` intermedio elegido en validation
  mejora a `alpha=1` y conserva el efecto bajo test.
- **Falla de transporte:** el alpha o selector favorable en validation pierde
  frente a unit/learned en grouped, o el predictor público cambia de signo o
  pierde asociación.
- **Doble reponderación compatible:** IRLS learned combina cola superpuesta,
  menor tamaño efectivo y degradación mayor; sigue siendo una hipótesis de
  mecanismo, no una prueba causal.

Un patrón distinto se informa tal cual. No se inventan umbrales de éxito, no se
promueve una arquitectura y GO/NO-GO permanece en manos del usuario.

## 11. Artefactos y presupuesto

El paquete canónico debe conservar:

- `resolved_config.json` y `source_attestation.json`;
- `static_selection.json` y `temperature_selection.json`, escritos antes de
  materializar resúmenes test;
- `public_features.npz`, con schema y whitelist explícitos;
- `raw_solver/<arm>|seed=<seed>.npz`, incluyendo alphas, shuffles, pesos
  finales, convergencia, iteraciones y offsets;
- `per_view_metrics.npz`, `weight_semantics.json`, `transport_diagnostic.json`
  y `effects.json`;
- `bootstrap_indices.npz`, copia byte-exacta;
- `SOLVER_INTERFACE_REPORT.md`, `replay.sh` y `manifest.json`;
- `runtime_observation.json`, fuera del perímetro byte-exacto.

Los NPZ deterministas usan timestamps ZIP fijados y orden estable. El replay
debe reproducir todos los artefactos salvo runtime. El runner no importa
`torch`, oculta CUDA, fija pools numéricos a un thread y aborta si supera `15
min` o `4 GiB` de RSS. La estimación actual es `4–8 min` CPU por corrida; se
ajusta con un smoke reducido antes de la ejecución oficial.

Antes del smoke deben pasar tests focales de: schema/config exactos; hashes y
alineación de ambas fuentes; validación ragged de aristas, nodos y paths;
whitelist pública; invariancia de selección ante mutaciones test/mecanismo;
endpoints alpha `0/1`; shuffles que preservan exactamente el multiconjunto;
folds agrupados; desempates; fallos IRLS; separación strict/rescue; aborto por
alias o ancestro de paths; artefactos deterministas y replay en destino nuevo.

## 12. Cola GPU preservada

Este plan no consume ni reserva GPU. Quedan en cola hasta nueva orden:

1. heads o pérdidas solver-específicas;
2. nuevos seeds y freeze confirmatorio;
3. transferencia a un dominio físico autorizado.

Una señal positiva de este diagnóstico sólo diseña mejor esas pruebas; no las
ejecuta ni las sustituye por otra infraestructura.
