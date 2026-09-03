# Ola 52 — plan de smoke CPU para transporte de política ordinal

> **Estado:** `FROZEN-AUDITED / OPENED-HISTORICAL-DATA / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-03
> **Antecedentes:** `WAVE_50_PROSPECTIVE_CLOSED.md` y `WAVE_51_FACTORED_SET_POLICY_SMOKE_CLOSED.md`

## Pregunta

La Ola 51 mostró que separar una cabeza de conjunto y otra de elección no basta
cuando ambas reciben el mismo target parcial: la segunda cabeza aprende señal,
pero no dispone de una razón externa para preferir un miembro compatible sobre
otro. La Ola 52 introduce esa razón como parte explícita del problema. Pregunta
si una representación set-valued puede reutilizarse frente a políticas
cambiantes y cuánto cuesta extraer una decisión cuando el orden de utilidades
cambia fuera de la distribución de entrenamiento.

El estimando primario es **ingenieril y sistémico**: transporte de una
representación de compatibilidad mediante una regla de composición conocida.
No se atribuirá una eventual ventaja frente a un selector end-to-end a una
representación superior, porque el reader explícito recibe hard-coded la regla
correcta. La contribución del reader se estudia aparte, sobre una misma
representación congelada.

La utilidad de este smoke es contractual y sintética. Autoriza una acción sólo
dentro del banco; no otorga autoridad física a las familias relacionales ni
convierte la preferencia fabricada en propiedad de la naturaleza.

## Régimen de evidencia

Se reutilizan únicamente `train` y `val` históricos ya abiertos del paquete de
Ola 50. No se lee el lockbox. Los inputs y targets de compatibilidad se ligan
por hash a sus manifiestos canónicos antes de cargar datos. El experimento es
de desarrollo y no puede promover una arquitectura ni decidir GO/NO-GO.

La unidad independiente candidata sigue siendo `pair_token`. Antes de ejecutar,
un preflight debe verificar si varios tokens comparten una unidad generativa
superior; si existe, esa unidad pasa a ser el cluster de remuestreo. Un mismo
token puede verse bajo varios contextos de utilidad, pero bootstrap, splits y
tests nunca tratan esas réplicas contextuales como observaciones independientes.

## Autoridad de la decisión

Para las cuatro familias catalogadas se construyen vectores de utilidad
`u in R^4` sin usar `family_id`, parámetros del generador, distancias del oracle
ni targets. Cada contexto es una permutación de cuatro niveles fijos y
distintos, por ejemplo `{1.0, 0.6, 0.2, -0.2}`. Dado el conjunto compatible
autorizado `C`, la acción correcta del contexto es:

`a*(C,u) = argmax_{f in C} u_f`.

Los niveles distintos garantizan unicidad. Los casos sin familia compatible no
entran a la loss de elección; conservan su estatuto de fuera de catálogo y se
reportan aparte. El target de acción se deriva después de firmar el manifest de
contextos y se registra junto con la regla que lo produjo.

El experimento prueba **órdenes ordinales**, no utilidad cardinal general. Las
24 permutaciones se dividen antes de entrenar en tres grupos disjuntos y
exactamente balanceados:

- `policy_train`: 8 órdenes;
- `policy_val`: 8 órdenes para selección de readers;
- `policy_shift`: 8 órdenes nunca usados para optimización o selección.

Cada familia ocupa exactamente dos veces cada rango dentro de cada grupo. Se
usan tres folds predeclarados que rotan los grupos, de modo que las 24 políticas
actúan exactamente una vez como `policy_shift`. Un manifest conserva
permutaciones, folds, asignación de grupo y hashes. El test cruza nuevos
`pair_token` de `val_monitor` con órdenes held-out; desplaza simultáneamente
observación y política sin abrir evidencia sellada.

Los ejes observacional y de política se separan de manera explícita:

- `train pair_token x policy_train` entrena readers aprendidos;
- `val_threshold pair_token x policy_val` selecciona `lambda` y el threshold
  set-valued, cada uno con su objetivo autorizado;
- `val_monitor pair_token x policy_shift` se usa sólo para la lectura final.

`val_threshold` y `val_monitor` son los splits disjuntos por `pair_token` ya
materializados por Ola 51; sus inventarios y hashes se revalidan antes de
ejecutar. Los readers usan presupuestos fijos, sin early stopping ni selección
de checkpoint; ningún threshold o hiperparámetro se selecciona con tokens de
`val_monitor`.

La población primaria es `NEAR_RIVAL` con `|C| >= 2`, porque sólo allí la
utilidad puede cambiar la acción. En el paquete abierto son `297/384` tokens;
los `87` singletons se reportan como control de factibilidad, nunca como
oportunidades decisionales.

## Arquitecturas y brazos

El set representation primario es el checkpoint `sigmoid_only@60` de Ola 51,
con sus tres seeds. Su receta, sampling y BCE ya están congelados; se
re-forwardean train y val para obtener logits donde no fueron persistidos. El
threshold se recalibra exclusivamente en `val_threshold` mediante métricas de
conjunto, sin utilidades, y queda fijo en los tres folds de política.

1. `explicit_set_policy` (**reader primario**): sobre el set checkpoint
   congelado, elige la mayor utilidad dentro del conjunto predicho. Si el
   conjunto queda vacío, usa el miembro con mayor probabilidad de compatibilidad
   y registra el fallback.
2. `learned_reader_same_set`: recibe por familia `[set_logit_f, u_f]` del mismo
   checkpoint congelado y aplica un MLP compartido por familia antes del argmax.
   Es permutation-equivariant bajo permutación conjunta de familias, logits y
   utilidades. Sólo este contraste con `explicit_set_policy` estudia el reader
   manteniendo fija la representación.
3. `direct_contextual_choice`: encoder DeepSets + cabeza de elección
   condicionada por `u`; aprende `a*` end-to-end con cross-entropy. Es un
   comparador de sistema, no una ablación causal de la factorización.
4. `joint_set_contextual_choice`: encoder común, `set_head` y cabeza de elección
   condicionada por `u`; optimiza BCE de compatibilidad más cross-entropy de
   acción. Es otro comparador de sistema con supervisión distinta.
5. `score_composition` (**exploratorio**): sobre el mismo set checkpoint elige
   `argmax_f [log sigmoid(set_logit_f) + lambda * normalized(u_f)]`. `lambda`
   se selecciona únicamente en `policy_val` sobre una grilla congelada. No entra
   al criterio diagnóstico primario.
6. `oracle_set_then_utility`: usa el conjunto compatible verdadero y aplica
   `a*(C,u)`. Es una referencia privilegiada de extracción, no una arquitectura
   deployable ni un techo global.

Controles:

- `utility_ignored`: misma arquitectura y presupuesto que
  `direct_contextual_choice`, pero recibe un vector constante; mide cuánto puede
  resolver la prevalencia de acciones sin contexto;
- `counterfactual_utility_eval`: sobre cada checkpoint congelado y cada token
  policy-sensitive, cambia `u` por otra política held-out que garantiza un
  ganador autorizado distinto. El target cambia de acuerdo con el nuevo
  contrato. Mide dependencia causal de contexto sin contaminar entrenamiento;
- `explicit_context_masked_eval`: aplica `explicit_set_policy` sobre el mismo
  conjunto predicho, pero reemplaza `u` por un orden fijo y conserva el target
  contextual original. Sólo cambia la disponibilidad de contexto; estima cuánto
  resuelve la prevalencia de acciones sin tocar representación ni conjunto.

## Comparabilidad y presupuesto

- tres seeds de desarrollo y tres folds de política;
- para `direct_contextual_choice`, `joint_set_contextual_choice` y
  `utility_ignored`: inicialización, batches, épocas y optimizer steps
  alineados; parámetros totales y entrenables, backprops y un proxy explícito
  de operaciones reportados;
- esos tres brazos usan `60` épocas fijas; `learned_reader_same_set` usa `30`
  épocas fijas sobre la representación congelada; no hay early stopping;
- la BCE del brazo joint se computa una vez por `pair_token` en cada step; la CE
  se promedia sobre sus ocho contextos, evitando multiplicar artificialmente el
  peso del set;
- el threshold del conjunto se elige sólo en `val_threshold` con métricas
  set-valued y sin utilidades; sólo `lambda` se selecciona en `policy_val`;
  `val_threshold` y `val_monitor` deben ser disjuntos por hash;
- el sistema primario promedia logits crudos sobre seeds antes de componer la
  decisión;
- ninguna selección usa el resultado de `policy_shift`.

Las comparaciones entre readers explícitos y aprendidos se presentan como
comparaciones de sistemas con costo medido, no como igualdad ficticia de
capacidad. Las comparaciones con distinta supervisión tampoco autorizan una
atribución causal a la factorización. El único contraste que fija la
representación es `explicit_set_policy` frente a `learned_reader_same_set`.

## Métricas

Sobre `val_monitor`, por `pair_token` y por régimen de política:

- exactitud de acción;
- tasa de acción compatible;
- regret restringido respecto de `a*(C,u)`: si la acción es compatible,
  `(max_{f in C} u_f - u_a) / (max(u)-min(u))`; si es incompatible, `1.25`, una
  penalidad mayor que cualquier regret compatible;
- peor regret entre contextos por token;
- set recall, ancho, incompatibilidad y membership AUC/AP;
- tasa de fallback por conjunto predicho vacío;
- sensibilidad a intervención contrafactual y a enmascaramiento de la utilidad.

Se reportan por separado `policy_seen` y `policy_shift`, cada una de las 24
políticas held-out en su fold, cada seed y cada fold. El estimando primario
agrega las predicciones held-out de los tres folds sobre un panel finito y
exhaustivo de 24 órdenes. Los intervalos se obtienen por bootstrap del cluster
generativo validado en preflight; cada remuestra conserva juntos todos los
contextos y folds de la unidad. El promedio sobre contextos no aumenta
artificialmente `n`.

## Criterio diagnóstico predeclarado

El sistema modular queda **prometedor para transporte de política ordinal**, sin
promoción científica, sólo si en la población policy-sensitive de
`NEAR_RIVAL/policy_shift` el reader primario `explicit_set_policy`:

1. supera a `direct_contextual_choice` en exactitud por al menos `0.03`, con IC
   95% del delta por encima de `0`;
2. reduce regret restringido por al menos `0.03`, con IC 95% por debajo de `0`;
3. no pierde más de `0.01` de tasa de acción compatible;
4. reproduce array-exact los logits del checkpoint `sigmoid_only@60`; la
   conservación de recall, ancho e incompatibilidad es un chequeo de integridad,
   no evidencia nueva;
5. `direct_contextual_choice` supera a `utility_ignored`, y
   `explicit_set_policy` con contexto verdadero supera al mismo reader bajo
   `explicit_context_masked_eval`, por al menos `0.05` de exactitud en cada
   contraste;
6. en `counterfactual_utility_eval`, la acción del reader cambia hacia el nuevo
   ganador autorizado en al menos `0.80` de los casos elegibles.

El patrón se interpreta en conjunto y debe satisfacerlo el mismo reader
primario; `score_composition` no puede rescatarlo post hoc. Aun si se cumple, la
ventaja frente al end-to-end se atribuye a la composición conocida del sistema,
no a superioridad causal de la representación. Si el control ignorando utilidad
ya resuelve la tarea o el shift no cambia la acción, el diagnóstico es
inválido. Los umbrales organizan el smoke; no son GO/NO-GO.

## Invariantes y artefactos

- la utilidad y su split se generan sin leer labels y quedan firmados antes de
  derivar acciones;
- ningún campo sellado entra al input;
- cada grupo de política balancea exactamente posiciones ordinales por familia
  y las 24 políticas aparecen una vez en el shift de los tres folds;
- la acción derivada pertenece al conjunto compatible en todo caso in-catalog;
- todo contrafactual primario cambia efectivamente el ganador autorizado;
- permutar conjuntamente familias, utilidades, logits y targets preserva la
  salida;
- cambiar sólo la utilidad puede cambiar la acción sin modificar logits ni
  métricas del conjunto;
- se guardan checkpoints `last_epoch`, logits crudos de ambas cabezas,
  utilidades, acciones, thresholds, `lambda`, métricas por token, mappings de
  shuffle, manifests y runtime;
- corrida primaria y replay exacto CPU antes de cerrar el smoke.

## Enmienda de implementacion tras auditoria independiente

La auditoria previa a ejecucion detecto ocho desajustes entre el contrato y el
primer runner. La corrida queda habilitada solamente despues de estas
correcciones:

- el panel de politicas, su hash y el snapshot del codigo se fijan antes de leer
  labels autorizados;
- `pair_token` queda materializado como `cluster_id`, porque el generador lo
  crea una vez por realizacion latente antes de emitir las vistas canonicas; los
  splits se prueban disjuntos y el bootstrap remuestrea ese cluster;
- el reader explicito y el reader aprendido reciben exactamente el mismo
  ensemble de logits crudos del set-head; las replicas del segundo miden
  sensibilidad de optimizacion, no representaciones distintas;
- el estimando contrafactual primario es cambiar hacia el nuevo ganador
  autorizado, aunque la accion original haya sido incorrecta; la correccion
  conjunta de ambas acciones se informa por separado y el control se repite por
  checkpoint;
- el peor regret se agrega con un maximo entre las 24 politicas, no promediando
  maximos parciales por fold;
- se preservan logits y scores por seed, split, fold y brazo, acciones derivadas,
  targets, mappings contrafactuales, metricas por token/politica/seed, costos y
  un replay desde checkpoints que debe ser array-exact antes del paquete final;
- los casos fuera de catalogo permanecen fuera del universo modelado heredado
  de Ola 51. No se amplifica silenciosamente el scope para incorporarlos a la
  BCE; se registra su inventario fuente y la exclusion. Las primitivas de
  eleccion continuan fallando cerradas ante un target vacio.

El informe independiente se conserva en
`agent_reports/291_wave52_implementation_independent_audit.md`. Su veredicto
inicial fue `REVISE`; no contiene resultados experimentales porque la auditoria
ocurrio antes de toda corrida.

Una segunda lectura estable (`agent_reports/293_wave52_implementation_stable_reaudit.md`)
encontro tres deudas adicionales, tambien resueltas antes de ejecutar:

- `tau` se selecciona sobre el mismo ensemble de logits ya promediado por
  `pair_token` que consume el reader primario, no sobre fixtures individuales;
- el replay recarga checkpoints, reconstruye scores, deriva nuevamente acciones
  ensemble y por seed y solo entonces recompone las metricas; los shuffles de
  entrenamiento se preservan mediante ordenes de tokens, semillas por batch,
  catalogo ordenado de fixtures y algoritmo de permutacion;
- las salidas por politica incluyen ensemble y cada seed, y los intervalos
  incluyen tambien el peor regret por token.

La reauditoria intermedia 292 queda marcada como invalida porque el snapshot
cambio mientras la instancia lo leia; no se usa como evidencia de correccion.
El unico finding residual de replay se registra en el informe 294 y su
verificacion focal final en el informe 295 dio `PASS` antes de ejecutar.

## Interpretación posible

Un positivo mostraría que una representación de compatibilidad puede soportar
órdenes de decisión variables sin reescribir la geometría aprendida y que la
composición conocida tiene utilidad computacional bajo este contrato ordinal.
No demostraría que la utilidad sea natural, que el set-head haya aprendido una
geometría universal ni que la representación sea causalmente superior al
selector end-to-end.

Un negativo distinguiría al menos tres causas: set insuficiente, política
directa que transporta igual o mejor, o utilidad cuyo efecto no logra atravesar
el error de compatibilidad. Cada causa exige una continuación diferente; no se
resuelve automáticamente con más épocas.
