# Plan CPU — paquete físico prospectivo para la rama set-valued nativa

> **Estado:** `R569-REVISION / OPENED-DATA-PREFLIGHT-ONLY / FRESH-HARD-DISABLED / NO-FRESH-DRAW / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-07
> **Antecedentes directos:** runner set-valued cerrado en R564 y separación
> física de Ola 59 cerrada

## 1. Resultado que debe producir este goal

Este goal debe dejar implementado y auditado el paquete analítico que podría
consumir, más adelante, una realización prospectiva de la comparación
`MARGINAL/JOINT × HARD/CONTEXTUAL`. El paquete no crea esa realización ni abre
un monitor nuevo. Su trabajo es más acotado: convertir la separación lógica
del runner R564 en una separación física verificable, probarla de extremo a
extremo sobre datos históricos ya abiertos y congelar la interfaz que deberá
respetar un preparador prospectivo futuro.

El cierre exige simultáneamente:

1. siete fases analíticas ejecutadas en procesos distintos y con archivos
   permitidos por fase;
2. separación física entre bundles públicos y bundles con truth;
3. estados portables, freezes, journals, receipts y transiciones monotónicas;
4. reinicio desde el último journal válido y replay desde un output vacío;
5. checker independiente y mutaciones adversariales que prueben tanto la
   ciencia como la frontera de acceso;
6. primaria y replay CPU sobre cuatro poblaciones históricas disjuntas;
7. auditoría independiente sin findings altos o medios abiertos;
8. documentación pública y privada coherente con el estatuto de preflight.

El estado máximo de este goal es
`PHYSICAL_PROSPECTIVE_PACKAGE_PREFLIGHT_VALID`. No es evidencia prospectiva,
no habilita por sí solo un draw, no promueve una arquitectura y no contiene
una decisión `GO/NO-GO`.

## 2. Alcance negativo

Durante este goal queda prohibido:

- generar, sortear, consultar o abrir una realización prospectiva fresca;
- leer un monitor o lockbox prospectivo, aunque el runner pudiera reconocer
  en el futuro su schema;
- consultar GPU, importar `torch`, iniciar CUDA o sustituir por CPU una tarea
  que naturalmente requiera GPU;
- entrenar sobre datos nuevos;
- reutilizar los resultados numéricos del preflight como si fueran una prueba
  confirmatoria;
- declarar que `MARGINAL`, `JOINT`, `HARD` o `CONTEXTUAL` ganó;
- convertir `NOT_EVALUABLE` en fracaso científico;
- copiar automáticamente la autoridad de Ola 59: sus invariantes se rebasan,
  pero los schemas y fases de este paquete son propios.

Las referencias históricas cuyos paths contienen `sealed_monitor` sólo pueden
usarse como antecedentes documentales. La corrida canónica de este goal no
los recibe como inputs: el cuarto rol se obtiene del split histórico
`decision_select` de Ola 54, ya abierto.

## 3. Rebase sobre el estado real

### 3.1 Lo que R564 ya resuelve

El runner vigente implementa por CPU:

- posterior `MARGINAL` mediante Platt pooled y posterior `JOINT` con selección
  OOF independiente sobre seis regularizaciones;
- `TARGET-SHUFFLED` con un único derangement por token y estrato;
- reader `HARD` ligado al MAP del set y reader `CONTEXTUAL` con 17 features;
- Ridge para propuesta, logísticas para daño e incompatibilidad y cinco
  controles matched por posterior;
- grilla de 344 políticas, selección lexicográfica, bootstrap por
  `pair_token`, sensibilidades y ocho familias de estimandos;
- estados sin pickle, serialización canónica, replay exacto y checker externo.

La corrida cerrada en R564 usó tres poblaciones disjuntas:

| Rol | Fuente histórica | Filas |
|---|---|---:|
| ajuste de posterior | Ola 54 `calibration_fit` | 192 |
| ajuste de política | Ola 59 `gate_fit` | 768 |
| selección | Ola 59 `gate_select` | 768 |

Su debilidad deliberada era operacional: `select_and_apply()` recibía en un
mismo proceso la vista pública y la truth de selección; luego la evaluación
volvía a ejecutarse sobre esa misma población. Los archivos estaban separados,
pero no existía una barrera de proceso ni un cuarto rol disjunto.

### 3.2 Cuarto rol observado

El bundle de Ola 54 contiene además 192 tokens con rol histórico
`decision_select`. Esos tokens no intersectan con `calibration_fit`,
`gate_fit`, `gate_select` ni con el monitor histórico de Ola 59. En este goal
se renombran `evaluate_fixture` y se usan exclusivamente para probar el último
par `apply → truth`.

Ese uso no les devuelve virginidad científica: fueron abiertos y tuvieron otro
papel en Ola 54. Su autoridad es sólo
`OPENED_DATA_PHYSICAL_PREFLIGHT`. La disjunción prueba la topología del
paquete, no una generalización fuera de muestra.

### 3.3 Qué se hereda de Ola 59

Se conservan como patrones operativos:

- coordinador privilegiado y workers `nobody:nogroup` sin capacidades;
- `no_new_privileges`, grupos suplementarios vacíos y threads acotados;
- stages cerrados por allowlist y hashes;
- root de corrida inaccesible al worker, promoción atómica y outputs
  inmutables;
- un journal durable por transición;
- prueba explícita de paths que cada worker no debe poder abrir;
- validación íntegra antes de reusar una fase;
- archivo recuperable de intentos fallidos;
- replay desde un output vacío y comparación por clase de artefacto;
- presupuestos de wall time y RSS aplicados durante la ejecución, no sólo
  informados al final.

No se heredan `joblib`, modelos HGB, la taxonomía train/validation/monitor ni
los schemas de recovery de Ola 59. La rama set-valued conserva sus estados
NumPy/JSON portables y su propio orden causal.

## 4. Dos clases de preparación, una sola máquina analítica

El coordinador reconoce dos clases, pero en este goal ejecuta una sola.

### 4.1 `OPENED_DATA_PHYSICAL_PREFLIGHT`

Es la clase habilitada ahora. Un preparador local, ejecutado antes de los
workers, extrae exactamente cuatro roles de fuentes cuyos paths y SHA-256 están
congelados. Produce seis bundles:

```text
posterior_fit_truth.npz
policy_fit_truth.npz
decision_select_public.npz
decision_select_truth.npz
evaluate_public.npz
evaluate_truth.npz
```

El preparador emite `opened_fixture_escrow.json`: un inventario root-only de
las fuentes y de los dos pares public/truth. No contiene una clave generativa,
no finge secreto y no se denomina `generation_escrow`. Su propósito es probar
que la máquina analítica liga una identidad de preparación inmutable.

### 4.2 `FRESH_PROSPECTIVE`

En v1 es un valor reservado y **terminalmente deshabilitado**. El parser lo
reconoce para evitar que una extensión futura cambie silenciosamente el
significado del campo, pero el authority preflight devuelve
`FRESH_PROSPECTIVE_NOT_AUTHORIZED_V1` antes de crear el input package, el
output, un stage o un worker. No existe flag, variable de entorno, fixture de
producción ni fallback que convierta un receipt abierto en autoridad fresca.

Este plan no congela todavía trust root, firma, payload, generador, inference,
checkpoints productores, cronología pre-oráculo, conteos prospectivos ni
stopping rule. Por eso sería incorrecto hacer consumible la ruta fresca. El
goal siguiente deberá diseñar y auditar esa preparación; sólo una versión de
schema posterior podrá habilitarla. La v1 implementada en este goal no genera
claves, no firma autoridad, no abre generation escrow y no consume paquetes
prospectivos aunque nominalmente se ajusten a los seis bundles.

La única máquina analítica ejercida y aceptable en v1 es
`OPENED_DATA_PHYSICAL_PREFLIGHT`; su status, receipts y manifests declaran
siempre `prospective_evidence=false`.

## 5. Schemas de datos congelados

Todos los arrays son little-endian canónicos, no-object, C-contiguous después
de carga, finitos cuando corresponda y se cargan con `allow_pickle=false`.
Cada bundle tiene keys exactas; agregar una key es un error, no una extensión
silenciosa. El preparador normaliza strings a anchos fijos antes de hashear:
`pair_token` y `cluster_id` son `<U64`, `design_stratum` es `<U16` y
`split_role` es `<U24`.

### 5.1 Bundles con truth de ajuste

`posterior_fit_truth.npz` y `policy_fit_truth.npz` contienen:

```text
pair_token        <U64[N]
cluster_id        <U64[N]
design_stratum    <U16[N]
cardinality       <i8[N]
ensemble_logits   <f8[N,4]
per_seed_logits   <f8[3,N,4]
target            bool[N,4]
split_role        <U24[N]
```

Los targets son sets no vacíos; `cardinality` pertenece a `{1,2,3,4}` y debe
ser exactamente `target.sum(axis=1).astype('<i8')`. `pair_token` es único,
`cluster_id == pair_token` elemento a elemento y `design_stratum` pertenece
exactamente a `{FAR_RIVAL, NEAR_RIVAL}`. Los roles exactos son
`posterior_fit` y `policy_fit`.

### 5.2 Bundles públicos de decisión y evaluación

`decision_select_public.npz` y `evaluate_public.npz` contienen:

```text
pair_token        <U64[N]
design_stratum    <U16[N]
cardinality       <i8[N]
ensemble_logits   <f8[N,4]
per_seed_logits   <f8[3,N,4]
```

No se admite ninguna key cuyo nombre contenga `target`, `truth`, `oracle`,
`label`, `gain`, `regret`, `harm`, `compatible` o `authorized`.

### 5.3 Bundles truth de decisión y evaluación

`decision_select_truth.npz` y `evaluate_truth.npz` contienen solamente:

```text
pair_token        <U64[N]
target            bool[N,4]
```

El orden, dtype y digest de `pair_token` deben coincidir exactamente con su
compañero público. En ambos pares, `cardinality` público debe ser exactamente
`target.sum(axis=1).astype('<i8')`; los sets son no vacíos y su rango es
`1..4`. El `design_stratum` público conserva el vocabulario cerrado anterior.
El preparador prueba que los cuatro conjuntos físicos de tokens —posterior,
policy, decision y evaluate— son disjuntos dos a dos.

### 5.4 Invariantes comunes de logits y eje de checkpoints

En los cuatro roles, `per_seed_logits` tiene dtype y shape exactos
`<f8[3,N,4]`. Su eje cero se liga, en ese orden, a los checkpoints históricos
`[17,29,43]`; el manifest conserva sus tres IDs/hashes productores ya abiertos
y no permite permutarlos. `ensemble_logits` tiene dtype/shape `<f8[N,4]` y es
bit a bit igual a
`np.mean(per_seed_logits, axis=0, dtype=np.float64)`. Todos los logits son
finitos.

Los checks de estas relaciones corren dos veces: el preparador antes de
publicar y el checker desde los bundles promovidos. La firma o el hash de un
bundle no sustituye la validez de sus relaciones internas. Los reason codes
específicos son `DTYPE_OR_SHAPE_INVALID`, `CARDINALITY_TARGET_MISMATCH`,
`LOGIT_ENSEMBLE_MISMATCH`, `CHECKPOINT_AXIS_INVALID`,
`STRATUM_VOCABULARY_INVALID` y `TOKEN_IDENTITY_INVALID`.

### 5.5 Utilidades y recetas

El catálogo de 24 acciones se materializa como `utilities.npy` a partir del
manifest de política ligado por hash. Nunca se importa `wave52_policy.py`,
porque su import carga `torch`. Las recetas numéricas, seeds, quantiles,
bootstrap, orden de selección y tolerancias son las de la config cerrada por
R564; cualquier cambio requiere un nuevo schema y una auditoría previa.

## 6. Grafo causal de siete workers

El preparador es una fase administrativa del coordinador. Las siete fases
analíticas siguientes corren en procesos distintos.

```text
PREPARED
  └─ POSTERIOR FIT (truth)
       └─ POLICY FIT (truth + posterior states)
            └─ SELECTION PROPOSE (public + states)
                 └─ SELECTION EVALUATE (truth + candidate actions)
                      └─ SELECTION FREEZE (public + states + decision)
                           └─ EVALUATION APPLY (public + frozen policy)
                                └─ EVALUATION TRUTH (truth + frozen actions)
```

### 6.1 `POSTERIOR_FIT`

Recibe sólo config, bindings, freeze de preparación, utilidades y
`posterior_fit_truth`. Ajusta `MARGINAL`, `JOINT` y sus controles
`TARGET-SHUFFLED` con el mismo mapa. Publica:

- handoff: estados portables y manifest de estado;
- auditoría privada: OOF, folds, mapa de shuffle, targets shuffled y
  diagnósticos de convergencia;
- freeze que liga ambas clases de salida.

No recibe policy, decision ni evaluate. Si una receta no puede ajustarse de
forma válida, termina en `NOT_EVALUABLE_POSTERIOR`; ninguna fase posterior se
ejecuta.

### 6.2 `POLICY_FIT`

Recibe `policy_fit_truth`, estados de posterior, config y utilidades. Ajusta
Ridge, guards y cinco familias control por posterior. Publica:

- handoff: feature schema, parámetros portables reales y control, IDs de
  seeds y diagnósticos agregados;
- auditoría privada: design, weights, gain, harm, incompatibility, mappings y
  scores de ajuste por token;
- freeze de política.

El handoff no contiene target, gain, harm, incompatibility, mapping por token
ni scores de entrenamiento. Las fases públicas sólo reciben parámetros
portables. Una clase ausente o un control no materializable produce
`NOT_EVALUABLE_POLICY` y detiene el grafo sin interpretarlo como refutación.

### 6.3 `SELECTION_PROPOSE`

Recibe `decision_select_public`, estados de posterior/política y utilidades.
Calcula por `MARGINAL` y `JOINT`:

- masa real y shuffled;
- MAP set, acciones `HARD`, design y scores de `CONTEXTUAL`;
- las 344 candidatas con acciones y masks de override;
- metadata pública dividida en dos vistas: `selection_key_metadata` contiene
  sólo `candidate_index`, `kind` y los tres cuantiles; `apply_metadata`
  contiene además los thresholds, pero no cruza al evaluator con truth.

Publica un candidate freeze y arrays públicos. No recibe ningún truth bundle.
Los probes deben demostrar que no puede abrir truth de posterior, policy,
decision o evaluate.

### 6.4 `SELECTION_EVALUATE`

Recibe únicamente `decision_select_truth`, acciones candidatas, masks de
override, acciones hard, `selection_key_metadata`, config y utilidades. No
recibe logits, estados, scores, `apply_metadata`, thresholds ni bundles de
otras poblaciones. Exige que tokens, actions, masks y metadata estén ligados
por el mismo candidate freeze.

Recalcula las métricas de cada candidata y aplica la clave lexicográfica
congelada. Publica dos clases:

- decisión mínima de handoff: `selected_index`, las cuatro métricas escalares
  seleccionadas —incluido `authorized_rows`, calculado sólo como
  `override[selected_index].sum()`— y hashes del candidate freeze, actions y
  masks elegidos;
- auditoría privada: métricas completas por candidata y target-aligned raw.

Así, el único proceso que ve simultáneamente truth y candidatas no puede
reajustar modelos ni fabricar una política nueva. La decisión mínima tiene
keys exactas y no puede contener quantiles, thresholds, scores, logits ni
state.

### 6.5 `SELECTION_FREEZE`

Recibe la vista pública de selección, estados portables, candidate freeze,
`apply_metadata` producida antes de truth y la decisión mínima. Recupera el
índice sólo desde la decisión; recupera quantiles y thresholds sólo desde el
candidate freeze target-blind, nunca desde un output del evaluator. Recalcula
scores y thresholds desde cero, exige igualdad exacta con `apply_metadata` y
reaplica la candidata. Actions y masks deben coincidir bit a bit con los hashes
fijados por el evaluator. Luego deriva, sin truth, thresholds y acciones
matched de los cinco controles.

Publica el `selection_policy_freeze`, acciones seleccionadas, masks, thresholds
control y soporte común. No recibe las métricas privadas de selección. El
status `NOT_EVALUABLE_CONTROL_SUPPORT` pertenece a la celda de control; no
detiene el resto de estimandos.

### 6.6 `EVALUATION_APPLY`

Recibe `evaluate_public`, estados portables y `selection_policy_freeze`.
Recalcula scores y aplica las políticas congeladas a una población distinta.
También materializa de forma target-blind:

- masas real y shuffled;
- acciones hard, contextual y matched-control;
- masks de override y soporte común;
- acciones y masas para los tres checkpoints de sensibilidad.

Publica `evaluation_action_freeze` y arrays públicos. No recibe truth de
evaluación ni los resultados privados de selección.

### 6.7 `EVALUATION_TRUTH`

Recibe sólo `evaluate_truth`, utilidades, config y los arrays congelados por
`EVALUATION_APPLY`. No recibe modelos, logits, scores ni thresholds. Recompone
métricas, bootstraps, ocho familias de estimandos, sensitivities, duplicaciones
de celdas y patterns diagnósticos.

El resultado declara siempre:

```text
status = OPENED_DATA_PHYSICAL_PREFLIGHT
scientific_decision = null
architecture_promoted = false
prospective_evidence = false
decision_authority = user
```

## 7. Matriz exacta de stages

Cada stage contiene `phase_request.json` más los archivos enumerados. Ningún
worker recibe la raíz del run ni paths originales por argumento o variable de
entorno.

| Fase | Inputs de stage | Outputs de handoff | Outputs privados |
|---|---|---|---|
| posterior fit | config, bindings, preparation freeze, posterior truth | state manifest, state arrays, fit freeze | OOF, folds, shuffle map/arrays, fit diagnostics |
| policy fit | config, bindings, preparation freeze, policy truth, posterior manifest/arrays/freeze, utilities | feature schema, policy manifest/arrays, policy freeze | fit scores, targets derivados, control maps/arrays |
| selection propose | config, bindings, preparation freeze, decision public, posterior/policy handoffs, utilities | selection-key metadata, apply metadata, candidate actions/masks/scores, candidate freeze | ninguno con truth |
| selection evaluate | config, bindings, preparation freeze, decision truth, selection-key metadata, candidate actions/masks, hard actions, candidate freeze, utilities | decisión mínima sin quantiles/thresholds/state, decision freeze | candidate metrics y raw de evaluación |
| selection freeze | config, bindings, preparation freeze, decision public, posterior/policy handoffs, apply metadata, candidate actions/masks/freeze, decisión mínima/freeze, utilities | selection policy, actions/matches, policy freeze | ninguno con truth |
| evaluation apply | config, bindings, preparation freeze, evaluate public, posterior/policy handoffs, selection policy/freeze, utilities | evaluation scores/actions/masses/sensitivities, action freeze | ninguno con truth |
| evaluation truth | config, bindings, preparation freeze, evaluate truth, evaluation actions/masses/action freeze, utilities | estimands, metrics, bootstrap, report inputs | raw target-aligned y diagnósticos completos |

Los nombres y keys exactos se fijan en constantes compartidas de schema, pero
el checker mantiene una copia propia y falla si la implementación amplía una
allowlist. Los outputs privados quedan root-only y nunca se copian a un stage
posterior.

## 8. Contención física y permisos

La ejecución canónica requiere coordinador efectivo `uid=0`. Antes de cada
worker:

1. el coordinador valida fuentes, preparación y journals existentes;
2. crea un directorio temporal dedicado, sin symlink, bajo una raíz explícita;
3. copia sólo la allowlist al stage;
4. deja stage y padres transitables pero no listables fuera de lo necesario;
5. archivos del stage: `root:root/0444`; directorios: `root:root/0555`;
6. output del worker: `nobody:nogroup/0700`;
7. ejecuta con `setpriv --reuid=65534 --regid=65534 --clear-groups
   --no-new-privs`;
8. congela un thread para BLAS/OpenMP y `CUDA_VISIBLE_DEVICES=''`;
9. captura wall time, RSS máximo, identidad, capabilities, grupos,
   `no_new_privileges`, threadpools, stage hashes y probes;
10. al terminar, el coordinador valida forma y hashes, cambia ownership a
    root, aplica `0444/0555` y promueve por rename atómico.

El comando agrega `--bounding-set=-all --inh-caps=-all --ambient-caps=-all`.
El worker debe observar `CapInh`, `CapPrm`, `CapEff`, `CapBnd` y `CapAmb` en
cero. Cualquier import de `torch`, módulo CUDA visible, threadpool mayor al
contrato, archivo extra, symlink, output no declarado o probe no denegado
invalida la fase.

### 8.1 Probes mínimos por fase

Además de no stagear material prohibido, cada worker intenta abrir paths
canónicos inaccesibles y registra `PermissionError` o `FileNotFoundError`:

- posterior fit: policy truth, decision truth, evaluate truth;
- policy fit: posterior truth, decision truth, evaluate truth;
- selection propose: los cuatro truth bundles;
- selection evaluate: estados de posterior/policy y evaluate truth;
- selection freeze: decision truth, métricas privadas de selección y evaluate
  truth;
- evaluation apply: decision truth, métricas privadas de selección y evaluate
  truth;
- evaluation truth: posterior/policy states, selection truth y cualquier
  output executable de policy.

El receipt hashea el path absoluto del probe, no lo publica en claro. Los
probes son evidencia negativa auxiliar: la prueba principal es la combinación
de raíz `0700`, stage cerrado, identidad sin privilegios y auditoría del código.

### 8.2 Runtime staged y entorno cerrado

El worker nunca se ejecuta desde el worktree. Para cada fase, el coordinador
construye `temporary/runtime/` con una lista exacta:

```text
_proportional_set_valued_phase_worker.py
geometria_proporcional/__init__.py
geometria_proporcional/proportional_set_valued_native.py
geometria_proporcional/wave53_uncertainty.py
geometria_proporcional/wave54_joint_set.py
```

Cada copia se contrasta con el source freeze, queda `root:root/0444`, sus
directorios `0555` y su inventario/hash se incorpora al phase request. No se
copian runner R564, checker, datos ni otros módulos del repositorio.

El subprocess usa `cwd=stage`, `python -s -P`, `PYTHONNOUSERSITE=1` y un único
`PYTHONPATH` que apunta al runtime staged. El environment se construye desde
cero con allowlist cerrada: `PATH`, locale, `PYTHONPATH`,
`PYTHONNOUSERSITE`, `PYTHONHASHSEED`, los cuatro límites de threads,
`CUDA_VISIBLE_DEVICES=''` y un marcador de runtime staged. No hereda variables
de usuario. El receipt registra `sys.path`, los paths y hashes de los cinco
blobs cargados, versiones NumPy/SciPy/sklearn y el environment efectivo no
sensible; coordinador y checker contrastan todo contra el freeze.

## 9. Autoridad de fuentes sin ciclos de hash

La implementación y la config se congelan en un commit exclusivo o de alcance
cerrado. Un archivo posterior
`proportional_set_valued_physical_source_freeze_v1.json` contiene:

- schema y status;
- commit de implementación, que debe ser su padre Git inmediato;
- paths y SHA-256 exactos del plan y de toda su cadena de auditoría —incluido
  el último dictamen efectivo PASS—, config, core NumPy,
  dependencias W53/W54, coordinador, worker y checker;
- hashes de manifest de utilidades y de las cuatro fuentes históricas;
- hashes de receipts históricos que acreditan que esas fuentes ya estaban
  abiertas;
- commit e informe R564, más los SHA-256 de los artifact manifests primario y
  replay que sirven de referencia de paridad;
- clases de ejecución y claims máximos.

La referencia inicial queda nominada, no descubierta por path mutable:

| Evidencia R564 | Identidad congelada |
|---|---|
| commit del informe | `f7ad9227868f83f381ebbc0a8995fefa5a1a272f` |
| SHA-256 del informe | `2221b3938b03728e28133ac0ac5b05918c56b5a62845966293ac113aea4479cb` |
| manifest primario | `a0401834c3958680ef687ad264b8b56a017a8996eaade904873b342319528a39` |
| manifest replay | `2752c103f80ae8655747cf92709fe9462ef753dd282c20f449694a90b6e44039` |

El source freeze copia además, por path y digest, las entradas concretas de
esos manifests que participan en paridad. El checker no confía en que el root
actual conserve identidad sólo porque su nombre coincide.

El commit que introduce ese source freeze modifica sólo ese archivo. El
runner encuentra el último commit que tocó el freeze, exige que su padre sea el
commit de implementación y verifica cada blob y archivo físico. Así se evita
que un archivo intente contener el hash del commit que lo contiene. Commits
documentales posteriores pueden ser descendientes, pero no pueden alterar una
fuente ligada.

La corrida requiere worktree limpio respecto de todos los paths ligados. No
exige que HEAD sea el commit de freeze, sí que éste pertenezca a la historia
lineal de HEAD y que no existan deltas físicos o de blob.

## 10. Preparación, escrow y manifests

La preparación abierta publica un input package inmutable y externo a ambos
outputs analíticos:

```text
immutable_input_package/
  opened_fixture_escrow.json       root:root/0400
  preparation_freeze.json          root:root/0444
  preparation_receipt.json         root:root/0444
  public_manifest.json             root:root/0444
  journals/
    prepare.json                    root:root/0444
  prepared/
    public/                         root:root/0555
      decision_select_public.npz    root:root/0444
      evaluate_public.npz           root:root/0444
    truth/                          root:root/0500
      posterior_fit_truth.npz       root:root/0400
      policy_fit_truth.npz          root:root/0400
      decision_select_truth.npz     root:root/0400
      evaluate_truth.npz            root:root/0400

primary_output/
replay_output/
```

`public_manifest.json` revela hashes, tamaños, schemas, conteos y commitments
de truth, pero no targets. `opened_fixture_escrow.json` enlaza además cada
salida a la fuente histórica exacta y registra que toda la evidencia ya estaba
abierta. Primaria y replay sólo conservan el path canónico, package ID y hashes
del input; nunca lo mueven, archivan, reparan ni reconstruyen. La v1 no tiene
un modo fresco operativo y, por tanto, no existe sustitución por atestación ni
acceso a generation escrow.

El manifest final clasifica cada archivo como:

- `source_snapshot`;
- `public_handoff`;
- `truth_private`;
- `fit_private_audit`;
- `selection_private_audit`;
- `evaluation_private_audit`;
- `freeze`;
- `journal`;
- `access_receipt`;
- `runtime_receipt`;
- `derived_diagnostic`;
- `regenerable_report`;
- `self_reference`.

Cada entrada fija path, bytes, SHA-256, modo, uid, gid, fase de creación y
visibilidad. El inventario recorre tipos y directorios, no sólo archivos.

## 11. Estados, restart y fallos

La única secuencia normal es:

```text
INITIALIZED
PREPARED
POSTERIOR_FIT_COMPLETE
POLICY_FIT_COMPLETE
SELECTION_CANDIDATES_FROZEN
SELECTION_DECISION_FROZEN
SELECTION_POLICY_FROZEN
EVALUATION_ACTIONS_FROZEN
COMPLETE
```

Estados terminales adicionales:

```text
NOT_EVALUABLE_POSTERIOR
NOT_EVALUABLE_POLICY
NOT_EVALUABLE_SELECTION
```

Cada journal fija estado anterior, estado nuevo, inputs, outputs, receipt,
duración, RSS y `maximum_truth_materialized`. No se reescribe un journal.
El coordinador rechaza estados salteados, dos journals para una fase, outputs
sin journal, journal sin outputs, artefactos de una fase futura o hashes que no
coinciden.

`maximum_truth_materialized` usa un enum total, no texto libre:

```text
NONE < POSTERIOR_FIT < POLICY_FIT < DECISION_SELECT < EVALUATE
```

La fase administrativa conserva `NONE`; posterior fit avanza a
`POSTERIOR_FIT`; policy fit a `POLICY_FIT`; selection propose conserva
`POLICY_FIT`; selection evaluate avanza a `DECISION_SELECT`; selection freeze
y evaluation apply conservan `DECISION_SELECT`; evaluation truth avanza a
`EVALUATE`.

La preparación también tiene commit protocol. Se construye en un sibling
`<package>.preparing` del mismo filesystem, escribe y `fsync`-ea cada archivo y
directorio, publica `journals/prepare.json` con transición
`INITIALIZED → PREPARED`, congela permisos, renombra atómicamente al path
canónico y hace `fsync` de su padre. Un `.preparing` parcial nunca es input y
se archiva con inventario antes de un nuevo intento abierto.

Cada fase analítica sigue el mismo orden durable:

1. worker escribe y hace `fsync` de su scratch;
2. coordinador valida inventario y hashes;
3. copia a un sibling `<phase>.pending` dentro del output filesystem;
4. hace `fsync` de archivos y directorios, congela owner/modo y ejecuta
   `os.replace(<phase>.pending, <phase>)`;
5. hace `fsync` del output root;
6. escribe el journal a `journals/.<phase>.pending`, hace `fsync`, lo renombra
   y hace `fsync` de `journals/`.

La tabla cerrada de estados y artefactos es:

| Estado nuevo | Directorio/artefacto requerido | Próximo estado admisible |
|---|---|---|
| `PREPARED` | input package completo + `prepare.json` | posterior complete/not-evaluable |
| `POSTERIOR_FIT_COMPLETE` | `posterior_fit/` + journal + fit freeze | policy complete/not-evaluable |
| `POLICY_FIT_COMPLETE` | `policy_fit/` + journal + policy freeze | candidates frozen |
| `SELECTION_CANDIDATES_FROZEN` | `selection_propose/` + journal + candidate freeze | decision frozen/not-evaluable |
| `SELECTION_DECISION_FROZEN` | `selection_evaluate/` + journal + decision freeze | policy frozen |
| `SELECTION_POLICY_FROZEN` | `selection_freeze/` + journal + policy freeze | evaluation actions frozen |
| `EVALUATION_ACTIONS_FROZEN` | `evaluation_apply/` + journal + action freeze | complete |
| `COMPLETE` | `evaluation_truth/` + journal + estimand freeze | ninguno |
| `NOT_EVALUABLE_POSTERIOR` | sólo `posterior_fit/not_evaluable.json`, freeze y journal | ninguno |
| `NOT_EVALUABLE_POLICY` | fases previas + sólo `policy_fit/not_evaluable.json`, freeze y journal | ninguno |
| `NOT_EVALUABLE_SELECTION` | fases previas + sólo `selection_evaluate/not_evaluable.json`, freeze y journal | ninguno |

`--resume` opera sólo sobre el mismo package ID y la misma preparación:

- revalida desde `PREPARED` todos los manifests y journals;
- reutiliza una fase únicamente si inputs y outputs son hash-idénticos;
- continúa desde la primera fase completamente ausente;
- no repara ni sobrescribe una fase divergente;
- ante scratch parcial sin promoción, lo mueve a
  `failures/<phase>.<timestamp>/` con inventario y failure receipt;
- ante un output promovido sin journal o con hash divergente, archiva el run
  output completo de forma recuperable y exige reinicio desde output vacío;
- nunca archiva ni mueve `immutable_input_package`; el run nuevo vuelve a
  ligarse al mismo package ID y hashes.

Las pruebas de recovery inyectan fallos después de cada una de las siete
promociones y en el intervalo promoción/journal. La referencia final debe ser
byte-exacta en artefactos científicos y funcionales.

## 12. Serialización, replay y comparación

Se conserva la serialización canónica R564:

- JSON UTF-8, keys ordenadas, separadores compactos, sin NaN/Inf y newline;
- NPZ con keys ordenadas, `allow_pickle=false`, NPY canónico, ZIP DEFLATED 9 y
  timestamp `1980-01-01`;
- nada de pickle ni joblib;
- orden de tokens y candidatos fijado antes de serializar.

Primaria y replay parten de outputs vacíos y consumen el mismo freeze de
preparación. Deben coincidir byte a byte en `public_handoff`, todas las clases
privadas científicas, freezes y diagnósticos. Se excluyen explícitamente:

- tiempos y RSS;
- PID, timestamps y paths absolutos hasheados de probes;
- receipts operativos de primaria/replay;
- manifest autorreferencial y reporte renderizado.

Para cada exclusión existe una comparación semántica exacta de keys, estados,
conteos y límites. No se permite una exclusión por directorio completo.

La comparación adicional con R564 cubre hasta la selección sobre los tres
roles compartidos. No admite un `allclose` global:

| Objeto | Comparador |
|---|---|
| folds, donor maps, clases, índices y selected index | `np.array_equal` / entero exacto |
| coeficientes, intercepts, masas, scores y candidate metrics | bits float64 exactos (`np.array_equal`) |
| thresholds y métricas escalares | valor JSON exacto y misma representación finita |
| candidate metadata target-blind | keys, orden y valores exactos |
| 344 candidate actions y override masks | `np.array_equal` |
| actions seleccionadas y matched | `np.array_equal` |
| hashes de estados/acciones | string exacto |

Si un cambio de serialización impide comparar archivos completos, se compara
cada path/key con la tabla anterior; no se relaja la tolerancia. La evaluación
sobre el cuarto rol no se compara con el resultado científico de R564 porque
usa otra población; se verifica por recomposición independiente.

## 13. Checker independiente

El checker físico no importa el coordinador, el worker ni
`proportional_set_valued_native.py`. Puede reutilizar las fórmulas del checker
R564 como fuente independiente, siempre que esa dependencia quede ligada por
hash y que el entrypoint físico agregue por cuenta propia permisos, stages,
journals, manifests, estados y recovery.

Checks mínimos:

| ID | Condición | Reason code |
|---|---|---|
| `P1_AUTHORITY` | source freeze, Git, config, clase abierta y rechazo duro de fresh v1 | `AUTHORITY_INVALID` |
| `P2_PREPARATION` | seis bundles, invariantes cruzados, disjunción, escrow abierto y manifests | `PREPARATION_INVALID` |
| `P3_PHYSICAL_BOUNDARY` | owners, modos, types, runtime staged, env, modules, stage allowlists, UID/cap sets/NNP y probes | `PHYSICAL_BOUNDARY_INVALID` |
| `P4_STATE_MACHINE` | journals, transiciones, verdad máxima y outputs por fase | `STATE_MACHINE_INVALID` |
| `P5_POSTERIOR` | marginal, joint, OOF, refit y shuffle recompuestos | `POSTERIOR_INVALID` |
| `P6_POLICY` | features, Ridge, logísticas y controles recompuestos | `POLICY_INVALID` |
| `P7_SELECTION_PROPOSE` | masas, MAP, scores y 344 candidatas target-blind | `SELECTION_PROPOSE_INVALID` |
| `P8_SELECTION_EVALUATE` | métricas, clave, decisión mínima y ausencia de state | `SELECTION_EVALUATE_INVALID` |
| `P9_SELECTION_FREEZE` | reapply bit-exacto, thresholds, matching y soporte | `SELECTION_FREEZE_INVALID` |
| `P10_EVALUATION_APPLY` | acciones, masas y sensitivities target-blind | `EVALUATION_APPLY_INVALID` |
| `P11_EVALUATION_TRUTH` | raw, bootstrap, ocho estimandos, precedencia y patterns | `EVALUATION_TRUTH_INVALID` |
| `P12_RESTART_REPLAY` | recovery y comparación primaria/replay | `RESTART_OR_REPLAY_INVALID` |
| `P13_INVENTORY` | inventario, visibilidad, serialización y hashes | `INVENTORY_INVALID` |
| `P14_SCOPE` | CPU, sin torch, no fresh, no promoción, decisión nula | `SCOPE_INVALID` |
| `P15_COST` | wall, RSS y disco dentro de budgets | `COST_INVALID` |

Cada check informa observación, evidencia y reason code. El checker no confía
en `REPORT.md`, freezes declarativos ni receipts sin contrastarlos con bytes y
recomposición.

## 14. Mutaciones adversariales

Cada caso cambia un solo elemento y debe producir el reason code previsto. La
suite cubre, como mínimo:

### Autoridad y preparación

- fuente o blob Git divergente;
- freeze cuyo commit padre no coincide;
- execution class renombrada o evidencia abierta presentada como fresca;
- bundle extra, key extra, symlink, hardlink externo, FIFO o directorio vacío
  no declarado;
- overlap, duplicado o reordenamiento distinto entre public/truth;
- truth key dentro de public;
- escrow abierto que se hace pasar por generation escrow;
- cualquier `FRESH_PROSPECTIVE`, con o sin firma o commitment, que no produzca
  `FRESH_PROSPECTIVE_NOT_AUTHORIZED_V1` antes de crear paths;
- cardinality/target, ensemble/per-seed, checkpoint axis, stratum, dtype o
  shape internamente inconsistentes.

### Frontera física

- worker con uid/gid incorrecto, cualquier set de capability no nulo, grupo
  suplementario o `no_new_privileges=0`;
- runtime blob cambiado, módulo resuelto desde worktree/user-site, `sys.path`,
  cwd o environment ampliados;
- stage con archivo extra/faltante, hash stale o modo writable;
- truth entregada a una fase pública;
- state ejecutable entregado a selection evaluate o evaluation truth;
- métricas privadas de selección entregadas a selection freeze;
- probe exitoso, probe omitido o receipt fabricado;
- `CUDA_VISIBLE_DEVICES` no vacío, import de torch o threadpool excedido.

### Ciencia y causalidad de fases

- posterior marginal o joint alterado;
- shuffle diferente entre representaciones;
- policy handoff que filtra gain/harm/target/mapping;
- candidata omitida, metadata reordenada o métrica calculada en el proposer;
- selector que recibe scores/modelos o freeze que no reaplica;
- acción de selección modificada después de ver truth;
- applier de evaluación que recibe truth o cambia thresholds;
- evaluator que recibe modelos o vuelve a generar acciones;
- soporte matched por unión, cantidad variable de controles o promedio con
  controles faltantes;
- bootstrap, orientación, penalty, utility, precedencia o pattern alterado;
- status `NOT_EVALUABLE` reinterpretado como refutación.

### Recovery, replay e inventario

- transición salteada, journal reescrito, output futuro o output sin journal;
- resume con otro package ID, config, source freeze o preparación;
- falla después de cada promoción y entre promoción/journal;
- replay divergente en un artefacto incluido o exclusión demasiado amplia;
- manifest que omite un raw privado, modo/owner incorrecto o disco no
  contabilizado;
- reporte que declara evidencia prospectiva, promoción o decisión científica.

La suite puede usar fixtures pequeños para mutaciones de permisos y schemas.
Las pruebas matemáticas conservan además los casos unitarios R564: MAP distinto
de threshold marginal, tie-breaks, gradiente JOINT, estados portables,
assignment óptimo, estabilidad por reordenamiento y soporte común exacto.

### 14.1 Receipts de campaña y manifest de evidencia

La aceptación no depende de stdout. Antes de la auditoría final deben existir:

```text
evidence/
  unit_test_receipt.json
  primary_check_receipt.json
  replay_check_receipt.json
  mutation_receipt.json
  recovery_receipt.json
  evidence_manifest.json
```

Cada receipt fija schema, source-freeze SHA-256, comando/argv, inputs y outputs
por hash, versión de Python/NumPy/SciPy/sklearn, exit observado, wall time, RSS
y bytes temporales/preservados. Los receipts de checker conservan las 15 filas
con status y reason code. El de mutaciones conserva una fila por caso con
mutación única, predicate esperado, reason code esperado/observado y exit. El
de recovery conserva cada crash point, estado previo, inventario fallido,
acción de recuperación y comparación final con la referencia limpia.

`evidence_manifest.json` excluye sólo su propio archivo y liga todos los demás
por path, bytes y SHA-256, además de los manifests de input, primaria y replay.
El checker tiene un modo final `--evidence` que valida esta cobertura después
de que los checks individuales hayan producido sus receipts; no intenta hacer
que un receipt se autentique circularmente a sí mismo. Los crudos fallidos
reutilizables se preservan por inventario, pero las copias regenerables de
fixtures de mutación pueden retirarse después de hashear y registrar el caso.

## 15. Presupuestos CPU y disco

La corrida canónica usa un thread por worker. Límites duros iniciales:

| Recurso | Límite |
|---|---:|
| primaria + replay, wall | 1 800 s |
| checker + mutaciones | 1 200 s |
| unit tests + fixtures de permisos | 600 s |
| campaña completa de recovery | 1 800 s |
| RSS máximo por coordinador/worker | 1.5 GiB |
| disco primario + replay | 512 MiB |
| scratch agregado de tests/mutaciones/recovery | 1 GiB |
| evidencia preservada después de cleanup | 128 MiB |
| archivo individual | 128 MiB |

Estos límites son holgados frente a R564 —aproximadamente 56.4 s combinados,
986 MiB RSS pico y 26.2 MiB para primaria+replay—, pero incluyen siete
subprocesos, receipts privados y recuperación. El coordinador aplica deadline
y muestreo de RSS a cada fase y al total. Los harnesses miden además wall/RSS y
disco agregado de su campaña, incluidos `failures/`, antes del cleanup; el
evidence receipt conserva ambos tamaños. Una excedencia invalida el preflight;
no se relanza con más threads ni se consulta GPU.

No se ejecutan esperas largas ni barridos. Si la implementación revela una
operación cuya versión CPU excedería materialmente este régimen, se la deja en
cola; no se convierte este goal en un entrenamiento.

## 16. Superficie de implementación

Después de aprobar este plan se crearán, como mínimo:

```text
experiments/geometria_proporcional/
  run_proportional_set_valued_physical_preflight.py
  _proportional_set_valued_phase_worker.py
  check_proportional_set_valued_physical_preflight.py
  configs/proportional_set_valued_physical_preflight_v1.json
  configs/proportional_set_valued_physical_source_freeze_v1.json
tests/
  test_proportional_set_valued_physical.py
  run_proportional_set_valued_physical_mutations.py
```

Se puede agregar un módulo de schema puro si evita duplicación entre
coordinador y worker; el checker no lo importa. El core R564 se modifica sólo
si hace falta una operación matemática reusable y esa modificación conserva
todos sus tests. No se edita el runner R564 para simular aislamiento.

Los artefactos pesados quedan ignorados bajo:

```text
data/geometria_proporcional/
  proportional_set_valued_physical_input_v1/
  proportional_set_valued_physical_preflight_v1/
  proportional_set_valued_physical_preflight_replay_v1/
  proportional_set_valued_physical_evidence_v1/
```

Los crudos de auditoría se archivan verbatim en `Biblioteca/`.

## 17. Orden de construcción y condición de aceptación

1. auditar este plan con una instancia independiente;
2. resolver findings y reauditar hasta no dejar altos o medios abiertos;
3. implementar schemas, preparador abierto, coordinador, workers y checker;
4. ejecutar unit tests y fixtures de permisos;
5. ejecutar primaria, checker, replay y comparación;
6. inyectar fallos de restart y correr mutaciones adversariales;
7. auditar independientemente código, artefactos, receipts y claims;
8. corregir findings válidos y repetir las corridas afectadas;
9. congelar evidencia durable y propagar documentación/wiki;
10. reauditar el cierre documental.

El goal se considera técnicamente completo sólo si:

- los 15 checks pasan en primaria y replay;
- todas las mutaciones son detectadas por el reason code esperado;
- recovery converge al mismo resultado funcional que una corrida limpia;
- no se abrió ni creó un draw, monitor o lockbox prospectivo;
- no se usó ni consultó GPU y `torch` no apareció en los procesos;
- la auditoría final queda limpia de findings altos y medios;
- `scientific_decision` continúa `null` y
  `architecture_promoted=false`.

## 18. Siguiente goal automático previsto

Si este paquete cierra, el siguiente goal no abre automáticamente un draw. Debe
ser un gate finito de readiness prospectivo que audite la interfaz del
preparador faltante, estime costo del draw y determine qué autoridad externa,
conteos y criterios de parada deben congelarse antes de materializarlo. Sólo si
esa evidencia muestra que la realización fresca es el próximo contraste con
mayor poder diagnóstico se redactará su protocolo específico.

La promoción de una arquitectura y cualquier `GO/NO-GO` permanecen bajo
decisión de Mariano.
