# Ola 59 — auditoría independiente del plan prospectivo de incompatibilidad y cola

## Dictamen: REVISE

El diseño tiene un próximo contraste científicamente pertinente: transporta a
un draw fresco dos políticas HGB/HGB preespecificadas, elimina la falsa elección
entre `JOINT` y `SEQUENTIAL`, comparte la propuesta entre los dos brazos
principales y mantiene nulas tanto la decisión científica como la promoción
arquitectónica. Los cuantiles declarados se corresponden con las políticas
abiertas que dice transportar y el cálculo histórico sobre Wave 57 confirma la
rareza y el máximo Hamming ponderado de incompatibilidad.

No obstante, el draft todavía no debe congelarse ni crear claves. Hay dos
problemas altos de autoridad: `CALIBRATE` recibe actualmente un objeto
validation que, bajo el preparador real, contiene truth y gain antes de congelar
thresholds; y el mínimo de soporte monitor contradice la regla que declara
evaluable una identidad transportada. Además, el factorial no identifica el
efecto del target separado de su cuantil, el control max-Hamming no es una
permutación nula exchangeable, y faltan contratos ejecutables de fases, estados,
estimandos, shards y replay. Todos son reparables antes del draw.

No hay findings P0. Los findings P1 y P2 siguientes requieren corrección y
reauditoría.

## Corpus y estado real revisados

Se leyó completo el plan Wave 59 y se contrastó con los planes y cierres Wave 57
y Wave 58, R414, las configs vigentes, la implementación de preparación y
ejecución, y los artefactos primario/replay de Wave 58. El `HEAD` observado fue
`0a2dc2ce2893910dde005a38ada2de3206852b46`; no había cambios tracked al iniciar
la auditoría.

| Artefacto | SHA-256 |
|---|---|
| `WAVE_59_FRESH_HGB_GUARD_BRACKET_PLAN.md` | `0f7ea220b4a39405f048966098d93a16afc314e3bf96dffa57753da38796eef2` |
| `WAVE_57_CONTEXTUAL_TAIL_GUARD_PLAN.md` | `db0d69ba197bfa0dcf853c9cecc1f848487ccc25910424036ebcde745a1ccedd` |
| `WAVE_57_CONTEXTUAL_TAIL_GUARD_CLOSED.md` | `1fecfb50172c130b2e627dd44cb869f338441c53f5c53ccbfeec49bd08120fb4` |
| `WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_PLAN.md` | `d7fbbad633a3f03d37de46ad505dbcd330ac9ffa5544838ea2c4707a3ac6ba69` |
| `WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_CLOSED.md` | `bbe860e63cf2a39a5ab5ee0f4609ca0d2832a68ab0e773b1e2b097b1aad599d1` |
| R414 | `0c47e034ff2e3c54bec13fa50d6ea9010c45efe9bceb7a318919f75057327b7b` |
| config Wave 57 | `fb21a43cb4e356a7f10c293e5cb0c0fc037a8ecc206883e99122115b18812a83` |
| config Wave 58 | `1f822c1161a5da4edf5039d6ad464d9ee1a39cdde69c7226460def2b4c854864` |

No existe todavía config, módulo, runner ni test Wave 59 bajo
`experiments/`, `src/` o `tests/`. Eso es compatible con el estado
preimplementación, pero vuelve necesario que el plan defina con precisión qué
se extiende y qué se reemplaza del runtime anterior.

## Comprobaciones factuales favorables

### Los dos brazos principales y sus cuantiles están trazados correctamente

El artefacto Wave 58 confirma que
`C-HGB-HGB-HARM-SEQUENTIAL` usó `q_p=0.8`, `q_g=0.7`, proposer threshold
`0.37786739818476994` y guard threshold `0.48806953128224845`
(`analysis.json:407-470`). Su delta monitor contra hard fue regret
`-0.0128449528` y worst regret `-0.0234204793`, con IC95 superior de worst
regret `-0.0032611656` (`analysis.json:408-445`).

El nominado `C-HGB-HGB-INCOMPATIBILITY-JOINT` usó el mismo proposer
`q_p=0.8` y threshold, con `q_g=0.9` y guard threshold
`0.05031062804095663` (`analysis.json:1083-1146`). R414 demostró que su
versión `SEQUENTIAL` era la misma política byte/array-exacta y que `JOINT` sólo
ganó por el ID (`R414:96-100`). Wave 59 hace bien en congelar la política y no
el nombre del selector.

Sobre validation Wave 58, la propuesta común tuvo `209` filas y `47` tokens.
Con los cuantiles transportados, harm HGB autorizó `146` filas/`35` tokens e
incompatibility HGB `188` filas/`45` tokens. Por tanto, la igualdad de máscara
proposer declarada en el draft es factual; las autorizaciones no son iguales y
no deberían tratarse como si fueran un mismo presupuesto de intervención.

### La rareza histórica de incompatibilidad y el máximo Hamming son correctos

Se recalcularon los targets desde
`wave57.../fit.complete/analytics.complete/gate_fit_bundle.npz`, restringiendo a
`primary & disagreement`, estratificando por
`(policy_index, disagreement_count)` y usando pesos `1/d_t`:

| Target | Filas activas | Positivas | Tokens con positiva | Prevalencia ponderada | Máximo Hamming global | Máximo Hamming ponderado |
|---|---:|---:|---:|---:|---:|---:|
| harm | 978 | 651 | 111 | 0.659661 | 0.417178 | 0.345198 |
| incompatibility | 978 | 26 | 11 | 0.023742 | 0.044990 | 0.034239 |

El `0.02374` y `0.03424` del plan son correctos. Los mínimos propuestos para
incompatibility (`20` filas y `8` tokens) quedan cerca, pero por debajo, del
único draw abierto de referencia; no garantizan evaluabilidad en el draw fresco
y el draft correctamente prohíbe redibujar si fallan.

### La frontera de claims es adecuada

El plan no reutiliza Wave 58 como baseline prospectivo, no vuelve a seleccionar
entre IDs, no promedia media y cola en un escalar, separa los patrones de harm e
incompatibility y conserva `scientific_decision=null`. Eso respeta la lectura de
R414: Wave 58 sólo localizó una interacción HGB/HGB acotada, no una victoria
global de HGB (`R414:108-114`).

## Findings priorizados

### P1 — el threshold de validation no queda físicamente congelado antes de abrir truth

El draft promete que ninguna métrica validation elegirá una celda y que
`CALIBRATE` sólo materializará cuantiles fijos (`plan:30-33,97-101`). Sin
embargo, su separación física entrega “validation” a `CALIBRATE`
(`plan:191-192`) sin tipar un bundle truth-free.

Ese detalle es material frente al runtime real. El bundle analítico vigente
contiene, entre otros, `target`, `gain`, `hard_actions`, `posterior_actions`,
`primary` y `disagreement`. El loader Wave 58 exige explícitamente `target` y
`gain` (`run_wave58_open_diagnostic.py:220-228`), y su fase SELECT carga el
bundle completo y la utilidad antes de construir las políticas
(`run_wave58_open_diagnostic.py:455-478`). Un worker que recibe ese objeto puede
calcular métricas outcome-aware antes de escribir el freeze. Auditar que el
código no lo hizo sería posible, pero la ausencia de selección quedaría
declarativa, no observable por frontera de acceso.

**Reparación obligatoria.** Cambiar la máquina a:

```text
PREPARE -> FIT -> CALIBRATE-SCORES -> VALIDATE -> ADJUDICATE
```

- `FIT` recibe train con truth y ajusta modelos/shams.
- `CALIBRATE-SCORES` recibe sólo validation inference-safe: design o scores,
  `pair_token`, `primary`, `disagreement`, acciones hard/posterior y metadatos
  estrictamente necesarios. Debe rechazar `target`, `gain`, `harm`,
  `incompatibility`, oracle, regrets, métricas y cualquier utilidad capaz de
  reconstruirlos. Calcula scores, thresholds, propuestas, autorizaciones,
  acciones y soportes; luego sella `calibration_freeze.json`.
- `VALIDATE` verifica ese freeze y recién entonces materializa truth validation
  para métricas diagnósticas, sin poder alterar thresholds, máscaras o acciones;
  publica un freeze propio.
- `ADJUDICATE` abre monitor sólo después de verificar los tres freezes.

Los tests deben inspeccionar allowlists negativas, no sólo ausencia de uso en el
código. La frase de `plan:187-188` también debe reemplazarse: los outcomes no
pueden estar ausentes de todas las fases futuras porque FIT necesita truth de
train; deben ser visibles sólo en la fase y split autorizados.

### P1 — el soporte monitor tiene dos semánticas terminales incompatibles

El plan exige al menos `25` tokens autorizados en validation **y monitor**
(`plan:204-208`) y repite ese requisito dentro de ambos patrones
(`plan:246,255`). Tres líneas después declara que cero acciones al monitor es un
resultado evaluable (`plan:210-213`). Las dos reglas no pueden conducir una
máquina determinista sin decidir si `0..24` significa resultado evaluable,
condición `false`, condición `NOT_EVALUABLE` o cierre terminal.

Además, el plan dice que el monitor es la única adjudicación prospectiva
(`plan:217-220`) pero no especifica si las condiciones de `plan:239-255` se
evalúan en validation, monitor o ambos.

**Reparación obligatoria.** Separar:

1. soporte validation `>=25` como mínimo truth-free de CALIBRATE que puede cerrar
   premonitor;
2. soporte monitor como condición descriptiva del patrón, no como permiso de
   abrir ni como evaluabilidad de las métricas;
3. una política con cero overrides monitor como identidad evaluable, con deltas
   contra hard exactamente cero y condición de soporte `false`;
4. todos los patrones científicos calculados sólo sobre monitor; validation se
   reporta como diagnóstico posterior al freeze.

Debe fijarse también la agregación ternaria: cuándo un patrón agregado es
`true`, `false` o `NOT_EVALUABLE` si un sham falta aunque otras condiciones sean
falsas. Wave 57 eligió conservar `null` ante un contraste sham no evaluable; no
debe quedar a criterio del runner.

### P1 — el plan no es ejecutable mediante el preparador vigente sin un contrato nuevo

El preparador compartido no es genérico. `validate_prospective_config` sólo
acepta los schemas Wave 56 y Wave 57
(`prepare_wave56_fresh.py:446-455`) y exige exactamente
`train->gate_fit`, `val->gate_select`, `lockbox->sealed_monitor`
(`prepare_wave56_fresh.py:461-466`). El coordinador vigente sólo reconoce las
fases `fit`, `select` y `adjudicate`; Wave 59 introduce `CALIBRATE` —y, tras la
reparación anterior, necesita además `VALIDATE`— sin definir adaptador, schema,
paths, estados ni allowlists.

También hay una diferencia concreta en no-redraw: el preparador real crea y
sella **tres** claves distintas de 32 bytes, no una única “clave de generación”
(`prepare_wave56_fresh.py:772-788,2410-2421`). Son las claves de identidad,
compromiso semántico y generación; el replay debe reutilizar el mismo triplete,
no crear otro.

**Reparación obligatoria.** Agregar al plan el inventario canónico de config,
wrappers, worker analítico, coordinador, tests y outputs Wave 59. Declarar si se
extiende el preparador compartido o se crea una versión nueva; en ambos casos,
especificar:

- schema Wave 59 y validación pre-escrow;
- mapping físico `train/val/lockbox` y mapping de fases;
- máquina de estados completa y terminales por fase;
- triplete de claves dibujado una sola vez, escrow durable y replay con el mismo
  triplete;
- recuperación pre-oráculo con el mismo escrow, y distinción entre error de
  implementación e inevaluabilidad científica;
- output primario/replay únicos y nombres de todos los freezes.

Sin esa decisión de diseño, una implementación puede degradar la frontera
prospectiva al intentar reutilizar un SELECT outcome-aware.

### P2 — el factorial identifica paquetes target+cuantil, no el efecto del target

Los ocho brazos cruzan proposer, clase de guard y target (`plan:122-128`), pero
harm usa siempre `q=.7` e incompatibility `q=.9` (`plan:89-95`). Por tanto, un
contraste harm−incompatibility cambia simultáneamente label, modelo ajustado y
presupuesto de autorización. En Wave 58, sobre la misma propuesta HGB, esos
cuantiles autorizaron `146` frente a `188` filas. La diferencia no puede
atribuirse sólo a qué riesgo se aprendió.

El factorial sí permite comparaciones contractuales Ridge/HGB y Logistic/HGB
dentro de un target y cuantil fijos, aunque “clase de modelo” sigue significando
el contrato completo —incluido scaler para lineales y features crudas para HGB—,
no no-linealidad aislada. El selector no se identifica ni se prueba en Wave 59;
eso está bien y debe decirse literalmente.

**Reparación.** Elegir una de dos formulaciones:

- si el objetivo es transporte de políticas, denominar cada tratamiento
  `target+q+model-contract`, retirar “atribución del target” y limitar el
  factorial a contrastes simples preenumerados dentro de target;
- si se quiere identificar target, cruzar ambos targets con ambos cuantiles
  `{.7,.9}` —o fijar un único presupuesto de autorización— y preregistrar los
  contrastes target a cuantil común y la interacción target×cuantil.

No se debe usar una celda factorial favorable como candidata post-hoc: todas son
diagnósticos y sólo los dos brazos principales tienen patrones prospectivos.

### P2 — max-Hamming es un control de desplazamiento máximo, no una permutación nula exchangeable

La construcción `2*min(n0,n1)` es matemáticamente correcta para maximizar
Hamming conservando prevalencia por estrato. También garantiza que **todas** las
posiciones de la clase minoritaria cambien. Esa condición selecciona una cola
extrema del espacio de permutaciones respecto del target observado; las cinco
réplicas no son draws uniformes de la distribución nula condicional. En
incompatibility, las `26` positivas históricas quedan obligatoriamente
desplazadas; las seeds sólo eligen receptoras negativas. El contraste puede
usarse como stress-test adversarial de señal destruida, pero no como estimación
de un null exchangeable ni como atribución causal por sí solo.

Además, “fracción permutable” (`plan:168-170`) no tiene fórmula. Wave 57 la
definió como la fracción de filas activas que pertenecen a estratos con tamaño
mayor que uno (`wave57_tail_guard.py:247-280`), que en el FIT abierto es
`0.950920`. Si se interpreta razonablemente como fracción en estratos con ambas
clases, el valor histórico para incompatibility es sólo `0.326176` y el mínimo
`.80` fallaría. Tampoco se dice si seeds `59031–59045` gobiernan sólo el mapping
PCG64 o también el `random_state` del HGB sham; para capacidad igualada deberían
gobernar el mapping y conservar el contrato HGB del guard verdadero.

**Reparación.** Definir cada numerador, denominador, máscara activa, tolerancia y
seed. Preservar los max-Hamming como controles de desplazamiento máximo es
válido si se los etiqueta así y se limita su inferencia. Si el patrón pretende
un control nulo, agregar una familia independiente de permutaciones uniformes
condicionales, sin rechazo por Hamming realizado, y tratar la variación de
mapping como parte del estimando o declarar explícitamente que el IC condiciona
en las cinco seeds. En ambos casos, el promedio debe ser:

```text
sham_metric_t = (1/5) * sum_s metric_t(action_sham_s)
```

nunca métrica de scores, probabilidades o acciones previamente promediadas.

### P2 — shards y mínimos heredados no tienen una operación Wave 59 definida

“Se conservan los mínimos Wave 57” y “dos shards” (`plan:204-205,232-233`) no
especifican el contrato aplicable al roster fijo. Wave 57 reejecutaba dentro de
cada shard una **selección** completa de proposer y guard; Wave 59 ya no tiene
selección. Falta decidir si cada shard:

- aplica los thresholds del validation completo sin recalibrar, para medir
  estabilidad de la política congelada; o
- recalcula cuantiles dentro del shard, para medir estabilidad del procedimiento
  de calibración.

Son estimandos distintos. Deben fijarse el salt y función de asignación, los
conteos mínimos exactos, qué brazos/shams se shard-ean, qué deltas/signos se
reportan y que ningún resultado de shard altera thresholds ni apertura del
monitor. Si se hereda Wave 57, la referencia exacta es LSB de
`SHA256(pair_token || "wave57-shard")` y los mínimos están en
`wave57_contextual_tail_guard_fresh.json:115-160`; conviene usar un salt Wave 59
nuevo y congelado para evitar una dependencia nominal innecesaria.

### P2 — estimandos, IC y patrones requieren fórmulas ejecutables

La unidad `pair_token`, el pairing y la separación validation/monitor están bien
(`plan:215-233`), pero faltan elementos que Wave 57/58 sí congelaron:

- bootstrap sobre **todos** los tokens de la población primaria, no sólo tokens
  autorizados;
- IC percentil exacto `[2.5,97.5]` y regla `equal_nan`/estado ante arrays vacíos;
- índices compartidos también para hard, proposer-only, head-to-head y promedio
  sham;
- alcance condicional en FIT, thresholds y cinco mappings observados; no incluye
  variación entre draws, refits ni recalibraciones;
- ausencia de corrección de multiplicidad y alcance descriptivo de los IC;
- definición token-wise exacta de `worst_regret` antes de promediar tokens;
- signos algebraicos y fronteras inclusivas/exclusivas de cada condición.

Por ejemplo, “reducción de regret de al menos `.005`” debe codificarse como
`mean(delta_regret) <= -0.005`; “límite superior” debe nombrar
`ci95_high(delta)`, y todas las diferencias deben fijar el orden
`policy-reference`. Para el head-to-head, `incompatibility-harm` ya está
declarado y debe mantenerse.

Los contrastes factoriales tampoco están enumerados. O se publican sólo las ocho
celdas y comparaciones simples congeladas, o se define una descomposición
factorial token-wise concreta; la palabra “atribución” no sustituye al
estimando.

### P2 — el contrato de artefactos/replay todavía es inventarial, no verificable

El plan enumera clases de artefactos (`plan:262-280`) pero no fija paths de
salida, lista cerrada de archivos científicos, modos de comparación ni
exclusiones. “Todos los artefactos científicos” no permite saber si receipts,
manifests, HGB nodes, labels autorizados o bundles deben ser byte-exactos,
array-exactos o equivalentes sólo por hash.

**Reparación.** Copiar la precisión del contrato Wave 57/58:

- output primario y replay canónicos y distintos;
- lista exacta de JSON/Markdown byte-exactos y NPZ array-exactos con
  dtype/shape/`equal_nan`;
- HGB nodes/estado de transporte hash-bound en `fit_freeze`, porque
  CALIBRATE-SCORES debe ejecutarlos en otro proceso antes de que existan scores
  validation;
- inventories completos, hashes pre/post y top-level de plan/auditoría/config;
- únicas exclusiones operativas enumeradas (`runtime`, manifest auto-referente,
  paths/timestamps/receipts que corresponda);
- replay desde vacío con el mismo triplete de claves y sin acceso del worker al
  output primario; comparación sólo por el coordinador después de producirlo;
- exactitud de preparación/visibles/logits/bundles además de la capa analítica.

Una metadata HGB puede ser “operativa” para portabilidad, pero durante la
frontera FIT→CALIBRATE es estado ejecutable y debe quedar autenticada antes de
producir scores. Los scores preservados pasan a ser la autoridad de reanálisis
una vez materializados; no reemplazan el binding del estado que los generó.

### P3 — la condición operativa de parada y la escalación son imprecisas

Wave 58 midió `53.6 s` y aproximadamente `1.02 GiB` RSS con tres HGB y una grilla
mucho mayor; trece fits HGB por corrida (tres reales más diez shams) y replay
siguen siendo plausiblemente CPU-minutes. El plan acierta al no reservar GPU.

“Muchas horas” y “materialmente más eficiente” (`plan:289-292`) no son límites
ejecutables. Fijar un presupuesto de wall time/RSS y un preflight pequeño evita
decisiones adaptativas de recursos. Además, la notificación directa por
Telegram contradice el protocolo local: `AGENTS.md:68` exige avisar antes del
uso GPU y `AGENTS.md:157` reserva `m2-alert` para intervención indispensable,
después de una nota inmutable en inbox. La reparación es detener antes de CUDA,
preservar estado, informar en el hilo; sólo si la intervención del usuario es
indispensable, publicar primero la nota con `request_id` y luego alertar.

### P3 — aclarar que las políticas principales están incluidas en las ocho celdas

Las dos políticas de `plan:114-120` son dos de las ocho combinaciones de
`plan:122-128`, no diez brazos adicionales. La config debe tener IDs únicos y un
inventario cerrado que evite materializarlas dos veces con nombres distintos.
Asimismo, `HARD-SET` y `hard_only` son una sola identidad de referencia; el
segundo no es una celda terminal de una grilla porque no hay búsqueda.

## Reparación mínima para reauditoría

El plan puede volver a auditoría cuando contenga, como mínimo:

1. frontera física `CALIBRATE-SCORES` truth-free y fase posterior `VALIDATE`;
2. semántica no contradictoria de soportes validation/monitor y patrones sólo
   monitor;
3. schema, paths, estados, roles y triplete de claves compatibles con un
   preparador Wave 59 explícito;
4. inferencia del factorial limitada a paquetes target+cuantil o bracket cruzado
   que identifique target;
5. naturaleza estadística del max-Hamming correctamente etiquetada, fórmulas de
   diagnósticos y seeds sin ambigüedad;
6. operación exacta de shards, estimandos, bootstrap y agregación ternaria;
7. matriz cerrada de artefactos y comparación replay;
8. presupuesto CPU y protocolo de escalación alineado con `AGENTS.md`.

Estas reparaciones no requieren cambiar la hipótesis ni abrir otra ola de
investigación. Requieren convertir una buena intuición experimental en un
contrato prospectivo observable y ejecutable. La decisión científica y todo
`GO/NO-GO` siguen reservados al usuario.
