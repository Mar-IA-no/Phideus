# Ola 59 — reauditoría independiente del plan prospectivo corregido

## Dictamen: REVISE

La revisión posterior a R416 resolvió la mayor parte de los problemas del
primer draft: separó físicamente la calibración de scores de la apertura de
truth validation, cruzó target con cuantil, reencuadró correctamente los
max-Hamming como stress-tests no exchangeable, definió shards, estimandos,
bootstrap, patrones, preparador, triplete de secretos, recursos y outputs.

El plan aún no debe congelarse ni crear el escrow. La reauditoría encontró
cuatro contradicciones materiales nuevas o residuales: las acciones monitor no
se congelan en un artefacto y subworker truth-free antes de abrir truth; el
soporte monitor se limita a `0..24` aunque el patrón exige al menos `25`; la
definición de población convertiría en `NOT_EVALUABLE` cualquier split que
contenga tokens no primarios; y la recuperación de `IMPLEMENTATION_ERROR` queda
autorizada sin acotarla a la cronología de acceso a outcomes. También persisten
dos problemas medios en los controles y el inventario replay.

No hay findings P0. Los P1 siguientes bloquean un `PASS`; todos admiten una
reparación localizada sin cambiar la pregunta científica.

## Corpus, hash y método

- Plan reaudidato:
  `Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_59_FRESH_HGB_GUARD_BRACKET_PLAN.md`.
- SHA-256 observado y esperado:
  `2cba5fb331fc6c1bff4a5cd9365c569a825f5809374e3360495c2b28751c4147`.
- R416: SHA-256
  `bf8c3f9e5cb74b9b4508794c9d83e4a65680e47b2b30dac129327ce22f047d5a`.
- `HEAD` al cierre de lectura: `8355a21caba67d048269be47b57e45d4ae1b1da7`.

Se leyó completo el plan vigente y se comprobó cada finding R416 contra las
configs, preparador, coordinadores y artefactos Wave 56–58. Los probes fueron
CPU-only con `CUDA_VISIBLE_DEVICES=''` y cuatro hilos máximos. No se consultó
web, GPU ni Mendieta.

## Matriz de resolución de R416

| Finding R416 | Estado en la revisión | Evidencia |
|---|---|---|
| Validation truth visible durante calibración | **RESUELTO** | `CALIBRATE-SCORES` recibe vista inference-safe sin target, gain, utilidad ni métricas; `VALIDATE` abre truth después del freeze (`plan:24-36,102-109,249-275`). |
| Soporte validation/monitor contradictorio | **PARCIAL** | Se separó el mínimo validation del soporte descriptivo monitor, pero apareció el rango imposible `0..24` frente a `>=25` (`plan:320-337,399-425`). |
| Preparador vigente no acepta Wave 59 | **PARCIAL** | Se declara schema, dispatch, roles, fases, paths y triplete (`plan:283-318`); la recuperación de errores aún es demasiado amplia. |
| Target confundido con cuantil | **RESUELTO** | Factorial completo `2×2×2×2`, 16 celdas y contrastes a cuantil común (`plan:130-163`). |
| Max-Hamming presentado como null | **MAYORMENTE RESUELTO** | Se etiqueta stress-test adversarial, se definen fórmulas y seeds (`plan:180-247`); falta exigir targets efectivos distintos, no sólo mappings distintos. |
| Shards sin operación | **RESUELTO** | Salt, universo, recalibración intra-shard, brazos, outputs y falta de autoridad definidos (`plan:377-390`). |
| Estimandos/bootstrap incompletos | **PARCIAL** | Pairing, percentiles, scope condicional y signos quedan exactos (`plan:339-375`), pero la pertenencia a la población primaria está formulada de modo terminalmente incorrecto. |
| Replay sólo inventarial | **PARCIAL** | Hay outputs y modos de comparación detallados (`plan:431-513`); falta cubrir todo el árbol real y congelar monitor antes de truth. |
| Recursos/escalación imprecisos | **RESUELTO** | Presupuesto `30 min/8 GiB` por corrida, preflight, CPU-only y secuencia inbox→alerta (`plan:515-532`). |
| Duplicación de políticas principales | **RESUELTO** | Se declara que son dos de las 16 celdas y no se duplican (`plan:132-143`). |

## Comprobaciones favorables adicionales

### El bracket ahora permite la comparación de target a cuantil común

Las 16 celdas corresponden exactamente a
`2 proposer × 2 guard × 2 target × 2 q`. Los conteos de contrastes también son
correctos: `8+8+8+8+4=36`. Se preserva el head-to-head de las políticas abiertas
como comparación de paquetes distintos, mientras los efectos de target se
calculan a `q` común. El texto limita adecuadamente “HGB” al contrato observado
y no vuelve a atribuir nada a `JOINT` frente a `SEQUENTIAL`.

### La calibración validation ya puede ser auditable por acceso

La división `gate_select_inference_bundle` / `gate_select_truth_bundle`, el
freeze intermedio y las allowlists positivas/negativas reparan el problema alto
de R416. La solución también reconoce que el preparador actual sólo acepta Wave
56/57 (`prepare_wave56_fresh.py:446-466`) y propone un dispatch Wave 59 tipado
sin cambiar los contratos históricos.

### Los cuantiles y magnitudes históricas permanecen correctos

El bracket conserva `q_p=.8`, `q_harm=.7` y `q_incompatibility=.9`, que son los
operating points Wave 58 citados por R416. El probe previo de FIT Wave 57 sigue
dando `978` entradas activas, `26` incompatibilidades, `11` pair tokens con
alguna positiva, prevalencia ponderada `0.023742` y máximo Hamming ponderado
`0.034239`. Los umbrales de evaluabilidad no fueron relajados post-hoc.

## Findings priorizados

### P1 — ADJUDICATE aún no hace observable que las acciones monitor preceden a truth

El plan afirma que ADJUDICATE “primero calcula scores y aplica thresholds, sin
truth; después materializa truth” (`plan:273-275`). Pero ambas operaciones
quedan dentro de la misma fase material, no se exige que el primer proceso
termine antes de crear el sandbox outcome-aware y no existe
`monitor_action_freeze.json` en la lista cerrada de artefactos
(`plan:444-474`). Así, la frase es una secuencia de código, no una frontera de
acceso verificable. Esto repite en monitor, en menor escala, el problema que la
revisión corrigió para validation.

Los thresholds globales ya están congelados, pero aún falta demostrar que las
scores, máscaras y acciones específicas del monitor fueron producidas sin
acceso a su truth. Una auditoría posterior podría recomputarlas, pero el plan
promete fronteras materiales y afirma que ADJUDICATE sólo lee outcomes después
de que sus acciones sean inmutables (`plan:66-69`). El contrato actual no hace
verdadera esa inmutabilidad antes de la apertura.

**Reparación obligatoria.** Mantener ADJUDICATE como fase lógica si se desea,
pero dividirla en dos subworkers físicos:

1. `MONITOR-APPLY`, UID/GID `65534`, recibe sólo
   `sealed_monitor_inference_bundle`, estados FIT y
   `calibration_freeze`; calcula scores, aplica thresholds y publica
   `monitor_scores.npz`, `monitor_policy_arrays.npz` y
   `monitor_action_freeze.json`; luego termina.
2. El coordinador valida y promueve atómicamente ese freeze. Sólo entonces crea
   un sandbox nuevo para el subworker outcome-aware, que recibe acciones
   read-only, truth monitor y utilidad, pero no estados ni una interfaz capaz de
   recalibrar.

El nuevo freeze debe hash-bind scores, propuestas, autorizaciones, acciones,
soportes, config y los tres freezes anteriores, y debe incorporarse a la lista
de replay byte-exacto. Tests negativos deben probar que el primer subworker no
ve truth y que el segundo no puede cambiar acciones.

### P1 — `0..24 overrides` contradice el soporte requerido y el tamaño real

El plan elimina correctamente un mínimo monitor que pudiera decidir la apertura,
pero luego dice que “`0..24` overrides son resultados evaluables”
(`plan:333-337`). Los patrones exigen
`authorized_pair_tokens >=25` (`plan:399-417`). Si `0..24` se refiere a pair
tokens autorizados, la condición positiva es imposible por definición. Si se
refiere a filas-policy, tampoco es un máximo: hay hasta 24 por token y muchos
tokens primarios.

El contraejemplo ya está en Wave 58. Aplicando los thresholds transportados al
monitor abierto, la política harm secuencial autorizó `208` filas en `48`
tokens y la de incompatibilidad `253` filas en `55` tokens. Ambos conteos de
filas exceden 24 y ambos soportes token-wise exceden 24.

**Reparación obligatoria.** Definir dos soportes sin un rango fijo falso:

```text
authorized_rows = count(primary[t] AND authorized[t,p])
authorized_pair_tokens = count_t(primary[t] AND any_p authorized[t,p])
```

Sus rangos son `0..24*N_primary` y `0..N_primary`, respectivamente. Cualquier
valor, incluido cero, permite computar la política; sólo la condición del
patrón `authorized_pair_tokens >=25` pasa o falla.

### P1 — la definición de universo primario volvería inevaluable todo split real

El plan dice que “un token sin filas primarias hace al split
`NOT_EVALUABLE`” (`plan:341-346`). Pero `primary` es una selección de población,
no una obligación para cada token físico. En el draw Wave 57 hay `768` tokens
elegibles por split y sólo `299/302/306` son primarios en train/validation/
monitor; existen respectivamente `469/466/462` tokens no primarios. La regla
literal haría `NOT_EVALUABLE` los tres splits por diseño.

La misma ambigüedad aparece en la tabla de mínimos, que habla de “tokens
totales” (`plan:324-331`). Wave 57 no contó los 768 tokens físicos: `_counts`
usa `int(primary.sum())` y restringe desacuerdos con `primary[:,None]`
(`_wave57_phase_worker.py:31-46`). Copiar los mínimos exactos exige conservar
esa semántica. Usar todos los tokens tornaría triviales los mínimos
`100/80/100` y rompería comparabilidad.

**Reparación obligatoria.** Definir primero

```text
T_primary = sorted({t : primary[t] is true})
```

y calcular métricas, mínimos y bootstrap sólo sobre `T_primary`, sin llamar
“eliminación silenciosa” a la exclusión predeclarada de la población
complementaria. Cada token de `T_primary` debe aportar exactamente las 24
políticas; una fila faltante, duplicada o fuera de schema es un error de
integridad/`NOT_EVALUABLE`. Reemplazar “tokens totales” por “pair tokens de la
población primaria”, también dentro de cada shard.

### P1 — la recuperación de IMPLEMENTATION_ERROR no está acotada por la apertura de outcomes

Los estados declaran que cualquier `IMPLEMENTATION_ERROR` conserva el escrow y
“admite recuperación pre-oráculo auditada con el mismo triplete”
(`plan:298-304`). Mantener las claves evita redraw, pero no evita adaptación de
código después de ver train, validation o monitor. FIT abre truth train;
VALIDATE abre truth validation; ADJUDICATE abre truth monitor. Un delta de
implementación posterior podría aprender de esos outcomes aun reutilizando
exactamente el mismo triplete.

El precedente Wave 57 fue mucho más estrecho: la recuperación corrigió un
conteo del preparador después de comprometer el benchmark pero antes de
inferencia y de acceso semántico, con amendment y auditoría de autoridad. No
autoriza una recuperación genérica después de outcomes.

**Reparación obligatoria.** Distinguir:

- reanudación con hashes idénticos tras crash operativo: permitida desde el
  journal y freezes ya publicados;
- delta de código/config antes de todo acceso semántico autorizado: sólo con
  amendment pre-oracle independiente que demuestre que el cambio es de frontera
  y reutiliza escrow;
- delta después de abrir train truth: no puede cambiar modelos, targets,
  cuantiles, métricas, mínimos, estimandos ni código científico para ese draw;
- delta después de abrir validation o monitor truth: el intento se preserva
  como inválido para adjudicación prospectiva y requiere un nuevo protocolo/draw,
  no se recodifica como `NOT_EVALUABLE`.

La máquina debe registrar qué truth fue materializada antes del error. “Terminal
por fase” también necesita nombres/transiciones exactos o un diagrama que haga
imposible saltar de error a una fase posterior.

### P2 — mappings distintos no garantizan cinco controles efectivos distintos

La revisión exige cinco hashes de mapping distintos (`plan:210-217`), pero dos
bijecciones pueden diferir sólo en permutaciones dentro de la misma clase y
producir el mismo array de target. En un estrato balanceado, el target
max-Hamming es necesariamente el complemento, aunque existan muchas
bijecciones. Repetir el mismo target cinco veces produce el mismo fit/control y
no constituye cinco réplicas efectivas.

**Reparación.** Exigir simultáneamente cinco `mapping_sha256` y cinco
`target_sha256` distintos por familia. Si la combinatoria del draw no permite
cinco targets max-Hamming diferentes, la familia es `NOT_EVALUABLE`, como ya
ocurre para otros mínimos. Congelar además el algoritmo exacto de emparejamiento
fuente→destino para que el mapping sea byte-reproducible, no sólo el subconjunto
receptor.

La naturaleza inferencial del control sí quedó corregida: el texto ahora lo
limita a stress-test adversarial condicionado en mappings y no lo llama null o
prueba causal (`plan:204-208,348-355`).

### P2 — la lista cerrada de artefactos no cubre el árbol que el propio replay compara

`scientific_artifacts` se presenta como lista exacta de paths
(`plan:440-475`), pero el comparador también debe verificar benchmark visible,
attestations, commitments y logits (`plan:494-501`). Ninguno de esos paths, ni
el escrow/generation receipt que PREPARE necesita, aparece en la lista. El
preparador real Wave 57 publica decenas de archivos bajo `benchmark/`,
`inference/logits/`, receipts, escrow y journals. A la vez, el plan dice que el
comparador rechazará todo path no clasificado (`plan:503-509`). Sin otra lista
cerrada, un output real conforme al preparador quedaría no clasificado.

Hay una segunda inconsistencia: los 16 joblib se describen como “dos
proposers, cuatro guards verdaderos y diez guards de control”, pero sus
filenames se derivan de los 16 IDs de **política** (`plan:440-442,480-483`). Las
cardinalidades coinciden por accidente; las entidades no. Un proposer se
comparte entre varias celdas y un ID de policy no identifica uno de los diez
guards de control.

**Reparación.** Definir una clasificación cerrada que cubra todo archivo
canónico:

- `scientific_exact`, `scientific_array_exact`, `functional_state`,
  `operational_semantic`, `secret_excluded_from_public_manifest` y
  `self_reference`, o categorías equivalentes;
- paths/patrones exactos para benchmark, attestations, commitments, visibles,
  logits, escrow, receipts, journals, freezes, joblib y
  `replay_comparison.json`;
- inventario real contra lista, sin faltantes ni extras, antes y después;
- IDs de modelo estables, por ejemplo proposers `ridge/hgb`, guards verdaderos
  `{logistic,hgb}×{harm,incompatibility}` y controles
  `{target}×{mapping_seed}`, mientras las 16 policies referencian esos IDs.

Agregar `monitor_action_freeze.json` y su comparación exacta a esta matriz.
Para joblib, “equivalencia funcional” debe fijar inputs, outputs, dtype y
tolerancia; los scores exactos en train/validation/monitor y el estado portable
ya ofrecen una autoridad más clara que un hash serializado potencialmente no
portable.

### P3 — dos precisiones menores antes de congelar

1. La secuencia final aún dice “creación única de clave” (`plan:539-543`),
   mientras el contrato correcto es un triplete creado una vez. Cambiarlo evita
   que un wrapper de replay interprete que debe crear otra clave.
2. “Los signos de los cinco contrastes primarios” por shard (`plan:386-387`) es
   ambiguo porque los ítems 1–5 contienen múltiples métricas y dos políticas.
   Enumerar los deltas/signos exactos que constituyen ese diagnóstico; ninguno
   debe actuar como gate.

## Condición de PASS en una nueva reauditoría

El plan quedaría listo para implementación si la próxima versión:

1. congela acciones monitor en un subworker inference-safe terminado y un
   `monitor_action_freeze.json` antes de abrir truth;
2. corrige los rangos/unidades de soporte y define `T_primary` como universo,
   no como requisito sobre los 768 tokens físicos;
3. restringe recovery según hashes y cronología de outcomes;
4. exige cinco targets de control efectivos distintos;
5. clasifica todo el árbol de artefactos/replay y separa IDs de modelo de IDs de
   policy;
6. resuelve las dos precisiones P3.

Las restantes decisiones del plan —pregunta, dos políticas principales,
bracket factorial, cuantiles, targets, métricas, stress-tests, shards,
bootstrap, patrones, recursos y frontera de claims— son coherentes con W57,
W58 y R414. Este `REVISE` no constituye una decisión científica ni un
`GO/NO-GO`; ambos siguen reservados al usuario.
