# Ola 59 — reauditoría final independiente del plan prospectivo

## Dictamen: REVISE

La versión posterior a R418 resuelve los cinco findings científicos y de
autoridad que quedaban abiertos: congela las acciones monitor en un subworker
truth-free terminado, corrige soportes y `T_primary`, acota recovery por
cronología de outcomes, exige mappings y targets de control distintos, y
separa IDs de modelo/policy con scorer portable autenticado.

Queda un único finding material de trazabilidad: la matriz cerrada de
`artifact_classes` no clasifica los artefactos que el propio camino de recovery
puede producir. En particular, el preparador que el plan ordena extender copia
`recovery_amendment.json` al output canónico recuperado, pero ese path no
pertenece a ninguna clase. El intento fallido preservado contiene además
`FAILURE.json`, también sin contrato de clase. Como el mismo plan exige rechazar
cualquier path no clasificado, una recuperación válida se autoinvalidaría en el
inventario.

No hay P0 ni otros P1 científicos. Corregir este único hueco y las dos
precisiones P3 indicadas abajo permitiría una reauditoría focal breve; no hace
falta modificar la hipótesis, los brazos ni las fases.

## Identidad y corpus

- Plan:
  `Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_59_FRESH_HGB_GUARD_BRACKET_PLAN.md`.
- SHA-256 observado y esperado:
  `caf5d3a37c0394075895e993b76e570959cf4fdc38f608475804344eeea54cff`.
- R418: SHA-256
  `50772e10c6bce64760c38d7e643d7f688c3763755659bfc65a3c4cbac849c4bf`.
- `HEAD` auditado: `8355a21caba67d048269be47b57e45d4ae1b1da7`.

Se leyó completo el plan vigente y se contrastó con R418, los contratos y
configs Wave 56–58, el preparador compartido, los coordinadores y los
artefactos Wave 57/58. Los probes fueron CPU-only con
`CUDA_VISIBLE_DEVICES=''` y cuatro hilos máximos. No se consultó web, GPU ni
Mendieta.

## Verificación de los seis requisitos de PASS de R418

### 1. MONITOR-APPLY, freeze y MONITOR-EVALUATE físicos — PASS

ADJUDICATE se divide ahora en dos subworkers materiales (`plan:280-288`):

- `MONITOR-APPLY` recibe sólo la vista inference-safe, estados y thresholds;
  publica scores, propuestas, autorizaciones, soportes, acciones y
  `monitor_action_freeze.json`, y termina;
- el coordinador valida y promueve ese freeze antes de crear otro sandbox;
- `MONITOR-EVALUATE` recibe truth, utilidad y acciones read-only, pero no
  estados ejecutables ni una interfaz de recalibración.

El freeze vincula config, bindings, freezes anteriores, estado portable,
scores y acciones (`plan:290-293`). Los tests negativos exigen que el primer
worker no pueda enumerar truth y que el segundo no pueda cambiar las acciones
(`plan:301-305`). El artefacto figura en `scientific_exact`
(`plan:507`) y en la comparación replay (`plan:547-559`). La ausencia de
post-selection monitor queda observable por acceso, no sólo declarada.

### 2. Soportes y `T_primary` — PASS

El plan define `T_primary` antes de mínimos y análisis, excluye explícitamente
la población complementaria y exige 24 policies íntegras por token primario
(`plan:365-369`). Los mínimos usan pair tokens de `T_primary`, no los 768
tokens físicos (`plan:371-380`).

Los soportes están tipados como filas y pair tokens con rangos correctos
`0..24*len(T_primary)` y `0..len(T_primary)` (`plan:382-394`). Cero overrides
permanece evaluable y sólo falla la condición descriptiva `>=25`. Esto repara la
contradicción que R418 contrastó con los `208/253` overrides por fila y
`48/55` por token de Wave 58.

### 3. Recovery gobernado por cronología — PASS

La máquina registra estados y máximo nivel de truth materializado
(`plan:323-328`). La matriz de recovery distingue:

- reanudación hash-idéntica tras crash;
- delta pre-oráculo auditado sólo antes de cualquier acceso semántico;
- prohibición de cambiar contrato científico después de train;
- invalidación prospectiva de cualquier delta después de validation o monitor
  (`plan:330-347`).

`NOT_EVALUABLE`, `INVALID-PROSPECTIVE-ATTEMPT` y `COMPLETE` son terminales
distintos y no hay salto desde error a fase posterior. Esta regla reproduce el
alcance estrecho de la recuperación Wave 57 sin convertirla en autorización
general.

### 4. Cinco targets max-Hamming efectivos — PASS

La evaluabilidad exige simultáneamente cinco `mapping_sha256` y cinco
`target_sha256` distintos (`plan:211-218`). El algoritmo fija orden
lexicográfico, uso de PCG64, elección de receptoras y emparejamiento
fuente→destino reproducible (`plan:238-242`). Si la combinatoria no produce
cinco mappings y cinco targets efectivos, la familia queda `NOT_EVALUABLE`
(`plan:244-253`).

El texto mantiene correctamente estos controles como stress-tests de
desplazamiento máximo, no permutaciones nulas exchangeable ni evidencia causal
(`plan:205-209`).

### 5. Clasificación exhaustiva y no solapada — REVISE

La tabla `artifact_classes` mejora sustancialmente R418. Define seis clases
mutuamente exclusivas, expansiones cartesianas cerradas, rechazo de faltantes,
extras y solapamientos, y modos de comparación adecuados
(`plan:501-515`). En el camino feliz cubre:

- benchmark, visibles, commitments, attestations y logits;
- bundles inference/truth, states, scores, métricas y freezes;
- receipts/journals/runtime;
- secretos, manifest auto-referente y comparación replay.

Sin embargo, la clasificación no cubre el camino de recovery que el plan
conserva. El preparador real declara
`RECOVERY_AMENDMENT_COPY_NAME = "recovery_amendment.json"` y, cuando existe
`recovery_context`, escribe esa copia en el output canónico y verifica su hash
(`prepare_wave56_fresh.py:2441-2453`). La comparación de preparación también
exige que el amendment aparezca en primary y replay y tenga el mismo digest
(`prepare_wave56_fresh.py:2348-2357`). El primario Wave 57 recuperado confirma
su presencia física en raíz.

`recovery_amendment.json` no aparece en ninguna fila de
`artifact_classes`. Por tanto, la regla “todo archivo pertenece exactamente a
una clase” y el rechazo de paths no clasificados (`plan:501-503,556-559`)
fallarían en una recuperación conforme al propio `plan:330-337`.

El runtime compartido también escribe `FAILURE.json` antes de archivar todo
intento fallido (`prepare_wave56_fresh.py:2660-2674`). El plan promete preservar
intentos inválidos y errores, pero no define si los árboles
`.failed_<timestamp>` tienen su propia matriz ni dónde clasificar
`FAILURE.json`. No es necesario comparar un failed attempt como si fuera el
primary final, pero sí darle un schema/inventario recuperable.

**Reparación obligatoria.** Añadir:

1. `recovery_amendment.json` como path condicional cerrado —preferentemente
   `scientific_exact` u `operational_semantic` con SHA exacto— presente en
   primary y replay sólo si el modo es recovery;
2. una clase/schema separado para archivos de intentos fallidos que incluya
   `FAILURE.json`, escrow/freeze parcial y el inventario disponible;
3. cardinalidades condicionales explícitas por modo `primary|replay|recovery|failed`,
   para que “ausente” y “extra” no dependan de heurísticas;
4. tests del camino recovery que ejecuten inventario primario/replay y comprueben
   que no queda ningún path sin clase.

### 6. IDs modelo/policy y scorer portable — PASS

Los IDs ya no dependen de la coincidencia cardinal con las 16 policies. El plan
enumera dos proposers, cuatro guards verdaderos y diez controles por
target+seed, mientras las policies sólo referencian esos IDs compartidos
(`plan:517-528`).

Los joblib quedan como copias funcionales, no inputs de workers posteriores.
`model_state_arrays.npz` contiene scaler, coeficientes y nodes HGB, está
hash-bound por FIT y es el estado ejecutable. El scorer portable se contrasta
contra el objeto ajustado antes de publicar y primary/replay preservan estados
y scores exactos (`plan:529-537`). La equivalencia joblib se define sobre las
matrices inference-safe de los tres splits con dtype `float64`, igual shape y
`rtol=0`, `atol=0`, `equal_nan=True`. Esto es más fuerte que depender del pickle
privado de sklearn y mantiene la autoridad de scores observada en Wave 58.

## Findings menores no bloqueantes por sí solos

### P3 — conteo nominal de freezes

La comparación replay promete todos los miembros de `scientific_exact`, por lo
que la cobertura operativa es inequívoca. No obstante, `plan:547-554` habla de
“cuatro freezes previos y `monitor_action_freeze`”, mientras la propia tabla
enumera cinco anteriores: `pre_generation`, `preparation`, `fit`, `calibration`
y `validation`. Corregir “cuatro” por “cinco” evita una aserción de conteo falsa.

### P3 — nombrar la tolerancia del chequeo portable-directo

La equivalencia entre joblib primario/replay sí tiene tolerancia exacta. Para el
chequeo previo a publicar FIT entre scorer portable y objeto sklearn
(`plan:529-535`) conviene fijar también la tolerancia. El antecedente Wave 58
usó `rtol=0`, `atol=2e-15` para absorber sólo redondeo de acumulación
(`wave58_open_diagnostic.py:376-378`). Heredar literalmente ese contrato o
declarar exactitud byte/float elimina libertad de implementación. Este punto
puede verificarse en auditoría de implementación y no altera por sí solo la
validez del diseño.

## Condición para PASS

Clasificar `recovery_amendment.json` y el árbol de fallos, fijar cardinalidades
por modo y corregir las dos precisiones P3. El resto del plan satisface los seis
frentes científicos/metodológicos exigidos por R418: no queda leakage
prospectivo observable, selector encubierto, confusión target×cuantil,
ambigüedad de soporte, null sham mal rotulado ni promoción automática.

Este dictamen es técnico. `scientific_decision` permanece `null`, no se promueve
arquitectura y todo `GO/NO-GO` sigue perteneciendo al usuario.
