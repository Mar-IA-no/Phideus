# R313 - auditoria independiente de resultados de la Ola 55

> **Fecha:** 2026-09-03  
> **Rol:** auditor independiente; Wave 55 tratada como trabajo ajeno  
> **Alcance:** plan congelado, implementacion, config, tests, benchmark fresco,
> bundles, corrida primaria y replay  
> **Veredicto:** `PASS`

## Veredicto ejecutivo

**PASS, sin findings materiales.** La implementacion ejecuta el estimando
congelado, la seleccion y los intervalos respetan la unidad `pair_token`, y la
cadena fresh/replay acredita no-redraw, separacion fisica y reproduccion exacta.
El resultado no satisface el patron diagnostico predeclarado (`4/9` condiciones
en el replay), sin que un error de signo, poblacion o bootstrap permita cambiar
esa adjudicacion.

La lectura provisional es sostenible con una precision importante: el
**umbral escalar unico seleccionado en la poblacion primaria** es
`"hard_only"`, por lo que `bridge_joint_full` no mejora al hard baseline sino
que lo reproduce exactamente. En cambio, la sensibilidad que selecciona sobre
todos los tokens in-catalog elige `gamma=0.2`; aplicada luego al monitor
primario, reduce regret pero tambien reduce accuracy. Por tanto, "no mejora"
debe significar **no obtiene la mejora conjunta exigida**, no "ninguna metrica
mejora bajo ningun selector". Esta salvedad no cambia el `PASS` ni el alcance
negativo de la interfaz escalar.

## Findings materiales

No se encontraron findings altos o medios. Tampoco aparecieron discrepancias
menores que requieran regenerar artefactos o revisar el veredicto.

## 1. Fidelidad al plan congelado

La corrida queda ligada al commit
`cbeabeba20a9e0b2b472672e86408e83b325eb08`. Los hashes actuales y los blobs de
ese `HEAD` coinciden con el freeze para plan
(`eaf178d9...e66c3d`), config (`61de5dc4...e4a0ad`), primitivas
(`5d0941ab...f951ad`), preparador (`5ea5d457...7f11b`) y worker
(`47c8eaa6...1e391`). El runner ejecutado pertenece al mismo `HEAD` y su hash es
`66ebdd31...caac`.

Los puntos ejecutables centrales coinciden con el contrato:

- la compuerta usa `advantage > gamma + 1e-12` y `hard_only` fuerza identidad
  exacta (`src/geometria_proporcional/wave55_policy_bridge.py:45-64`);
- la grilla es exactamente
  `[0, .01, .02, .05, .10, .20, .40, "hard_only"]`, con factibilidad por
  accuracy/compatibilidad, minimo regret y desempate hacia mayor gamma
  (`wave55_policy_bridge.py:151-179`; config, JSONPath `$.gamma_grid`);
- cada familia posterior selecciona su gamma en `decision_select`, tanto en la
  poblacion primaria como en la sensibilidad in-catalog
  (`run_wave55_policy_bridge.py:422-436`);
- el monitor no se carga hasta despues de escribir `selection_freeze.json`
  (`run_wave55_policy_bridge.py:437-453`);
- las 24 politicas se promedian dentro de token antes del bootstrap; el maximo
  regret tambien se calcula primero por token
  (`wave55_policy_bridge.py:74-95`);
- el bootstrap es pareado, usa una unica matriz `PCG64(5507)` de
  `5000 x 302`, y la copia preservada se reproduce array-exact;
- los cinco contrastes, las nueve condiciones diagnosticas, el soporte ausente
  y las sensibilidades de selector/gamma comun corresponden al plan
  (`run_wave55_policy_bridge.py:268-326,457-503`).

Los bundles contienen `768` tokens por split, tres logits por seed y cuatro
familias. La reduccion tambien es fiel: cada uno de los `768` tokens de `train`
y de `val` tiene exactamente cuatro vistas, primero promediadas dentro de seed;
`ensemble_logits` es array-exact al promedio de los tres seeds.

## 2. Freshness, no-redraw y provenance

La evidencia disponible sostiene la cronologia declarada:

- `pre_generation_freeze.json` registra `key_drawn=false`, el commit, hashes de
  fuentes, bindings upstream y el preflight historico antes del sorteo;
- el preflight re-forward reproduce array-exact `pair_token`, target,
  `set_logits` y `choice_logits` para seeds `17/29/43`, `384` tokens cada uno;
- las tres claves primarias son distintas entre si y sus archivos son
  byte-identicos en replay; el compromiso de generacion nuevo
  `de0321f1...d37fe` difiere del de Wave 50 `967ca4d8...4014`;
- las observaciones visibles nuevas tienen hashes
  `7582a27f...99eb6` (`train`) y `5bfa77df...a3345` (`val`), distintos del val
  historico `60198a13...4582`;
- hay `0` `pair_token` compartidos entre decision/monitor y `0` frente a los
  bundles fit/select/monitor de Wave 54;
- cada visible tiene `4992` fixtures; cada bundle elegible tiene `768` tokens y
  el primario contiene `300` tokens de seleccion y `302` de monitor;
- el worker ve solo visibles, protocolo, config, normalizador y checkpoints
  inference-only. Su inventario rechaza oracle, labels, sealed, lockbox,
  optimizer e history (`_wave55_infer_worker.py:31-50`), y el receipt declara
  `fit_operations=false`;
- se materializaron labels solo para `train` y `val`; no existe oracle de
  `lockbox` ni `authorized_labels/lockbox.jsonl`;
- la preparacion replay coincide en `12/12` checks: compromisos, hashes
  analiticos del benchmark, dos bundles por arrays/hash canonico y seis NPZ de
  logits;
- adjudicacion y replay comparten hashes exactos para `analysis_core.json`
  (`77c5e778...28cdb`), `selection_freeze.json`
  (`2aef410a...66ed7`), `result_arrays.npz`
  (`7a8d1c51...6698`) y `bootstrap_indices.npz`
  (`fea87e47...2361`);
- los manifests de resultados cubren `6/6` archivos primarios y `7/7` de
  replay, sin faltantes, entradas stale ni hash/tamano incorrectos. Todos los
  JSON parsean en modo estricto.

La proteccion no-redraw esta implementada sobre una raiz y nombres canonicos,
detecta intentos fallidos o superseded que ya contienen clave y obliga a
recovery con las mismas claves (`prepare_wave55_fresh.py:131-182`). Recovery
liga integralmente config y plan al freeze anterior
(`prepare_wave55_fresh.py:185-195`); `--force` archiva en vez de borrar
(`prepare_wave55_fresh.py:113-124`). No hay directorios failed o superseded
implicados por la corrida observada.

## 3. Seleccion y correccion estadistica

### Seleccion primaria

Sobre `decision_select` primario (`n=300`), los gammas seleccionados fueron:

| Brazo | Gamma |
|---|---:|
| `bridge_joint_full` | `hard_only` |
| `bridge_joint_unary_cardinality` | `hard_only` |
| `bridge_independent_platt` | `hard_only` |
| `bridge_joint_target_shuffled` | `0.2` |

La seleccion de `joint_full` es reproducible desde la grilla. `gamma=0.4` es
factible, pero su regret de seleccion es `0.120000`, frente a `0.119722` de
`hard_only`; por eso el minimo correcto es el sentinel. No hay un empate oculto
ni una inversion de signo.

### Monitor primario

En los `302` tokens `NEAR_RIVAL` con cardinalidad `>=2`, la candidata primaria
es exactamente igual al hard baseline en todas las acciones:

| Brazo | Accuracy | Compatible | Regret medio | Media del max regret por token |
|---|---:|---:|---:|---:|
| `hard_set_policy` | `0.831126` | `0.937362` | `0.129622` | `0.394316` |
| `bridge_joint_full` | `0.831126` | `0.937362` | `0.129622` | `0.394316` |
| `pure_joint_full` | `0.781457` | `0.974614` | `0.114008` | `0.426876` |

Por construccion, los tres deltas bridge-hard son `0` con IC95 `[0,0]`. Frente
a `pure_joint_full`, el bridge recupera accuracy (`+0.049669`, IC95
`[+0.030077,+0.070778]`) pero tiene mayor regret puntual (`+0.015614`, IC95
`[-0.000747,+0.032631]`), de modo que falla la no-inferioridad de regret. Frente
al bridge shuffled, su regret es peor: `+0.004289`, IC95
`[+0.000483,+0.008578]`. Los contrastes con bridge Platt y
unary+cardinality son exactamente cero porque ambos tambien seleccionaron
`hard_only`.

El replay resuelve las nueve condiciones: pasan identidad/no-inferioridad de
accuracy y compatibilidad frente a hard, superioridad de accuracy frente a pure
joint y replay exacto; fallan regret frente a hard, regret frente a pure joint,
regret frente a Platt, regret frente a shuffled y ausencia de sensibilidad de
selector. El agregado correcto es `4/9`, `all_satisfied=false` (replay report,
JSONPath `$.criteria`).

### Sensibilidad del selector

La seleccion sobre todos los `768` tokens in-catalog cambia
`bridge_joint_full` de `hard_only` a `gamma=0.2`. En el mismo monitor primario,
cambian de estado algebraico `5/7` contrastes: regret, accuracy y compatibilidad
frente a hard; regret frente a Platt; y regret frente a shuffled. Por ello
`selector_sensitive=true` esta correctamente adjudicado.

Como verificacion focal adicional, regenere los IC pareados de esa sensibilidad
con la misma matriz bootstrap preservada:

| Delta global-selector `joint_full - hard` | Estimacion | IC95 |
|---|---:|---:|
| Regret | `-0.010934` | `[-0.020925,-0.001644]` |
| Accuracy | `-0.011727` | `[-0.023593,-0.000276]` |
| Compatibilidad | `+0.018350` | `[+0.011589,+0.025938]` |

Este brazo alternativo hace `353/7248` overrides (`4.87%`). Solo
`134/353=37.96%` de los overrides no neutrales son beneficiosos por frecuencia,
aunque sus beneficios de magnitud bastan para reducir regret medio; la accuracy
cae de forma simultanea. Entre las 24 politicas hay heterogeneidad: ocho tienen
delta de regret negativo, diez positivo y seis no realizan overrides. Esto
explica por que un unico escalar puede producir un tradeoff agregado sin
constituir una mejora conservadora estable.

## 4. Adjudicacion de la lectura provisional

1. **"El puente escalar conservador global no mejora al hard baseline en la
   poblacion primaria": sostenido con precision.** El selector primario elige
   `hard_only`, por lo que la candidata reproduce el baseline. La sensibilidad
   global alternativa mejora regret, pero sacrifica accuracy mas alla de `-0.01`
   en punto e IC; no satisface el patron conjunto.
2. **"El resultado es selector-sensitive": sostenido.** Cambian `5/7` estados
   algebraicos predeclarados al pasar del selector primario al in-catalog.
3. **"El soporte de cinco conjuntos ausentes no es evaluable": sostenido.** Los
   indices `[0,4,8,10,12]` tienen `monitor_count=0`, por debajo de `n_min=30`;
   summaries e IC son `null` y el estado es `NOT_EVALUABLE`.
4. **"Esto rechaza esta interfaz escalar, no la utilidad del posterior ni toda
   politica residual": sostenido.** Wave 54 ya habia mostrado NLL y cardinalidad
   mejores y señal de interacciones. Wave 55 solo prueba una compuerta global
   escalar sobre el advantage del posterior congelado. No prueba compuertas
   condicionales, aprendizaje de utilidad, otro encoder ni transferencia fuera
   de la ley sintetica.

La evidencia permite rechazar la suficiencia de **esta interfaz y este selector
bajo este protocolo**. No autoriza declarar inutil el posterior, clausurar la
familia residual, extrapolar a utilidad natural ni declarar un techo.

## 5. Proximo experimento discriminante

El siguiente experimento mas informativo es una **compuerta residual condicional
de baja capacidad y con control de riesgo**, no otro posterior ni un encoder mas
grande. Debe mantener congelados generador, tres logits por seed, posterior
`joint_full`, utilidades, hard anchor y perdida; el unico objeto nuevo debe ser
la regla que predice si conviene reemplazar la accion hard.

La compuerta puede usar solamente variables disponibles al decidir: advantage
estimado, entropia/margen del posterior, cardinalidad hard, desacuerdo
hard-posterior, dispersion entre seeds y estructura de la politica/utilidad. Una
parametrizacion monotona o regresion regularizada sobre esos pocos features es
preferible a 24 umbrales libres. Debe ajustarse y seleccionarse con
cross-fitting agrupado por `pair_token`, conservando juntas las 24 politicas, y
conservar como brazos obligatorios hard, pure joint, el escalar Wave 55, la
compuerta condicional y el control target-shuffled.

Este contraste discrimina directamente dos explicaciones aun abiertas:

- si la señal del posterior permite identificar overrides valiosos pero el
  umbral unico mezcla regimenes/politicas heterogeneos, una compuerta condicional
  deberia mejorar el frente de regret-accuracy de forma estable;
- si tampoco una regla baja en capacidad supera al escalar en un monitor fresco,
  la evidencia apuntaria a que el advantage del posterior no ordena de manera
  decisionalmente util los errores del hard anchor bajo esta utilidad.

La adjudicacion debe ocurrir una sola vez en un nuevo monitor de la misma ley,
con estabilidad del selector evaluada entre shards independientes de
`decision_select`. Los cinco conjuntos ausentes requieren un **segundo brazo de
benchmark enriquecido**, explicitamente declarado como otra ley, con al menos
`30` tokens por conjunto; no deben mezclarse silenciosamente con la poblacion
primaria. Este diseño aumenta poder diagnostico sin confundir interfaz,
representacion y cobertura. No implica promocion ni decision GO/NO-GO.

## Verificaciones ejecutadas

- `PYTHONDONTWRITEBYTECODE=1 venv/bin/pytest -p no:cacheprovider tests/test_wave55_policy_bridge.py -q`: `14 passed`.
- Recalculo independiente de shapes, poblaciones, ensemble, acciones, metricas,
  bootstrap, selector alternativo e IC desde bundles/NPZ preservados.
- Comparacion array-exact entre bundles y resultados primario/replay.
- Verificacion integral de manifests de artefactos, hashes source-vs-HEAD,
  bindings, secretos reutilizados/distintos, solapamientos y JSON estricto.

No se modificaron codigo, config, tests, datos ni documentacion fuera de este
informe.
