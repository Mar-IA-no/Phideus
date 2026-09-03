# Ola 55 — plan prospectivo del puente conservador posterior-decisión

> **Estado:** `AUDITED-FROZEN / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-03
> **Antecedente:** `WAVE_54_JOINT_SET_POSTERIOR_CLOSED.md`

## Pregunta

La Ola 54 mostró una separación precisa: el posterior conjunto regularizado
representó mejor el conjunto compatible que las marginales independientes, pero
su decisión bayesiana pura perdió exactitud frente a la política de conjunto
duro. La Ola 55 pregunta si esa representación puede incorporarse de manera
conservadora: conservar la decisión dura como ancla y permitir que el posterior
la reemplace sólo cuando estima una ventaja decisional suficientemente grande.

El estimando es estrecho. Se mantienen congelados el generador, el encoder
`sigmoid_only` de la Ola 51, sus tres seeds, el normalizador, los parámetros del
posterior conjunto seleccionados en la Ola 54, las 24 políticas ordinales, la
penalización incompatible y la función de regret. Sólo cambia la interfaz
posterior-decisión. No se prueba una PPU completa, una utilidad natural ni una
geometría física.

## Evidencia fresca y cronología

La arquitectura fue concebida después de abrir el monitor histórico de la Ola
54. Por eso ese monitor no vuelve a adjudicarla. Antes de extraer claves se
versionan plan, config, código, grillas, conteos esperados, bindings upstream y
política de fallos. Recién desde ese `HEAD` limpio se genera una única corrida
primaria con tres claves criptográficas nuevas y distintas entre sí. El
compromiso SHA-256 de la clave de generación debe diferir del benchmark de Ola
50; además se comparan hashes de observaciones canónicas para demostrar que no
sólo cambió el HMAC de identidad. De ese benchmark:

- `train` funciona como `decision_select`: selecciona únicamente la compuerta;
- `val` permanece físicamente separado como `sealed_monitor` hasta escribir el
  freeze de selección;
- `lockbox` se genera por compatibilidad con el generador, pero no se materializa
  su oracle, no se usa y queda reservado.

La inferencia sobre los visibles `train` y `val` ocurre antes de materializar sus
oracles, en un worker CPU sin operaciones de fit. Después se congelan logits y
manifiestos; recién entonces se calculan los targets de `train` y `val`. Un
preparador sin métricas reduce las cuatro vistas canónicas de cada `pair_token`
promediando logits por seed y luego construye dos bundles físicamente separados.
Debe demostrar cero solapamiento con todos los tokens históricos de ajuste,
selección y monitor de la Ola 54.

El directorio primario debe no existir antes de extraer claves. Un fallo técnico
se registra y se cuarentena; nunca habilita borrar y redibujar un paquete. La
recuperación reutiliza las tres claves del intento fallido y acepta sólo deltas
de código auditados que no cambien arquitectura, grillas, estimandos ni
criterios. `--force` archiva, no elimina. Ningún inventario o resultado observado
puede disparar otra generación primaria.

La corrida primaria y un replay reutilizan las claves de generación preservadas
y deben coincidir por array y por hash analítico. El replay verifica cálculo y
trazabilidad; no vuelve independiente el aparato generador.

## Representación congelada

Para cada token, el ensemble primario es el promedio de logits crudos de los
checkpoints `seed{17,29,43}__sigmoid_only.pt`. Antes de tocar el benchmark nuevo,
un preflight carga los checkpoints originales, reproduce array-exact los tres
`raw_eval/frozen_set/*__val_monitor.npz` de la Ola 52 y recién entonces exporta
copias inference-only que contienen sólo `model_state`, `seed` y `output`. El
worker recibe esas copias, nunca optimizer ni historial.

El re-forward histórico usa exclusivamente el visible `val` de Wave 50, sus
labels autorizados, el protocolo del benchmark, el split manifest de Wave 51 y
el normalizador ya ligado. Debe reconstruir exactamente orden de tokens, targets
y `set_logits` de los NPZ Wave 52. Tanto referencias como insumos se validan
contra hashes literales antes de comparar.

El posterior `joint_full` usa sin refit el theta **primary** de
`selection_state.npz`, verificado contra `selection_freeze.json`; nunca usa la
selección global de sensibilidad. También se preservan los theta primary
`joint_unary_cardinality` y `joint_full_target_shuffled`, y el posterior
independiente Platt ligado por hash.

Bindings upstream literales:

| Artefacto | SHA-256 |
|---|---|
| checkpoint seed 17 | `3df5c2b1e8560d7d1e65c9033c7fd1cb75cc0993b125d2ec5872d8e1b7d20777` |
| checkpoint seed 29 | `37d7cba18174c8489fad0d73c118708b9ad8b65041c8cda81c8426dfc9f9a6db` |
| checkpoint seed 43 | `57a77da762f4080bcce16b1e8e21b75a7f66c5a1fedf8fef8718da38e035772a` |
| normalizador Wave 51 | `2e5fb858364a4efaf36b2aa9c3adf220cc5cea3cd5edca05e43cc8e6db36b6ce` |
| `selection_freeze.json` Wave 54 | `621c2c7883eb42f530f21eb6e96e4136d8a49d49b49c9f1ddb41de45ed955972` |
| `selection_state.npz` Wave 54 | `2f8aa9062669d6a9fd6e5fd29cf5b0756093ad627dbb22de609d2c0f0c1549f3` |
| policy manifest Wave 52 | `f8a608d396ad48ba3b0336df2dc1955940515be0f51fa08328cd5ddb1e9a21e1` |
| Platt Wave 53 | `23e2c255bd04ff59ba07390322ff516b94a976bfe345def7e5a5cdfeebbae875` |
| visible val Wave 50 | `60198a13054a95320bcce780d5fdb29970292950ed1dafe90ded768458ce4582` |
| labels val autorizados Wave 50 | `ea77e914812d4738b2904a7f7b28a71daa097aa230273a15b6f96844c8567b3d` |
| protocolo benchmark Wave 50 | `c45a7fb245950521ceac4c6de75b51e746152f506522c52697d05bdc30673468` |
| split manifest Wave 51 | `2bfcf151992b0c04cb9bf34d6d7e5fda50cd733bd3ac8c4fff74d6324873d80a` |
| raw eval Wave 52 seed 17 | `f32afb0d31e5072e4f8d33682c2e023ee304bb24620f93ba062d30136923dbf0` |
| raw eval Wave 52 seed 29 | `fc857480e7cd9d3924c639fc68ef321e555da4c879951b6583c05c0f79ce4f14` |
| raw eval Wave 52 seed 43 | `017e1e5e0dd75cf955bb91b087bffa6aa1f221492d43992062a726ae65f169d6` |

El conjunto duro `H(z)` se obtiene como en la Ola 54: `sigmoid(z) >= 0.5`, con
fallback a `argmax(z)` si queda vacío. Para cada política ordinal produce la
acción ancla `a_h`, máxima utilidad dentro de `H(z)`.

## Arquitectura: posterior-gated residual policy

Sea `R_p(a)` el regret esperado de la acción `a` bajo un posterior sobre los
quince conjuntos no vacíos. Sea `a_p = argmin_a R_p(a)` la acción bayesiana y

```text
advantage_p = R_p(a_h) - R_p(a_p) >= 0.
```

La política residual conserva `a_h` salvo que la ventaja estimada supere una
compuerta global:

```text
a_bridge = a_p       si advantage_p > gamma + 1e-12
           a_h       en otro caso.
```

El sentinel serializable `"hard_only"` reproduce exactamente el baseline duro.
La decisión bayesiana pura queda como brazo separado, no como un valor de la
grilla. Riesgos iguales se resuelven por el menor índice, como en el `argmin`
vigente, y cada implementación debe assertar identidad exacta entre
`"hard_only"` y el hard baseline. La arquitectura no mezcla etiquetas ni
consulta el resultado verdadero en inferencia: usa sólo logits, posterior,
utilidad y una compuerta congelada.

## Selección de la compuerta

La grilla cerrada es:

```text
gamma in {0.00, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, "hard_only"}
```

Cada brazo selecciona su propio `gamma` sólo en `decision_select`, dentro de la
población primaria `NEAR_RIVAL` con cardinalidad mayor o igual que dos. Son
elegibles las compuertas cuya accuracy no cae más de `0.01` frente al conjunto
duro y cuya tasa compatible no cae frente a él. Entre elegibles se minimiza
regret; empate: mayor `gamma`, para preferir menos overrides. `"hard_only"` garantiza
una opción factible idéntica al baseline. Se conserva como sensibilidad una
selección sobre todos los tokens in-catalog; si cambia el signo de un contraste
primario en monitor, el resultado se marca selector-sensitive.

La sensibilidad global vuelve a seleccionar el `gamma` de **cada** brazo sobre
todos los tokens in-catalog de `decision_select`. Con esos gammas alternativos se
recalculan en monitor siete deltas `bridge_joint_full - referencia`: regret,
accuracy y compatibilidad frente a hard; regret y accuracy frente a pure joint;
regret frente a independent Platt; y regret frente a target-shuffled. El signo
es `-1/0/+1` con tolerancia absoluta `1e-12`; cualquier cambio de estado entre
selección primaria y global marca `selector_sensitive=true`. El signo es
puramente algebraico; su dirección favorable se interpreta por métrica.

## Brazos y controles

1. `hard_set_policy`: ancla sin overrides.
2. `pure_joint_full`: decisión bayesiana pura de la Ola 54.
3. `bridge_joint_full`: candidata primaria.
4. `bridge_joint_unary_cardinality`: controla si las interacciones heterogéneas
   agregan valor decisional más allá de cardinalidad.
5. `bridge_independent_platt`: controla si cualquier incertidumbre marginal
   basta para obtener el efecto.
6. `bridge_joint_target_shuffled`: conserva arquitectura y prevalencias de la
   Ola 54, pero rompe allí la asociación logits-target.
7. `oracle_set_then_utility`: referencia privilegiada, nunca deployable ni techo
   universal.

Se reporta además una sensibilidad de `gamma` común: aplicar a todos los brazos
el `gamma` seleccionado por `bridge_joint_full`. La lectura primaria permite que
cada familia de posterior seleccione su propia compuerta, porque el rango de
ventajas depende de su calibración; la sensibilidad común separa ese efecto.

## Métricas y criterio diagnóstico

La unidad de análisis es `pair_token`; las 24 políticas son mediciones fijas. Se
reportan accuracy de acción, tasa compatible, regret medio, media entre tokens
del máximo regret sobre las 24 políticas, tasa de override y regret condicionado
a override. Cada override se clasifica por su regret observado frente al hard
con tolerancia `1e-12`: beneficioso, neutro o perjudicial; `override_precision`
es `beneficiosos / (beneficiosos + perjudiciales)`. Se reportan además las
fracciones beneficiosa, neutra y perjudicial sobre **todos** los pares
token×política, y los conteos `n_overrides` y `n_non_neutral_overrides`. Si el
denominador correspondiente es cero, `override_precision` y el regret
condicionado se serializan como JSON `null`, nunca `NaN`. Primero se promedian
las 24 políticas dentro de cada token y recién después se aplica bootstrap
pareado por `pair_token`, con índices comunes a todos los contrastes.

Los IC95 son percentiles `2.5/97.5` de `5.000` remuestras pareadas de tokens,
generadas una sola vez con `numpy.random.Generator(PCG64(5507))` y reutilizadas
para todos los brazos y métricas de la población primaria. No se remuestrean las
24 políticas dentro de un token.

Los cinco conjuntos ausentes del fit Wave 54 se reportan con `n_min=30`. Si el
monitor contiene menos de 30 tokens de ese soporte, no se calcula un intervalo ni
se interpreta desempeño fuera de soporte; el campo queda `NOT_EVALUABLE` y el
claim se restringe al soporte observado. Un benchmark futuro enriquecido para
esos conjuntos sería otra ley y otro experimento, nunca una extensión silenciosa
del primario.

`bridge_joint_full` queda marcado como candidato a réplica posterior, sin
promoción, sólo si en el monitor fresco primario:

1. reduce regret frente a `hard_set_policy` al menos `0.01`, con IC95 del delta
   debajo de cero;
2. el IC95 inferior de `accuracy_bridge - accuracy_hard` es al menos `-0.01` y
   el IC95 inferior de `compatible_bridge - compatible_hard` es al menos `0`;
3. frente a `pure_joint_full`, el IC95 inferior de la diferencia de accuracy es
   mayor que `0` y el IC95 superior del delta de regret es como máximo `0`;
4. reduce regret frente a `bridge_independent_platt` y
   `bridge_joint_target_shuffled` por al menos `0.01` en punto estimado y con
   IC95 superior debajo de cero en ambos contrastes;
5. no es selector-sensitive y el replay es exacto.

El contraste con `bridge_joint_unary_cardinality` es atributivo y se reporta con
IC95, pero no se fuerza como condición conjunta: la Ola 54 ya estableció valor
representacional de las interacciones; aquí se pregunta primero si existe una
interfaz conservadora utilizable.

## Artefactos

- benchmark fresco visible/sealed, claves y attestation;
- manifiesto de inferencia anterior al oracle y logits por seed para `train` y
  `val`;
- bundles físicamente separados `decision_select` y `sealed_monitor`;
- hashes de checkpoints, normalizador, parámetros posteriores, calibrador,
  políticas, plan, config y fuentes ejecutadas;
- freeze de selección antes de abrir monitor;
- acciones, riesgos, ventajas, overrides y métricas por token y token×política;
- índices de bootstrap, sensibilidad de selector y de compuerta común;
- receipts de acceso, runtime, paquetes, manifest de artefactos y replay exacto;
- JSON estricto, sin `NaN` ni infinito no estándar.

## Archivos previstos

- `src/geometria_proporcional/wave55_policy_bridge.py`
- `experiments/geometria_proporcional/_wave55_infer_worker.py`
- `experiments/geometria_proporcional/prepare_wave55_fresh.py`
- `experiments/geometria_proporcional/run_wave55_policy_bridge.py`
- `experiments/geometria_proporcional/configs/wave55_policy_bridge.json`
- `tests/test_wave55_policy_bridge.py`

## Límites operativos

CPU-only. La implementación comienza sólo después de una auditoría independiente
del plan y de resolver findings sustantivos. El benchmark fresco procede de la
misma ley sintética de las Olas 49–54: puede probar generalización a nuevas
realizaciones y corregir reutilización del monitor, pero no transferencia a otro
aparato. Si el puente falla, la evidencia favorece rediseñar el objetivo
decisional o el encoder; no autoriza escalar por reflejo. Cualquier experimento
GPU posterior se comunica al usuario antes de iniciarlo. Todo `GO/NO-GO` y toda
promoción arquitectónica permanecen en el usuario.

## Resolución de auditoría independiente

R307 emitió `REVISE` por cinco defectos materiales. Esta revisión: (1) congela
todo antes de extraer claves, exige compromiso de generación distinto y prohíbe
redraw tras fallos; (2) convierte el soporte ausente en diagnóstico con
`n_min=30` y no en criterio presumiblemente vacío; (3) fija hashes literales,
theta primary, preflight histórico array-exact y checkpoints inference-only;
(4) exige IC95 para todos los contrastes vinculantes; y (5) define tolerancia,
desempates, sentinel y métricas de override. Requiere reauditoría focal antes de
implementar.

R308 reauditoró esas resoluciones y conservó cuatro huecos ejecutables. La nueva
revisión liga por hash tanto las referencias Wave 52 como todos los insumos del
re-forward, define exactamente los siete contrastes de sensibilidad y su signo,
congela bootstrap PCG64 de 5.000 remuestras y especifica denominadores y `null`
para métricas de override vacías. R309 verificó las cuatro resoluciones, no abrió
findings nuevos y dio `PASS` para implementar.

La implementación se auditó después del primer hito funcional. R310 detectó dos
fallos altos de replay/no-redraw y tres huecos de evidencia; todos se corrigieron.
R311 mantuvo abierto que la unicidad dependía del padre arbitrario del output y
que recovery no ligaba el contrato congelado. La revisión final fija una única
raíz canónica, detecta allí todo intento con claves y exige igualdad integral de
config y plan antes de recuperar. R312 reauditoró esos cambios y dio `PASS` sin
nuevos findings materiales.
