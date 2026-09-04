# R395 — Auditoría independiente del plan de Ola 57

## Dictamen

**REVISE**

El plan propone un experimento prospectivo pertinente, acota correctamente el
claim a una realización fresca de la misma ley sintética y hereda una frontera
pre-oracle físicamente implementable. No está todavía listo para freeze ni para
extraer claves. Hay dos defectos altos: el contraste contra el proposer no
identifica el efecto incremental del guard, y el estimando no queda definido de
manera inequívoca para `worst_regret` ni para el promedio de shams. Un tercer
finding, medio-alto, alinea los mínimos con la unidad inferencial real. Los demás
hallazgos son reparaciones necesarias de determinismo, control nulo e
implementabilidad.

No se ejecutó GPU ni se inspeccionó ningún oracle nuevo. La comprobación
empírica auxiliar usa sólo el FIT ya abierto y cerrado de Ola 56.

## Alcance auditado

- Plan de Ola 57 completo.
- Cierre y plan de implementación prospectiva de Ola 56.
- Config congelada de Ola 56.
- Primitivas de `wave56_contextual_gate.py`.
- Runner retrospectivo, phase worker y coordinador prospectivo de Ola 56 en las
  secciones necesarias para métricas, selección, bootstrap, estados, replay y
  preservación.
- Artefacto FIT cerrado de Ola 56, únicamente para cuantificar la eficacia real
  de una permutación binaria con los estratos propuestos.

## Findings priorizados

### F1 — ALTO — El contraste principal no aísla el efecto incremental del guard

La hipótesis sostiene que la segunda cabeza filtra daños de las propuestas de la
primera (`WAVE_57_CONTEXTUAL_TAIL_GUARD_PLAN.md:31-34`). Sin embargo, el proposer
selecciona uno de 7 umbrales, mientras la interfaz conjunta selecciona de nuevo
`tau_mu` dentro de 49 pares `tau_mu × tau_harm`
(`WAVE_57_CONTEXTUAL_TAIL_GUARD_PLAN.md:126-155`). Por tanto, la diferencia
`mean_plus_harm_guard - mean_proposer_only` combina al menos tres intervenciones:

1. incorporación de la cabeza de daño;
2. cambio potencial del conjunto de propuestas por un `tau_mu` distinto;
3. búsqueda sobre 49 operating points y una restricción extra, frente a 7
   operating points y dos restricciones.

El control sham iguala la búsqueda de la interfaz conjunta y permite preguntar
si las labels verdaderas superan labels permutadas; no repara la atribución
incremental frente al proposer. En consecuencia, aun si la condición 4 pasa, no
se podría afirmar que el guard «retiró» daños del proposer seleccionado: la
mejora podría provenir de haber elegido otro proposer.

**Reparación obligatoria.** Elegir una de estas dos formulaciones antes del
freeze:

- Diseño identificable preferido: seleccionar una sola vez `tau_mu` para
  `mean_proposer_only`, congelar exactamente su máscara de propuestas y dejar
  que el guard verdadero y los cinco shams seleccionen sólo `tau_harm` sobre esa
  máscara.
- Si se conserva el selector bidimensional como política principal, agregar un
  comparador obligatorio `mean_proposer_at_joint_tau_mu`, que quite el guard
  manteniendo exactamente el `tau_mu` elegido por la interfaz. La condición que
  atribuye filtrado al guard debe comparar contra ese brazo matched. El proposer
  seleccionado independientemente puede permanecer como contraste de pipeline,
  pero no como identificador causal de la segunda cabeza.

El patrón y la narrativa deberán distinguir explícitamente «efecto de la
interfaz completa» de «efecto incremental del guard a propuestas fijas».

### F2 — ALTO — Falta una definición completa del estimando y del régimen de incertidumbre

El plan declara que la unidad inferencial es el pair token «después de promediar
las veinticuatro políticas» (`WAVE_57_CONTEXTUAL_TAIL_GUARD_PLAN.md:176-180`),
pero el código heredado no promedia las 24 políticas para `worst_regret`: calcula
el máximo por política dentro de cada token y recién después promedia tokens
(`src/geometria_proporcional/wave55_policy_bridge.py:74-95`). Esa cantidad es la
media poblacional del peor regret entre las 24 utilidades fijas por token; no es
el máximo global, un cuantil entre tokens ni CVaR. Las condiciones 3–5 dependen
de esa diferencia (`WAVE_57_CONTEXTUAL_TAIL_GUARD_PLAN.md:183-192`), de modo que
la ambigüedad cambia el resultado adjudicado.

También falta declarar que el bootstrap del runner heredado remuestrea sólo
tokens del monitor y calcula medias de deltas token-wise
(`run_wave56_retrospective.py:407-429`). Esos intervalos son condicionales a los
modelos, shuffles y operating points obtenidos en el FIT/SELECT realizado; no
incorporan variación por volver a generar y ajustar train/val. Con una sola
realización de fit/select, el patrón no estima la variación completa de todo el
pipeline.

Finalmente, «promedio de cinco shams» no fija el orden de las operaciones. Para
`worst_regret`, promediar acciones, regrets por política, máximos por token o
métricas finales produce cantidades distintas. Ola 56 promedió arrays de
métricas token-wise después de evaluar cada sham por separado
(`_wave56_phase_worker.py:1217-1226`), pero Ola 57 debe declararlo y no depender
de una herencia tácita.

**Reparación obligatoria.** Añadir fórmulas para cada token `t`:

- `accuracy_t = (1/24) Σ_p 1[action_tp = oracle_tp]`;
- `compatible_t = (1/24) Σ_p compatible_tp`;
- `regret_t = (1/24) Σ_p regret_tp`;
- `worst_regret_t = max_p regret_tp`;
- cada contraste poblacional es la media de los deltas token-wise sobre la
  población primaria del monitor;
- el comparador sham es, si ésa es la intención, la media por token de las cinco
  métricas ya calculadas para las cinco políticas sham seleccionadas por
  separado.

Debe decirse que los IC95 cuantifican incertidumbre de muestreo del monitor
condicional al FIT/SELECT y a las cinco permutaciones observadas. La inferencia
debe limitarse a ese estimando; replay comprueba determinismo, no agrega una
réplica estadística.

### F3 — MEDIO-ALTO — Los mínimos están expresados en filas aunque la unidad inferencial es el token

El plan exige 60 filas propuestas y 20 autorizadas por candidato
(`WAVE_57_CONTEXTUAL_TAIL_GUARD_PLAN.md:137-140`). Con 24 políticas por token,
esos mínimos admiten, en el extremo, propuestas concentradas en 3 tokens y
autorizaciones concentradas en 1. Del mismo modo, 150 labels perjudiciales y 100
no perjudiciales no aseguran diversidad de tokens portadores de cada clase
(`WAVE_57_CONTEXTUAL_TAIL_GUARD_PLAN.md:160-168`). La selección y el bootstrap,
sin embargo, tratan al token como cluster/unidad. El mínimo del monitor sólo
exige 100 tokens y no exige ninguna fila de desacuerdo, por lo que una
realización sin oportunidad real de override podría adjudicarse como resultado
negativo en vez de `NOT_EVALUABLE`.

Esto no es hipotético en cuanto a la diferencia de unidades: FIT de Ola 56 tuvo
299 tokens primarios pero sólo 189 tokens con algún desacuerdo; las 1.114 filas
de desacuerdo se repartieron de forma desigual. Esos conteos son derivables de
`gate_fit__primary`, `gate_fit__disagreement` y `gate_fit__gain` en
`data/geometria_proporcional/wave56_contextual_gate_fresh_v1/phases/fit.complete/analytics.complete/fit_arrays.npz`
(SHA-256 `46b3846c6cdd7e394c23a0b9dbd9e434001cdb4b3a0a509882c8bf81049070bf`).

**Reparación obligatoria.** Congelar antes de las claves, además de filas:

- tokens con al menos un desacuerdo en FIT, SELECT y monitor;
- tokens que aportan al menos una label perjudicial y tokens que aportan al
  menos una no perjudicial en FIT;
- tokens con al menos una propuesta y con al menos una autorización para cada
  candidato evaluable, tanto en SELECT completo como en cada shard;
- un mínimo de desacuerdos en monitor, separado del total de tokens primarios.

Los valores deben justificarse y fijarse antes del draw usando el diseño del
generador o evidencia histórica ya abierta; esta auditoría no inventa esos
umbrales. Si se decide que cero overrides en monitor es un resultado de
transporte y no falta de soporte, esa decisión y su estimando deben quedar
explícitos antes del draw.

### F4 — MEDIO — La fracción movible no demuestra que el target binario haya sido destruido

El plan controla sólo que cada shuffle tenga fracción movible de al menos 0,80
y promete registrar mapping y fracción efectivamente movida
(`WAVE_57_CONTEXTUAL_TAIL_GUARD_PLAN.md:72-85,160-173`). En una label binaria,
permutar índices puede mover observaciones sin cambiar labels; los estratos
homogéneos no pueden alterar el target en absoluto.

La comprobación sobre el FIT cerrado de Ola 56 lo cuantifica: con las cinco
seeds propuestas para Ola 57, la fracción de mappings no idénticos fue
`0.825–0.846`, pero la fracción Hamming real de labels modificadas fue sólo
`0.327–0.366`; 204 de 1.114 filas de desacuerdo pertenecían a estratos
homogéneos. Esto no invalida la permutación condicional —una permutación binaria
no necesita cambiar todas las labels—, pero sí demuestra que `movable_fraction`
no basta para certificar un sham no degenerado.

**Reparación obligatoria.** Definir por separado y preservar:

- fracción de índices permutables;
- fracción de mapping no identidad;
- Hamming real `mean(y_sham != y_true)` global, ponderado y por estrato;
- número/peso de estratos con ambas clases y de estratos homogéneos;
- hashes de mappings y targets.

La regla de `NOT_EVALUABLE` debe depender de una condición predeclarada sobre
destrucción efectiva o, alternativamente, el plan debe definir el control como
una permutación condicional exacta y limitar el claim a ese null, sin presentarlo
como destrucción completa de correspondencia. No debe elegirse el umbral después
de ver el draw fresco.

### F5 — MEDIO — El selector no tiene un desempate total

Después de minimizar regret, el plan desempata por menor autorización y mayor
umbral `mu_hat` (`WAVE_57_CONTEXTUAL_TAIL_GUARD_PLAN.md:142-155`). Quedan sin
orden total dos o más cuantiles de riesgo que materialicen el mismo threshold o
la misma máscara, situación plausible con probabilidades repetidas y regla
`method="higher"`. Depender del orden accidental de los loops puede ser
byte-reproducible en una implementación particular, pero no es una especificación
científica inequívoca.

**Reparación obligatoria.** Congelar una clave lexicográfica total para main y
shams, incluyendo como último término `q_harm` y, si fuera necesario, el índice
canónico de la celda. Fijar también el orden exacto de enumeración del producto
cartesiano y la representación de `hard_only`.

### F6 — MEDIO — El contrato no define fallos de convergencia ni puede reutilizar ciegamente la logística heredada

La cabeza solicitada fija `tol=1e-10`, `penalty=l2` y target de daño negativo
(`WAVE_57_CONTEXTUAL_TAIL_GUARD_PLAN.md:59-65`). El helper logístico disponible
en Ola 56 convierte su target con `y > 1e-12`, no con `g < -1e-12`, y construye
`LogisticRegression` sin pasar `tol`, `penalty` ni `fit_intercept`
(`run_wave56_retrospective.py:249-287`). Por lo tanto, reutilizar ese helper
violaría el plan aunque los defaults coincidan parcialmente. Además, el plan no
dice qué estado corresponde a `ConvergenceWarning`, coeficientes no finitos o
fallo numérico después de abrir FIT.

**Reparación obligatoria.** Implementar una primitiva Wave 57 específica y
testear la dirección de clase (`classes_ == [0,1]`, columna 1 = daño), todos los
kwargs congelados, `sample_weight`, convergencia, finitud y reconstrucción exacta
de probabilidades desde el estado preservado. Declarar antes del draw si un fallo
de ajuste produce `FIT_NOT_EVALUABLE` o un fallo terminal preservado; nunca
permitir reintento con `tol`, solver, `C` o `max_iter` distintos. Congelar y
registrar la versión de scikit-learn; el entorno observado durante esta auditoría
usa `1.8.0`, donde `penalty` aparece como parámetro deprecado, por lo que también
conviene testear que la invocación explícita no altera el contrato esperado.

### F7 — MEDIO — El soporte “por set” exige cambiar el worker heredado, no sólo copiarlo

Ola 57 exige 30 tokens por cada uno de cinco sets y reporte separado
(`WAVE_57_CONTEXTUAL_TAIL_GUARD_PLAN.md:160-174`). El worker de Ola 56 construye
una máscara unión para todos los sets ausentes y decide evaluabilidad con un
único conteo agregado (`_wave56_phase_worker.py:1301-1332`). Copiar ese armazón
sin reemplazar esa sección permitiría que el soporte de un set o la suma de
varios autorizara resultados para otros sin soporte.

**Reparación obligatoria.** Materializar una dimensión `set_index`; producir
conteos, estado, summaries, contrastes e índices bootstrap separados para cada
set. El agregado, si se conserva, debe ser diagnóstico adicional y nunca
sustituir el mínimo por set. Añadir un test donde la unión supere 30 pero cada
set individual no, que debe devolver cinco estados `NOT_EVALUABLE`.

## Comprobaciones favorables

1. **Leakage y autoridad pre-oracle.** Los roles físicos son disjuntos, no hay
   redraw y el monitor de Ola 56 no aporta filas ni thresholds al nuevo ajuste
   (`WAVE_57_CONTEXTUAL_TAIL_GUARD_PLAN.md:102-122`). El esquema
   `PREPARED → FIT → SELECT → ADJUDICATE`, la prohibición de deltas post-label y
   el worker sin privilegios están especificados (`:204-226`) y tienen soporte
   implementado en el coordinador de Ola 56 (`run_wave56_contextual_gate.py:324-445,1018-1182`).
2. **Features inference-safe.** Las 17 features heredadas se construyen a partir
   de logits, posterior, riesgos estimados, acciones y utilidades; truth sólo se
   usa para gain/labels y métricas (`wave56_contextual_gate.py:44-157`). No se
   detectó leakage directo de truth al diseño.
3. **Control de búsqueda contra sham.** Main y shams recorren la misma grilla y
   restricciones. Una vez resueltos F1 y F4, esto constituye un control válido
   de la correspondencia semántica de las labels bajo el null condicional
   elegido.
4. **Replay y preservación.** El inventario propuesto conserva estados crudos,
   scores, máscaras, grillas, métricas por política e índices bootstrap
   (`WAVE_57_CONTEXTUAL_TAIL_GUARD_PLAN.md:228-250`). Es suficiente para reanálisis
   sin re-forward ni retraining si se agregan los estados/diagnósticos precisados
   arriba. Replay exacto debe describirse sólo como verificación determinista.
5. **Costo e implementabilidad.** Ridge y seis logísticas sobre alrededor de un
   millar de filas de desacuerdo son CPU nativas. La reutilización del runner de
   Ola 56 es viable, siempre que Wave 57 tenga nombres, source bindings, schemas,
   estados y tests propios como exige el plan (`:252-279`). No hay razón técnica
   observable para usar GPU.

## Condición para una reauditoría PASS

Una revisión puede aspirar a `PASS` cuando:

1. exista un contraste a propuestas fijas o se limite la causalidad al pipeline
   completo;
2. estén formulados todos los estimandos y el alcance condicional del bootstrap;
3. los mínimos incluyan soporte por token acorde a la unidad inferencial;
4. el sham preserve diagnósticos de cambio real y un null inequívoco;
5. el selector tenga orden total;
6. convergencia y fallo terminal de Logistic estén congelados;
7. los support sets sean realmente evaluados por separado.

Hasta entonces no corresponde versionar una config como
`FROZEN_PROSPECTIVE_PROTOCOL_PRE_KEY_DRAW` ni extraer las tres claves. La
decisión científica `GO/NO-GO` sigue perteneciendo al usuario.
