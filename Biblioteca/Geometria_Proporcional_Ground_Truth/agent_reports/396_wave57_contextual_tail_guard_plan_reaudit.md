# R396 — Reauditoría focal independiente del plan revisado de Ola 57

## Dictamen

**PASS**

La versión de `WAVE_57_CONTEXTUAL_TAIL_GUARD_PLAN.md` contenida en el commit
`b271b8885cd5bcf08421fecf9e81a9df884bea2a` resuelve los siete findings de R395.
No quedan findings materiales abiertos en validez causal, estimando,
incertidumbre, control sham, mínimos, determinismo, estados terminales,
preservación o implementabilidad del plan. Tampoco se detectaron regresiones
nuevas que obliguen a revisar el diseño antes de implementarlo.

Este `PASS` habilita la implementación y su auditoría posterior; no habilita por
sí solo el draw fresco. El propio plan mantiene como condición previa que config,
código y tests estén versionados, el worktree esté limpio y el paquete
implementado reciba un `PASS` independiente sin findings materiales
(`WAVE_57_CONTEXTUAL_TAIL_GUARD_PLAN.md:342-351`). No se usó GPU ni se abrió
oracle nuevo.

## Material verificado

- HEAD exacto: `b271b8885cd5bcf08421fecf9e81a9df884bea2a`.
- Plan vigente completo: 374 líneas.
- R395 completo: 273 líneas.
- Artefactos analíticos ya abiertos de Ola 56 usados sólo para verificar los
  conteos históricos y Hamming declarados.
- Inference worker, materializador, preparador y coordinador de Ola 56 en las
  secciones necesarias para comprobar la estrategia de reutilización.

## Matriz de resolución F1–F7

| Finding R395 | Estado | Evidencia en el plan revisado | Verificación |
|---|---|---|---|
| F1 — contraste no matched contra proposer | **RESUELTO** | `:98-104`, `:145-159`, `:259-264` | `tau_mu` se selecciona una sola vez; proposer, main y shams comparten regla y propuestas. Cada guard selecciona sólo `tau_harm`. La condición 4 compara contra el proposer de máscara fija. |
| F2 — estimando y bootstrap ambiguos | **RESUELTO** | `:228-251` | Define las cuatro métricas token-wise, incluido `worst_regret_t=max_p`, el orden exacto del promedio sham y el carácter condicional del IC. Replay queda correctamente separado de réplica estadística. |
| F3 — mínimos por filas incompatibles con unidad token | **RESUELTO** | `:184-224` | Agrega mínimos de tokens con desacuerdo, clase, propuesta y autorización en FIT/SELECT/shards/monitor, además de filas. Decide prospectivamente que cero overrides transportados es evaluable sólo con soporte suficiente. |
| F4 — shuffle movible sin destrucción efectiva | **RESUELTO** | `:82-104`, `:201-214`, `:313-315` | Define el null condicional, preserva mapping y Hamming global/ponderado/por estrato, composición mixta/homogénea y hashes; fija `0.25` antes del draw futuro. |
| F5 — desempate incompleto | **RESUELTO** | `:167-182` | Establece orden de enumeración y claves lexicográficas totales para proposer, guard verdadero, shams y `hard_only`, con índice canónico terminal. |
| F6 — Logistic y fallos de ajuste indefinidos | **RESUELTO** | `:59-75`, `:293-303` | Congela semántica L2 compatible con scikit-learn 1.8.0, clase positiva, kwargs, pesos, convergencia, finitud y reconstrucción. Distingue fallo predeclarado del estimador de error de implementación y prohíbe retries adaptativos. |
| F7 — soporte unión en vez de por set | **RESUELTO** | `:220-224`, `:318-321` | Materializa `set_index`, exige evaluabilidad, summaries, contrastes e índices bootstrap por set y deja la unión sólo como diagnóstico no autorizante. |

## Verificación detallada

### F1 — atribución causal del guard

La reparación adopta la opción preferida de R395. El proposer se selecciona
primero y queda inmutable; si selecciona `hard_only` o carece de soporte, SELECT
termina sin abrir monitor (`WAVE_57_CONTEXTUAL_TAIL_GUARD_PLAN.md:145-151`). Sólo
después, main y shams recorren la misma grilla unidimensional de riesgo sobre la
misma máscara (`:153-159,178-182`). Así quedan separados:

- el contraste main–proposer, que identifica el efecto incremental de retirar
  propuestas mediante el guard;
- el contraste main–sham, que identifica la correspondencia de las labels de
  daño bajo el null condicional predeclarado.

Ya no existe la búsqueda 49-vs-7 ni un `tau_mu` distinto entre los brazos del
contraste causal principal. La condición 4 usa explícitamente el proposer de
máscara fija (`:259-262`). F1 está cerrado sin residuo.

### F2 — estimando y régimen de incertidumbre

Las fórmulas `accuracy_t`, `compatible_t`, `regret_t` y `worst_regret_t` son
inequívocas (`:228-240`) y coinciden con las primitivas heredadas. El plan aclara
que `worst_regret` es una media entre tokens de máximos dentro del catálogo fijo,
no una cola entre tokens ni CVaR (`:237-240`). También fija que cada sham se
evalúa antes de promediar sus métricas token-wise (`:242-245`).

El bootstrap remuestrea tokens del monitor y su alcance queda explícitamente
condicionado al FIT, SELECT, operating points y permutaciones observados
(`:247-251`). Por ello el resultado no podrá presentarse como incertidumbre de
un pipeline vuelto a entrenar sobre múltiples realizaciones. F2 está cerrado.

### F3 — mínimos y corrección reproducible `189→165`

El plan ahora alinea soporte y unidad inferencial: cuenta tokens con desacuerdo,
tokens portadores de cada clase, tokens propuestos y tokens autorizados, además
de filas (`:184-203`). Las condiciones aparecen tanto en la selección completa
como en shards y monitor. También decide antes del draw cómo interpretar cero
overrides transportados (`:212-218`).

La fe de erratas de R395 es correcta. Sobre
`data/geometria_proporcional/wave56_contextual_gate_fresh_v1/phases/fit.complete/analytics.complete/fit_arrays.npz`
(SHA-256 `46b3846c6cdd7e394c23a0b9dbd9e434001cdb4b3a0a509882c8bf81049070bf`):

```text
primary.sum()                                      = 299
(primary & disagreement.any(axis=1)).sum()         = 165
tokens con alguna label perjudicial                 = 119
tokens con alguna label no perjudicial              = 70
tokens que aportan ambas clases                     = 24
119 + 70 - 24                                      = 165
```

R395 sumó erróneamente `119+70` como si los conjuntos fueran disjuntos. La
revisión no reescribe el informe crudo y registra la corrección con la operación
exacta (`WAVE_57_CONTEXTUAL_TAIL_GUARD_PLAN.md:205-210`). La misma recomputación
da `165/164/165` tokens activos en FIT/SELECT/monitor y `1114/1051/1175` filas de
desacuerdo, coherente con el plan. F3 está cerrado.

### F4 — control sham

La revisión no sobreactúa la permutación como destrucción total: define un null
condicional dentro de `(policy_index, disagreement_count)` y reconoce que los
estratos homogéneos no cambian labels (`:84-96`). Los cinco shams mantienen
modelo, X, scaler, pesos, proposal mask, selector y presupuesto; sólo cambia el
target.

El umbral de Hamming `0.25` fue fijado usando evidencia ya abierta. La
recomputación con seeds `57031..57035` sobre FIT de Ola 56 devuelve:

```text
Hamming global:    0.327, 0.337, 0.338, 0.345, 0.366  (rango, ordenado)
Hamming ponderado: 0.318, 0.319, 0.322, 0.339, 0.359  (rango, ordenado)
```

Los intervalos declarados en `:94-96` son reproducibles. El umbral no garantiza
el resultado del draw futuro —ni debe hacerlo—, pero sí impide que un control
degenerado se adjudique como sham válido. Si una réplica falla, su contraste y
las condiciones dependientes quedan `NOT_EVALUABLE` (`:212-215`). F4 está
cerrado.

### F5 — determinismo de selección

Los cuantiles, su orden, el lugar terminal de `hard_only`, las cantidades usadas
para desempatar, los thresholds, `q` y el índice canónico completan un orden
total (`:167-182`). Esto cubre probabilidades repetidas, cuantiles con igual
threshold y máscaras idénticas. El mismo contrato rige para shams y shards. F5
está cerrado.

### F6 — modelo logístico y estados terminales

La especificación usa `l1_ratio=0.0`, que en scikit-learn 1.8.0 expresa L2 sin
depender del parámetro `penalty` deprecado (`:59-66`). También fija explícitamente
la orientación de `predict_proba`; no puede heredarse por accidente la
conversión a gain positivo del helper de Ola 56. Los checks de kwargs,
`sample_weight`, warnings, finitud y reconstrucción están prescritos
(`:68-75`).

Una clase, no convergencia o no finitud son fallos predeclarados del ajuste y
terminan FIT; un bug no puede camuflarse como `NOT_EVALUABLE` ni repararse sobre
el mismo draw (`:293-303`). Esta distinción cierra el riesgo adaptativo de R395.
F6 está cerrado.

### F7 — soporte individual

El plan exige explícitamente conteo y estado por `set_index`, bootstrap y
contrastes independientes, sin préstamo de soporte desde la unión
(`:220-224`). La preservación incluye la dimensión y sus estados (`:318-321`).
Esto reemplaza inequívocamente la conducta agregada del worker de Ola 56. F7
está cerrado.

## Búsqueda de regresiones nuevas

No se encontraron regresiones materiales. Quedan tres puntos de implementación
que deberán verificarse en la auditoría del paquete, pero el plan ya los delimita
y no requieren otra revisión científica:

1. **Componentes compartidos de Ola 56.** El inference worker actual espera un
   staged file llamado `wave56_config.json` y emite receipts con nombres Wave 56
   (`_wave56_infer_worker.py:36-57,105-197`); el materializador también conserva
   nombres de fase Wave 56 (`_wave56_oracle_materializer.py:18-32,115-132`). La
   implementación deberá demostrar mediante source bindings y tests si esos
   nombres se preservan como interfaz compartida versionada o se vuelven
   genéricos. No hay impedimento funcional: splits, UID/GID, visibles, seeds,
   checkpoints y autoridad son los mismos. Sí sería un finding de trazabilidad
   si el paquete final presentara un receipt Wave 56 como si fuera nativamente
   Wave 57 sin declarar la compatibilidad.
2. **Granularidad de `NOT_EVALUABLE`.** Tests deben probar que los mínimos de
   propuesta/autorización invalidan sólo la celda correspondiente, salvo los
   mínimos globales y el proposer `hard_only`, que sí terminan SELECT conforme a
   `:145-159,184-218`.
3. **Hooks sobre código cerrado de Ola 56.** El preparador y coordinador actuales
   contienen constantes y validaciones hard-coded a Wave 56. El plan autoriza
   hooks tipados y wrappers (`:342-349`); la auditoría de implementación deberá
   comprobar que no reducen las garantías del replay de Ola 56, que no eluden el
   source-at-HEAD y que seleccionan exclusivamente el worker analítico ligado a
   la config congelada.

Estos puntos son checks verificables de la implementación todavía inexistente,
no huecos del plan: éste exige nombres, source bindings, schemas, runtime closure
y tests explícitos antes de cualquier draw (`:342-351`).

## Cierre

El plan revisado puede pasar a implementación. El próximo gate legítimo es una
auditoría independiente de config, código y tests ya versionados; sólo su `PASS`
sin findings materiales puede habilitar la extracción de claves. Este dictamen
no promueve la arquitectura, no adjudica el patrón prospectivo y no declara
`GO/NO-GO`.
