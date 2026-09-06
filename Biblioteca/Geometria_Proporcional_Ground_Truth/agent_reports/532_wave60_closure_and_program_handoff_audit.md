# R532 — Auditoría independiente del cierre de Ola 60 y del relevo experimental

## Dictamen técnico: REVISE

El cierre de Ola 60 es fiel a R509, R530 y R531: reproduce correctamente el
inventario físico, la normalización condicional `36/36`, los soportes, las
métricas frente a `hard`, los dos contrastes contra controles matched y los
patrones terminales `false/false`. También conserva la observación histórica
`MISMATCH 35/36`, mantiene `scientific_decision=null`, no promueve arquitectura
y acota la inferencia al generador sintético y a la ley congelada de Ola 59.

La cartera de tres líneas conserva sus estatutos de forma razonable. El núcleo
relacional queda candidato y condicionado por executor; el posterior conjunto
queda recuperable sin rehabilitar el bracket HGB/HGB; y el router permanece
como integración condicionada, no como próximo modelo. La condición general de
terminar la campaña bibliográfica expansiva tampoco excede la evidencia.

El relevo experimental, sin embargo, aún no define un factorial causalmente
ejecutable. Reúne bajo un supuesto factor de «representación» tres objetos que
en la evidencia vigente no consumen el mismo contrato ni producen la misma
clase de salida, y su lifecycle de selección usa un rol no declarado. Encontré
**0 HIGH, 2 MEDIUM y 0 LOW**. Ambos findings afectan el diseño sucesor, no la
validez del resultado de Ola 60.

## Findings

### R532-01 — MEDIUM — La IR común está declarada, pero no existe todavía un mapeo tipado que vuelva comparables los tres brazos

La síntesis afirma que `EIV + conformal`, un encoder genérico y el núcleo
relacional tipado recibirán la misma observación y emitirán una misma IR de
compatibilidad e incertidumbre, tras lo cual se cruzarán con `hard` y una
política contextual (`PROGRAM_TERMINAL_ARCHITECTURE_SYNTHESIS_AND_HANDOFF.md:89-117`).
Esa igualdad no se deriva de los artefactos preservados:

- Ola 49 entrega a EIV mediciones `x,y`, covarianza y contratos de familia, y
  su output es un conjunto de familias compatibles o abstención
  (`WAVE_49_CLASSICAL_BENCHMARK_CLOSED.md:12-22`). El propio cierre lo define
  como **referencia externa**, no como tercer encoder comparable
  (`WAVE_49_CLASSICAL_BENCHMARK_CLOSED.md:102-115`). Además, `EIV + conformal`
  ya combina score, calibración, reader set-valued y abstención; no es sólo una
  representación aguas arriba de una decisión común.
- R342 opera sobre grafos de log-razones y el núcleo neuronal produce una
  relación corregida y una confiabilidad por arista para WLS/IRLS. R349 mostró
  que esa confiabilidad tiene semántica dependiente del executor y que el
  paquete no es solver-agnóstico. No produce actualmente compatibilidades entre
  familias ni un posterior sobre conjuntos.
- Ola 54 ajusta un posterior sobre los quince subconjuntos no vacíos a partir
  de cuatro logits de familia ya congelados
  (`WAVE_54_JOINT_SET_POSTERIOR_CLOSED.md:8-23`). Ese objeto no es la relación
  corregida por arista de R342/R349, aunque ambos puedan describirse de manera
  genérica como «incertidumbre».

Por eso «mismos bytes» y «IR expresiva» en el preflight
(`PROGRAM_TERMINAL_ARCHITECTURE_SYNTHESIS_AND_HANDOFF.md:143-148`) no bastan.
Unir todos los campos públicos permitiría que cada brazo use información
distinta; proyectarlos a un tensor común podría retirar precisamente la
orientación, gauge o composición que define al núcleo tipado; y tratar EIV
conformal como una celda podría atribuir a representación una operación de
reader/calibración que los brazos neuronales ejecutan en otra capa. En esas
condiciones, la interacción representación × política no localizaría el cuello
que el experimento dice estimar.

La resolución debe introducir antes del factorial un gate explícito de
`MAPPING-FEASIBILITY`. Debe elegir una única query y unidad científica; congelar
el schema público y el target lógico; especificar, por brazo, la transformación
autorizada desde observación a estado y desde estado a scores de compatibilidad;
separar score EIV, calibración conformal, reader y abstención; declarar qué
executor y checker son realmente comunes; y probar por mutación que ningún
adapter cambia autoridad, target o información. La utilidad contractual debe
quedar marcada como sintética y externa, sin tomar prestada la autoridad de
usuario que R374 declara ausente. Si el mapeo no conserva la operación nativa
del núcleo grafo, la comparación honesta son dos experimentos coordinados y no
un único factorial. EIV puede permanecer como referencia externa aun cuando no
sea una celda del efecto causal.

### R532-02 — MEDIUM — `validation`, `selection` y el freeze no tienen un orden compatible

El diseño declara cuatro roles físicos: `train`, `calibration`, `selection` y
`monitor` (`PROGRAM_TERMINAL_ARCHITECTURE_SYNTHESIS_AND_HANDOFF.md:119-125`).
Luego dice que márgenes y familia confirmatoria se derivarán de `validation`,
un rol que no aparece en esa partición (`ibid.:136-139`). Finalmente, el hito de
freeze exige congelar también los márgenes y resolver la auditoría **antes** de
abrir `selection` o `monitor` (`ibid.:149-151`). Si `validation` significa
`selection`, el margen no puede a la vez derivarse de ese split y estar
congelado antes de abrirlo; si son roles distintos, falta el quinto split y su
jurisdicción.

Esto deja indeterminado dónde se eligen modelo, thresholds, controles, familia
confirmatoria y márgenes, y por tanto impide auditar leakage o interpretar el
transporte prospectivo. La corrección debe fijar una única nomenclatura y un
orden de apertura. Una forma válida sería reservar `selection` para elegir
modelo/threshold/márgenes, auditar y congelar después su receipt, y abrir sólo
entonces `monitor`; otra sería derivar márgenes exclusivamente de
train/calibration y conservar `selection` como adjudicación intermedia sin
ajuste. En ambos casos debe declararse por campo qué rol puede escribirlo y qué
hash demuestra el freeze.

## Comprobaciones sin findings

- El commit target es hijo directo de
  `ee36ea58b9a71f3a67f9a014e370a54d9683a89b`, añade exactamente los dos paths
  declarados y sus SHA-256 físicos coinciden con el dispatch.
- El cierre de Ola 60 coincide con R509/R531 en `65 + 66 + 10 = 141` archivos,
  138 manifestados más tres self-manifests, 14 acciones, 56 arrays métricos,
  301 pair tokens, 5.000 bootstraps, 18 contrastes y 12 soportes.
- Los ocho intervalos frente a `hard`, los soportes `46/35` y los dos
  contrastes matched coinciden con la recomposición independiente. La lectura
  `false/false` no depende del falso negativo operacional de replay.
- Las tres líneas no fusionan autoridad física, checker ni decisión en el
  proposer. `BudgetPath` y R374 se presentan como prueba mecánica sintética con
  cero rutas históricas bajo utilidad real.
- El texto distingue observación, hipótesis e inferencia; no declara techo,
  `GO/NO-GO` ni promoción. El cierre del goal se condiciona todavía a auditoría
  y propagación documental.
- La frontera de recursos es adecuada: preflight CPU primero y detención antes
  de CUDA si la medición demuestra que GPU es materialmente más eficiente.
  Esta auditoría no usó ni consultó GPU.

El runtime no ofrece introspección independiente del identificador del modelo
ni del esfuerzo. `gpt-5.6-sol` con esfuerzo `high` constaba como requisito del
dispatch, pero no lo afirmo como verificado desde dentro.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R532",
  "scope": "WAVE60_CLOSURE_AND_PROGRAM_HANDOFF",
  "target": {
    "content_commit": "aca9a57ca4fd95d57d6258e83be6e580abb49815",
    "files": {
      "Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_FROZEN_POLICY_TRANSPORT_CLOSED.md": "426b9d4273829c8115b92681bcfce3c55955b043c117df20b4531953d37c4618",
      "Biblioteca/Geometria_Proporcional_Ground_Truth/PROGRAM_TERMINAL_ARCHITECTURE_SYNTHESIS_AND_HANDOFF.md": "04ddc735fd2d8977109d1582400795a40bbd3d0fcc588efc7ee74d25d386c020"
    }
  },
  "technical_verdict": "REVISE",
  "findings": {
    "high": 0,
    "medium": 2,
    "low": 0
  },
  "files_modified": false,
  "gpu_used_or_queried": false
}
```
