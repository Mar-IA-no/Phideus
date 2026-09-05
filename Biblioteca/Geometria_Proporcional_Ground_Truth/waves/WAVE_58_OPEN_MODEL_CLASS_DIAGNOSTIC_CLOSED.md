# Ola 58 — cierre del diagnóstico abierto de clase de modelo

> **Estado:** `COMPLETE / OPEN-DATA / ADAPTIVE / EXACT-REPLAY / AUDIT-PASS / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-04
> **Plan:** `WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_PLAN.md`
> **Auditoría de resultados:** `../agent_reports/414_wave58_open_model_class_results_audit.md`

## Qué pregunta respondió

La Ola 57 había mostrado que separar propuesta de valor y autorización de daño
era mejor que dejar actuar al proposer sin guard, pero no había aislado qué
forma debía tomar cada cabeza ni qué criterio de autorización merecía una nueva
realización prospectiva. La Ola 58 abrió deliberadamente train, validation y el
antiguo monitor de ese mismo draw para comparar, como diagnóstico de diseño,
proposers Ridge/HGB, guards Logistic/HGB, daño armónico o incompatibilidad
posterior y tres procedimientos de selección.

Ese acceso abierto cambia la autoridad del resultado. La ola puede localizar
familias de políticas y formular un contraste futuro; no puede validar una
arquitectura, estimar generalización entre realizaciones ni reclamar una mejora
prospectiva.

## Ejecución e integridad

La implementación auditada quedó congelada en `d558d98` y los nueve hashes de
fuente en `41a333c`. Primaria y replay corrieron con GPU invisibilizada, cuatro
hilos y workers UID/GID `65534`. Cada ejecución tomó `53.6 s`; el RSS máximo fue
`1,067,512 KiB` en primaria y `1,066,704 KiB` en replay.

Los artefactos canónicos son:

- primaria: `data/geometria_proporcional/wave58_open_model_class_diagnostic_v1/`;
- replay: `data/geometria_proporcional/wave58_open_model_class_diagnostic_v1_replay/`;
- `analysis.json` primario: SHA-256 `266049f57e28f98f29bcfdb36c6a99d7fcef6bbc735234ef6b5bdaa2e7d283d8`;
- `REPORT.md` primario: SHA-256 `6e8d829cd5e1bce6e83bd456cac2fc89406c4ace68be09f35d1d2a272aef65d9`;
- `scores_and_masks.npz` primario: SHA-256 `66c2976d5e07e88e6644268ac7f8ae829eae1a226ccd66696a7442dc4e6cb1de`.

El replay reprodujo exactamente los diez artefactos científicos. El brazo
`LEGACY-W57` reprodujo `34/34` estados, scores, thresholds, máscaras, acciones y
métricas. R414 recorrió además `120` combinaciones candidato×split y recomputó
sin divergencias acciones, soportes, medias, deltas e intervalos bootstrap.

## Resultado abierto

Los `36` IDs canónicos y `24` probes históricos quedaron presentes y
evaluables. El orden total congelado nominó
`C-HGB-HGB-INCOMPATIBILITY-JOINT`, con proposer HGB y guard HGB de
incompatibilidad posterior:

| Split | Soporte | Accuracy | Compatibilidad | Regret | Worst regret |
|---|---:|---:|---:|---:|---:|
| validation | 45 | 0.842163 | 0.945640 | 0.114583 | 0.375000 |
| monitor abierto | 55 | 0.847631 | 0.949483 | 0.108127 | 0.382898 |

Frente al hard-set, validation muestra compatibilidad `+0.004967`, con IC95
`[+0.000138,+0.010072]`, pero el intervalo de regret `-0.005427` todavía cruza
cero y worst regret cambia en dirección adversa `+0.008002`, también con
intervalo cruzando cero. En el monitor abierto, compatibilidad mejora
`+0.009123`, IC95 `[+0.000950,+0.016612]`, y regret baja `-0.014490`, IC95
`[-0.023818,-0.005106]`; accuracy y worst regret no se separan de cero.

Estos intervalos son `CONDITIONAL / ADAPTIVE / POST-SELECTION`: describen el
draw inspeccionado, no la variación entre draws.

## Lo que el roster no identifica

La nominación no equivale a una victoria limpia entre 36 arquitecturas. Los
`21` candidatos elegibles forman también el frente Pareto completo. Los 36 IDs
colapsan a sólo `19` firmas conductuales sobre ambos splits; nueve son la misma
política `HARD_ONLY`.

En particular, el nominado `JOINT` y su variante `SEQUENTIAL` eligieron los
mismos thresholds y produjeron exactamente las mismas propuestas,
autorizaciones, acciones, métricas e intervalos. `JOINT` quedó primero por el
desempate lexicográfico del ID. La evidencia identifica una política compartida
HGB/HGB con guard de incompatibilidad; no identifica el procedimiento de
selección.

La señal de incompatibilidad también es interaccional y estrecha: en ese
guard-set, las combinaciones que cambian sólo el proposer o sólo el guard
terminan en `HARD_ONLY`. Sólo HGB+HGB activa una política. Esto justifica probar
la combinación, no declarar que HGB sea globalmente superior.

## Alternativa de cola preservada

El orden total priorizaba regret medio y por eso nominó incompatibilidad. El
brazo `C-HGB-HGB-HARM-JOINT` conserva otro compromiso: regret de monitor
`0.109284`, apenas mayor que `0.108127`, pero worst regret mejora respecto del
hard en `-0.025327`, IC95 `[-0.049292,-0.002996]`; el intervalo de worst regret
del nominado cruza cero. El guard combinado harm+incompatibility queda como
tercera alternativa de compromiso, no como resultado promovido.

El siguiente experimento con poder diagnóstico no debe multiplicar IDs que
realizan la misma política. Debe congelar sobre un draw fresco, como mínimo, la
política HGB/HGB de incompatibilidad y una HGB/HGB orientada a harm/cola, junto
con hard, el legado Ridge+Logistic y controles shuffled válidos por
construcción. La prioridad entre regret medio y cola debe declararse antes de
abrir el monitor.

`scientific_decision` permanece `null`; `architecture_promoted` permanece
`false`. No hay autoridad física, validación prospectiva ni decisión
`GO/NO-GO`.
