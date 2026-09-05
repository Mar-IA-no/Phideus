# Ola 59 — cierre prospectivo del bracket HGB sobre draw fresco

> **Estado:** `COMPLETE / FRESH-DRAW / PROSPECTIVE-MONITOR / EXACT-REPLAY / AUDIT-PASS / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-05
> **Plan científico:** `WAVE_59_FRESH_HGB_GUARD_BRACKET_PLAN.md`
> **Protocolo sucesor:** `WAVE_59_REPLAY_NORMALIZATION_SUCCESSOR_PLAN.md`
> **Auditoría final:** `../agent_reports/454_wave59_successor_draw_final_audit.md`

## Qué pregunta respondió

La Ola 58 había nominado, sobre datos ya abiertos, dos políticas HGB/HGB: una
autorizaba por incompatibilidad posterior y priorizaba regret medio; la otra
autorizaba por harm y preservaba una señal de cola. La Ola 59 llevó ese bracket
a una realización nueva, con train, validation y monitor físicamente separados,
hard-set y proposer-only como referencias y cinco controles de desplazamiento
condicional máximo por target.

Los dos patrones quedaron definidos antes de abrir el monitor y se adjudicaron
por separado. El patrón de incompatibilidad exigía mejorar regret frente a hard
sin perder compatibilidad, sostener cola y superar sus controles matched. El de
harm exigía mejorar worst regret frente a hard, preservar accuracy y
compatibilidad y superar sus propios controles. Ninguno seleccionaba
automáticamente una arquitectura: la decisión científica permaneció reservada
al usuario.

## Antecedente operacional y sucesor válido

El primer intento de la ola produjo igualdad científica entre primaria y replay,
pero el comparador confundió un hash derivado del modo de ejecución con una
divergencia semántica. El defecto apareció después de abrir validation y
monitor. El protocolo vigente prohibía corregirlo y reaplicar el mismo draw, de
modo que ese intento quedó preservado sin adjudicación.

El sucesor corrigió antes del freeze la normalización tipada del receipt,
incorporó pruebas realistas `primary ↔ replay` y `recovery ↔ replay`, mantuvo el
contrato científico y generó claves, escrow y benchmark nuevos. Plan,
implementación y config recibieron auditorías independientes antes de ejecutar.
No hubo amendment, recovery ni redibujo durante la corrida válida.

## Ejecución e integridad

La implementación aceptada quedó ligada a
`40defeddb9f93bb6355cd9696549c3558057f0c3`; la config congelada, al commit
`c5e3e68295b41dda65235c65603305299833860d`; y la autoridad final de ejecución,
a `8edc23d1120a91981e01aeb8c385c344da42b3fb`.

Los artefactos canónicos son:

- primaria: `data/geometria_proporcional/wave59_fresh_hgb_guard_bracket_replay_normalized_v1/`;
- replay: `data/geometria_proporcional/wave59_fresh_hgb_guard_bracket_replay_normalized_v1_replay/`;
- config: SHA-256 `f6edfd2106fe87c8150562d096469e29b64a108a73de2dae0d371bd689a4a9b6`;
- `analysis.json`: SHA-256 `25a262c22bfe1d924578031d3d8a6faf7cc6e2632af4e8bc31e90f3ad8fca9b8`;
- `REPORT.md`: SHA-256 `0bc83f6cff43ee4881c0f4b59dd29346a8159ec22f3a4317a7de560b67fb1f6d`;
- comparación de replay: SHA-256 `4b0d2a420bf62966a451f8fd00ad6bc8e871c43e6c55cd836c73123242da156a`.

Cada split contiene `4.992` filas y `1.152` pair tokens: `768` elegibles,
`192` no canónicos y `384` fuera de catálogo. Ambas raíces recorrieron
`PREPARED → FIT_COMPLETE → CALIBRATION_FROZEN → VALIDATION_COMPLETE →
MONITOR_ACTIONS_FROZEN → COMPLETE`. El replay igualó `26/26` compromisos de
preparación, `21/21` artefactos científicos byte-exactos, `21/21` arrays,
`11/11` hashes opacos de secretos, `16/16` estados funcionales en cada raíz y
`11/11` comparaciones operacionales semánticas.

R454 recomputó `192/192` arrays analíticos, el bootstrap `5.000 × 303`,
`32/32` resúmenes y `36/36` contrastes factoriales. Los manifests cerraron
`82/82` archivos primarios y `84/84` de replay, sin faltantes, extras,
solapamientos ni artefactos sin clase. Primaria y replay consumieron en conjunto
`219,540 s`, muy por debajo del presupuesto de una hora; el RSS máximo de la
preparación fue `1.112.358.912 B` y el del análisis `751.722.496 B`. La
ejecución fue exclusivamente CPU y mantuvo CUDA invisible.

## Resultado prospectivo

Las dos políticas mejoraron resultados locales frente a hard, pero ninguna
satisfizo su patrón completo.

### Prioridad de regret medio: incompatibilidad

La política `P-HGB-HGB-INCOMPATIBILITY-Q90` actuó sobre `37` pair tokens. Frente
a hard, cambió accuracy `+0,006876`, compatibilidad `+0,006876`, regret
`-0,011941` y worst regret `-0,023102`. Los intervalos cumplieron siete de ocho
condiciones: el IC95 de regret frente a hard quedó completamente bajo cero y el
de compatibilidad completamente sobre cero.

Falló el contraste predeclarado contra el promedio de cinco controles de
desplazamiento condicional máximo. La diferencia de regret fue `-0,000786`, con
IC95 `[-0,002730,+0,000843]`; el extremo superior no quedó bajo cero. El patrón
cerró por eso `7/8`, agregado `false`.

### Prioridad de cola: harm

La política `P-HGB-HGB-HARM-Q70` actuó sobre `28` pair tokens. Frente a hard,
cambió accuracy `+0,009626`, compatibilidad `+0,003575`, regret `-0,010566` y
worst regret `-0,029978`. El intervalo de worst regret quedó completamente bajo
cero y la magnitud media superó el mínimo predeclarado.

Fallaron dos condiciones. El IC95 de compatibilidad frente a hard tuvo extremo
inferior `-0,001513`; y la diferencia de worst regret frente al promedio de los
cinco controles fue `-0,006766`, con IC95
`[-0,019087,+0,003465]`. El patrón cerró `6/8`, agregado `false`.

## Lectura y próximo discriminante

La observación no es que las políticas sean inertes: ambas mejoran varias
métricas frente a hard en este draw. La inferencia acotada es que esos cambios
no quedaron atribuidos a la ley HGB/guard bajo el control causal exigido. Una
política de desplazamiento matched alcanza una banda compatible con la mejora
principal; además, el brazo de harm no garantiza conservar compatibilidad.

Esto debilita la promoción del bracket actual, no refuta toda separación entre
proposer y guard ni establece un techo para la familia. El siguiente contraste
debe cambiar la pregunta, no repetir el mismo draw ni acumular más variantes del
mismo roster. Una alternativa recuperable es congelar la ley aprendida en una
realización y medir su transporte a otra sin recalibrar, junto con controles que
separen desplazamiento, target y capacidad. Otra es rediseñar la representación
del guard para que el estimando causal no quede agotado por una permutación de
igual magnitud. Ambas requieren diseño y auditoría antes de ejecución.

`scientific_decision` permanece `null`; `decision_authority` permanece `user`.
No hay arquitectura promovida, autoridad física ni decisión `GO/NO-GO`.
