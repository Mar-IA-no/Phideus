# Wave 56 Stage 1 — Plan v2 de cierre documental y lineage del recovery

**Estado:** `FROZEN_IMPLEMENTATION_PLAN`
**Alcance:** resolver R384 y los findings R385 sin cambiar el protocolo científico
**Decisión científica:** reservada a Mariano; no declara `GO/NO-GO`

## 1. Base preservada

El recovery v2 llegó a `PREPARED` y falló antes de abrir oracle porque el
runner omitía `calibration_null.jsonl` en su inventario esperado. I3
(`7b37b5381b0c7540e86de2d53001903475d321ab`) corrigió ese borde; R384
confirmó la corrección pero concluyó `REVISE` porque una cabecera `PASS` podía
contradecir una decisión terminal `REVISE`. R385 aprobó conceptualmente una
decisión terminal estructurada, pero rechazó el plan anterior porque su propia
auditoría no quedaba ligada a la DAG y faltaban negativos de lineage y fences.

Todos esos artefactos permanecen inmutables. Ningún `REVISE` autoriza recovery.

## 2. Gramática de informes de autoridad

Los nuevos informes de plan, implementación y paquete deben tener cabecera
canónica y terminar exactamente con estas tres líneas, más LF terminal:

```text
## Machine-verifiable decision

**Final decision:** `PASS`
```

El parser deriva el resultado de la línea única `**Result:** ...` de cabecera y
exige igualdad con `**Final decision:** ...`. Para reducir la superficie
Markdown, los informes que participan de esta DAG no pueden contener ningún
fence de backticks o tildes en parte alguna. También se rechazan sección
terminal ausente/duplicada, decisión duplicada, contenido posterior,
comentarios HTML y separadores no canónicos.

Este control no juzga la semántica del cuerpo. Cada informe se lee completo por
una instancia distinta y por el coordinador antes de incorporarlo.

## 3. DAG completa y verificable

Se crea un path nuevo para evitar ambigüedad con el plan rechazado:

```text
P5 → R386(plan audit) → I4 → R387(implementation audit) → J5 → R388(final audit)
```

Cada flecha es parent directo. P5, R386, R387, J5 y R388 introducen un único
path cada uno; I4 cambia exactamente preparador y test. El amendment registra:

- `plan`: commit, path y SHA-256 de P5;
- `plan_audit`: path y SHA-256 de R386;
- `implementation.runner_commit`: I3 exacto;
- `implementation.commit`: I4;
- blobs finales de preparador, runner y test;
- paths/hashes de R387 y R388.

El preflight obtiene los commits de introducción de P5/R386/R387/J5/R388,
verifica diffs exclusivos, parents directos, cabeceras y decisiones terminales,
exige `HEAD == R388` y worktree globalmente limpio.

Paths canónicos nuevos:

```text
Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_56_STAGE1_REPORT_DECISION_RECOVERY_AMENDMENT_PLAN_V2.md
Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/386_wave56_stage1_report_decision_recovery_plan_reaudit.md
Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/387_wave56_stage1_report_decision_recovery_implementation_audit.md
experiments/geometria_proporcional/configs/wave56_stage1_report_decision_recovery_amendment_v5.json
Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/388_wave56_stage1_report_decision_recovery_final_package_audit.md
```

## 4. Lineage I3/I4 y blobs

I3 queda fijado por constante y amendment. Debe ser ancestro de P5, tener un
solo parent y cambiar exactamente preparador, runner y test. El hash del runner
en I3 debe ser
`a9f2cd4e1826b9d1290d48faa0d5ead5cd48468488164462b1cce7c859ffde30`.

I4 es hijo directo del commit de R386 y cambia sólo:

1. `experiments/geometria_proporcional/prepare_wave56_fresh.py`;
2. `tests/test_wave56_preoracle_recovery.py`.

El runner en I4 debe ser byte-idéntico al de I3. Preparer y test se resuelven
desde I4. La cabecera de R387 liga ambos commits y los tres hashes finales.

Frente al contrato del escrow deben existir dos y sólo dos source deltas:
preparador y runner. Se rechaza perder uno, agregar un tercero, cambiar el
runner después de I3 o declarar hashes que no correspondan a los blobs.

## 5. Tests obligatorios

El fixture Git sintético debe modelar commits separados I3/P5/R386/I4/R387/J5/
R388. Debe rechazar al menos:

- plan audit omitida, mutada, `REVISE`, no exclusiva o con parent incorrecto;
- I4 no hijo directo de R386;
- runner_commit incorrecto/no ancestro, runner modificado tras I3 o hash falso;
- I4 con un path distinto de preparador+test;
- pérdida de preparador/runner o tercer source delta;
- cabecera `PASS` con decisión final `REVISE` para R386, R387 o R388;
- bloque terminal ausente, duplicado, seguido por contenido o dentro de fence
  de backticks/tildes.

Los positivos usan cabecera y decisión terminal concordantes. Se conservan los
34 tests de inventario visible, calibración, no-redraw, origen, replay y
manifest. R387 corre la focal; R388 repite focal y suite amplia Wave 49–56.

## 6. Amendment y recuperación

El schema/path v5 reemplaza sólo la autoridad futura; no reescribe los JSON
anteriores. Conserva inventario y hashes del intento fallido original,
population contract, no-redraw, revalidación antes/después de generación,
manifest primary/replay y exactitud.

Sólo con `HEAD == R388` se reejecuta recovery desde el intento fallido original.
`--force` archiva el primario v2 `PREPARED`; nunca se usa como fuente de claves.
Luego se ejecutan primary `fit/select/adjudicate`, replay exacto y sus tres
fases. Todo es CPU-only por contrato y se preservan estados crudos y receipts.

## 7. Cierre

La ejecución queda habilitada únicamente con `PASS` consistente de R386, R387
y R388, DAG completa, focal y suite amplia verdes, dos source deltas exactos,
`HEAD`/worktree válidos y ninguna apertura anticipada de oracle. La
interpretación científica posterior sigue separando observación, hipótesis e
inferencia y reserva `GO/NO-GO` al usuario.
