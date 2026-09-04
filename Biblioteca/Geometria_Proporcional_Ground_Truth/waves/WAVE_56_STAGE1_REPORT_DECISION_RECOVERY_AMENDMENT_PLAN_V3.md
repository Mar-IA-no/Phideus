# Wave 56 Stage 1 — Plan v3 de autoridad documental del recovery

**Estado:** `FROZEN_IMPLEMENTATION_PLAN`
**Alcance:** cierre literal de R384–R386; protocolo científico sin cambios
**Decisión científica:** reservada a Mariano; no declara `GO/NO-GO`

## 1. Estado heredado

I3 (`7b37b5381b0c7540e86de2d53001903475d321ab`) contiene la corrección
válida del inventario visible real. R384 la confirmó, pero rechazó el parser que
aceptaba cabecera `PASS` con decisión `REVISE`. R385 exigió ligar la auditoría
del plan y cubrir lineage/fences. R386 confirmó esos cierres y dejó sólo dos
correcciones: eliminar el hash circular del informe final y registrar el commit
de la auditoría de plan. Este plan incorpora literalmente ambas.

Ningún artefacto `REVISE` se reutiliza como autorización.

## 2. Informes canónicos

R387, R388 y R389 deben tener cabecera canónica, UTF-8/LF, ningún comentario
HTML y ningún fence de backticks o tildes. Deben terminar exactamente con:

    ## Machine-verifiable decision

    **Final decision:** `PASS`

El resultado de cabecera es único y debe coincidir con la decisión terminal
única. Se rechazan ausencia, duplicación, `REVISE`, contenido posterior, fence,
separadores alternativos o campos ocultos. Esto cierra la contradicción
mecánica; lectura completa independiente y del coordinador sigue siendo
obligatoria para la semántica del cuerpo.

## 3. DAG sin ciclos

La cadena nueva es:

    P6 → R387(plan audit) → I4 → R388(implementation audit) → J6 → R389(final audit)

Cada flecha es parent directo. P6, R387, R388, J6 y R389 introducen exactamente
un path; I4 modifica exactamente preparador y test.

El amendment v6 registra:

- `plan = {commit, path, sha256}` para P6;
- `plan_audit = {commit, path, sha256}` para R387;
- `implementation.runner_commit = I3` y `implementation.commit = I4`;
- deltas/hashes de preparador y runner, y path/hash del test;
- path/hash de R388 como auditoría de implementación;
- solamente `final_audit_path` para R389.

No registra el hash ni el commit futuro de R389. Tras J6, el preflight deriva
ambos desde el único commit de introducción, exige diff exclusivo, parent
directo J6, `HEAD == R389`, worktree limpio y cabecera de R389 ligada al commit
J6 y al hash ya congelado del amendment. Así no existe dependencia circular.

Paths nuevos:

    Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_56_STAGE1_REPORT_DECISION_RECOVERY_AMENDMENT_PLAN_V3.md
    Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/387_wave56_stage1_report_decision_recovery_plan_final_reaudit.md
    Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/388_wave56_stage1_report_decision_recovery_implementation_audit.md
    experiments/geometria_proporcional/configs/wave56_stage1_report_decision_recovery_amendment_v6.json
    Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/389_wave56_stage1_report_decision_recovery_final_package_audit.md

## 4. Lineage y sources

I3 queda fijado por constante y amendment, es ancestro de P6, tiene parent único
y cambió exactamente preparador, runner y test. Su runner tiene SHA-256
`a9f2cd4e1826b9d1290d48faa0d5ead5cd48468488164462b1cce7c859ffde30`.

I4 es hijo directo del commit declarado/derivado de R387 y cambia sólo:

1. `experiments/geometria_proporcional/prepare_wave56_fresh.py`;
2. `tests/test_wave56_preoracle_recovery.py`.

El runner final en I4 debe ser byte-idéntico al de I3. El preparador/test final
se toman de I4. R388 liga ambos commits y los tres hashes. Frente al escrow de
origen, el contrato final difiere exactamente en preparador y runner; se
rechaza perder uno, agregar un tercero o modificar runner después de I3.

## 5. Verificación del primer borde

El preflight obtiene el commit de introducción único de P6 y exige igualdad con
`plan.commit`. Obtiene el de R387 y exige igualdad con `plan_audit.commit`.
Verifica blobs/hash, paths canónicos, diffs exclusivos, resultado/decisión
`PASS`, parent directo P6→R387 y parent directo R387→I4. La focal rechaza
omisión, mutación, commit declarado falso, `REVISE`, path no exclusivo e
interposición en cualquiera de esos bordes.

## 6. Matriz adversarial

El fixture Git modela por separado I3/P6/R387/I4/R388/J6/R389 y rechaza:

- runner_commit falso/no ancestro, blob divergente o runner cambiado tras I3;
- I4 con path adicional/faltante o parent distinto de R387;
- tercer source delta o ausencia de uno requerido;
- plan audit omitida, mutada, `REVISE`, no exclusiva o con commit/parent falso;
- cabecera `PASS` con decisión final `REVISE` para los tres informes;
- decisión ausente/duplicada/seguida por contenido y fences de ambos tipos.

Se conservan los 34 tests de calibración/inventario, no-redraw, origen físico,
manifest primary/replay y exactitud. R388 corre focal; R389 repite focal y suite
amplia Wave 49–56.

## 7. Ejecución posterior

Sólo con `HEAD == R389`, recovery vuelve a usar el intento fallido original y
el amendment v6. `--force` archiva el primary v2 `PREPARED`, que nunca es fuente
de claves. Luego se ejecutan primary `fit/select/adjudicate`, replay exacto y
sus fases. No se borra nada, se preservan estados crudos y todo permanece
CPU-only por contrato.
