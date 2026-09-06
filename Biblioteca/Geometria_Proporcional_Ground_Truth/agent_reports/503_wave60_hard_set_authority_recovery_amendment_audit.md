# R503 — Auditoría del amendment de recuperación de autoridad `hard_set_tau` de Ola 60

**Dictamen técnico: `PASS` — 0 HIGH / 0 MEDIUM / 0 LOW.**

El amendment congelado en
`3a21c039118a4d9d95e84802d5b0e50b48be7cc9` autentica de manera cerrada el
único recovery autorizado: reutilizar el draw sellado del intento v3 e inyectar
`hard_set_tau=0.5` desde la autoridad física ya ligada por la source law v2. La
cadena no habilita scoring anticipado, redraw, otra realización, otro delta de
sources ni la materialización de config v4. Esta auditoría no ejecutó una
corrida prospectiva ni creó esa config.

## Target y forma canónica

El target es hijo directo de
`7182d97533c801cc718360fa2f444a140d0e06dc` y modifica un único path:
`Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_HARD_SET_AUTHORITY_RECOVERY_V4_AMENDMENT.json`.
El archivo físico y el blob Git tienen 28.811 bytes y SHA-256
`1e4b097d7c9882e4037a608f65ac29ef10080addff8e4c24c2c922db2d84f4d8`;
la serialización coincide con el hash del JSON canónico, el commit de
introducción es exactamente el target y `git diff --check` queda limpio.

El keyset superior contiene exactamente 21 campos: status/schema/tipo de
recovery; terminal, inventario y source baseline v3; terminal y ledger v2;
draw/escrow; contrato hard-set y población; plan/R501; implementación/R502; y
el débito único. Los validadores comparan también de forma exacta los keysets y
valores internos pertinentes. No hay un campo abierto, opcional o incongruente
que pueda autorizar una extensión lateral.

## Cadena R501 y resolución de R502

El plan físico tiene SHA-256 `6a4e95e4...5472c`, fue introducido de forma
exclusiva en `a9bb0cfd...` y R501 lo auditó en el hijo directo exclusivo
`2fe5f1f8...`; el informe físico R501 tiene SHA-256 `efa2f289...3134` y su
autoridad canónica es PASS, `0/0/0`, sobre
`HARD_SET_AUTHORITY_RECOVERY_PLAN`.

La cadena de implementación es lineal y físicamente verificable:
`2b1359bc...` (implementación inicial) → `00b433a6...` (R502 REVISE,
`0/1/0`) → `410a9189...` → `e29069f6...` → `7eba44b9...`
(implementación final) → `7182d975...` (R502 PASS, `0/0/0`). Cada commit de
implementación/corrección modifica sólo preparer y test; cada auditoría modifica
sólo su informe. Los hashes físicos de R502 REVISE y PASS son respectivamente
`3c7ed313...f706` y `c09195f6...f72f`, iguales a los bindings del amendment.

R502-01 queda efectivamente resuelto: ambas entradas programáticas reconstruyen
la autoridad canónica v4 antes de la primera mutación y rechazan contextos
fabricados. El amendment conserva `resolution_of` con el target inicial, el
R502 REVISE, el finding `0/1/0`, las dos correcciones y el target final; no
borra ni reescribe la revisión intermedia.

## Partición de sources

El baseline `prior_source_sha256` coincide clave por clave con los ocho sources
de la config v3 aceptada por R500, no con el mapa histórico v1 del escrow. El
delta v3→implementación final contiene exactamente:

- preparer: `85069cd9...9999` → `19942cb8...b539a`;
- test: `ca344305...2bef` → `d249aba7...f449f`.

Los cinco sources declarados invariantes coinciden simultáneamente en config
v3, blob de `7eba44b9...` y filesystem: runner `b35cd563...261`, módulo
científico `46e31fa1...c65`, worker `c6c5c832...ac7`, R475
`e5c49ca1...996` y R476 `497bb87f...a30`. El roster permanece en ocho; el
source de config se reserva para el futuro self-binding y auditoría R504. El
amendment no autoriza ningún tercer source modificado.

## Origen v3, draw y ausencia de truth científico

La revalidación física recompuso firmas, pair terminal e inventarios exactos:
pair `PAIR_ABORTED_PRE_TRUTH`, primary/replay `INVALID_PREPARATION`,
`any_truth_accessed=false`, `recovery_allowed=true`, con 44 records en primary
y 8 en replay. No existen `prepared`, labels durables, receipt ni attestation
de preparación. El error conserva el hash exacto de
`str(KeyError("hard_set_tau"))`.

El origen canónico es el único nested draw
`attempt_v3/primary/failed_preparation`. Su mapa preservado es closed-world,
coincide con escrow/freeze/manifest y los archivos declarados; no hay hardlinks,
aliases ni entradas extra. El access receipt acredita inferencia visible sin
fit ni labels, UID/GID `65534`, nueve logits esperados y probe de truth denegado
con `PermissionError`. Esto no niega el cómputo transitorio de labels de train
dentro del materializador antes del `KeyError`; sí confirma que no persistieron
y que no hubo acceso científico de scoring/evaluación ni acciones de lockbox.

## Contrato hard-set y presupuesto

La cadena física manifest → `source_law_request.json` → alias único
`wave59_config_snapshot.json` → snapshot de Ola 59 valida los hashes ligados y
resuelve un único valor finito, `hard_set_tau=0.5`. La config canónica v3 no
contiene esa clave; el seam real recibe sólo una copia efímera y el test focal
materializa los cinco bundles esperados sin alterar el origen.

El ledger firmado v2 se revalidó desde sus terminales y conserva
`215.36700256168842 s`. El amendment añade exactamente una vez el débito
conservador no firmado de `60.0 s`, ligado al fallo v3, para un inicio primary
v4 de `275.3670025616884 s`; replay deberá heredar el acumulado firmado de
primary. La mutación a doble débito es rechazada. Los tiempos transcript-only
`36.47 s` y `2.29 s` quedan como contexto del débito y no se presentan como
ledger firmado.

## Verificación CPU-only

La validación read-only directa recorrió JSON, Git, R501/R502, source partition,
autoridad hard-set, origen v3 y ledger v2: exit 0 en `2.42 s` wall, RSS máximo
`828.204 kB`, swaps `0`.

La suite focal ejecutó el seam real, la revalidación read-only del origen y
presupuesto, y cuatro ataques de deriva a pair failure, inventario, terminal v2
y doble débito: `6 passed in 67.67s`; `/usr/bin/time -v` registró `68.84 s`
wall, RSS máximo `864.268 kB`, swaps `0`, exit 0. Se usaron
`CUDA_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1` y plugins externos de
pytest deshabilitados. No se usó ni se consultó GPU.

La configuración pedida al launcher fue Codex `gpt-5.6-sol`, effort `high`.
Esta instancia no expone una vía interna para verificar de manera independiente
el identificador efectivo del modelo/esfuerzo, por lo que no simulo esa
confirmación.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R503",
  "scope": "HARD_SET_AUTHORITY_RECOVERY_AMENDMENT",
  "target": {
    "amendment_sha256": "1e4b097d7c9882e4037a608f65ac29ef10080addff8e4c24c2c922db2d84f4d8"
  },
  "technical_verdict": "PASS",
  "findings": {
    "high": 0,
    "medium": 0,
    "low": 0
  },
  "files_modified": false,
  "gpu_used_or_queried": false
}
```
