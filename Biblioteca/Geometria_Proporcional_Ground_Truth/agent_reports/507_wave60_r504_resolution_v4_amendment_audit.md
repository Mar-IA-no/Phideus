# R507 — Auditoría del amendment suplementario de resolución R504 de Ola 60

## Dictamen: PASS

El amendment suplementario congelado en
`fd1270b084d4ebcbd6b4b675169665ce5f9c207b` cierra de forma íntegra la
resolución de autoridad requerida por R504. Conserva byte-exacta la autoridad
hard-set ya aceptada por R502/R503, liga sin reinterpretarlo el terminal R504
`REVISE` con `1/0/0`, y añade la secuencia R505/R506 que corrige la procedencia
de los sources invariantes. La transición acumulativa desde la config v3
modifica sólo preparer y test; los otros cinco sources no autorreferentes
permanecen iguales al baseline autenticado por R500.

No encontré findings altos, medios ni bajos. Esta auditoría no creó una nueva
config, no ejecutó el preflight final ni el experimento, no materializó
`attempt_v4`, no accedió a truth y no usó ni consultó GPU. La autorización de
config y ejecución continúa reservada a R508 y los eslabones posteriores del
plan.

## Identidad, canonicalidad y exclusividad

El target es exactamente el commit
`fd1270b084d4ebcbd6b4b675169665ce5f9c207b`, hijo directo de R506
`fb5dfa64ebe1f8a2b73b5123c3453695678c962e`. El commit añade un único path:

`Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_HARD_SET_AUTHORITY_R504_RESOLUTION_V4_AMENDMENT.json`

El archivo físico y el blob Git coinciden byte por byte. Su SHA-256 es
`51f3972a5f73178b1594376a92ac252d416736901870c44205a025ebac09d16b` y
la serialización coincide exactamente con
`json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"`.
El keyset superior contiene exactamente 25 campos: los 21 del amendment
hard-set original más `base_recovery_authority`,
`rejected_config_authority`, `correction_plan` y
`correction_plan_audit`. El schema
`wave60-hard-set-authority-r504-recovery-amendment-v1`, el estado `APPROVED`
y el tipo `HARD_SET_AUTHORITY` son los esperados por el parser vigente.

La comparación estructural con
`WAVE_60_HARD_SET_AUTHORITY_RECOVERY_V4_AMENDMENT.json` confirma que los 19
campos de autoridad base exigidos por el validador son idénticos. El suplemento
sólo cambia el schema, sustituye la implementación activa por la resolución
R506 y agrega las cuatro capas de procedencia de R504. No reescribe el
amendment inicial ni el informe R503.

## Cadena R502–R506 y resolución de R504

La secuencia relevante es lineal y cada transición tiene parent directo:

```text
amendment inicial      3a21c039118a4d9d95e84802d5b0e50b48be7cc9
R503 PASS              2b98cfa8af9d207547ab81bc9eafc0e0cc77b399
config v4 rechazada    201f9257c1e016b925719f6917dd1a9298ca493b
R504 REVISE            b70ee7e8f3d4f561a52017419aee0012914e2f54
plan correctivo        0e2768b665a5df329ca53de1a68ab6e14a86d70f
R505 PASS              e84bfe701bbc9cf0443c92f02372f74520a135ea
implementacion R506    464ceb59b8b634e34d625fe9458201075cdc8e3f
R506 PASS              fb5dfa64ebe1f8a2b73b5123c3453695678c962e
amendment suplementario fd1270b084d4ebcbd6b4b675169665ce5f9c207b
```

Los commits documentales son exclusivos de sus artefactos y la implementación
R506 modifica exactamente preparer y test. Los bindings físicos relevantes
coinciden con sus valores declarados:

| Autoridad | SHA-256 físico |
|---|---|
| amendment inicial | `1e4b097d7c9882e4037a608f65ac29ef10080addff8e4c24c2c922db2d84f4d8` |
| R503 | `b219d9d6273d378a0c1bd6ab5af0cf598b76fc1fad4425a2c7d2d9574a2d1f81` |
| config rechazada | `92bf02867579281f5df02b0cdd2c6980cbb8d1697f8df60a353457b86cea3960` |
| R504 | `3dafae137041ddf183c5bd1d64aa6e7e728c1a89fcf59c623103f799fa01c6f6` |
| plan de resolución | `49e6ba42b7f6802ead82b90f042a47022902dec8b1a9e53ae1c774b99adaef0d` |
| R505 | `efdb831cb4bd89fea25187c4635e1f9f53475b0d043b8fd0ad22d7649564a08f` |
| R506 | `4b402247bd37c87923a8c53ea23ee2b993c04979983f83958104d1ad599175ff` |

El parser normativo autentica R504 como `REVISE`, scope `CONFIG`, target exacto
y findings `high=1`, `medium=0`, `low=0`; no lo actualiza ni lo presenta como
PASS. R505 y R506 contienen un único bloque JSON normativo cada uno, ambos con
PASS y `0/0/0`. La validación recompone además la cadena R502 original,
incluidos su REVISE `0/1/0`, sus dos correcciones y el PASS final ligado por
R503. De este modo, la capa suplementaria preserva en lugar de sustituir la
historia de autoridad previa.

## Partición acumulativa de sources

`prior_source_sha256` coincide exactamente con los ocho sources de la config
v3 autenticada por R500. Tras retirar la config autorreferente, el universo de
siete sources queda cerrado, sin faltantes, extras ni intersección:

- cambiados, `2/7`: preparer y test;
- invariantes, `5/7`: runner, módulo científico, worker, R475 y R476;
- config self-bound, `1/8`, fuera de ambas particiones.

Los cambios acumulativos v3→R506 son:

- preparer: `85069cd9...4999` → `0dd0f338...92c8`;
- test: `ca344305...2bef` → `d8ca7d06...6960`.

Para ambos, `old_sha256` coincide con el mapa v3 y `new_sha256` coincide a la
vez con el blob del commit R506 y con el archivo físico. En los cinco
invariantes, el hash v3 coincide simultáneamente con el blob de R506 y el
filesystem. En particular, el runner conserva
`b35cd563f715bdff9b6e7489ac04712c728673563898d4a6aebf0144d4a50261`;
no revierte al blob pre-v3 que originó R504-01.

Esta composición implementa la regla del plan: una config previa tipada
gobierna todos los sources que la transición declara invariantes, mientras el
commit correctivo gobierna sólo la partición cambiada. Se mantienen las
guardas de keysets exactos, hashes físicos, blobs Git, parentage, exclusividad,
self-binding y roster cerrado.

## Ausencia de intento v4 y frontera de truth

El path
`data/geometria_proporcional/wave60_frozen_policy_transport_attempt_v4` no
existe ni como archivo, directorio o symlink. Entre R503 y el target sólo se
publicaron config/documentos y los dos sources de implementación R506; ningún
commit de la cadena creó outputs prospectivos. R504 registra que su preflight
falló antes del re-forward y de truth, y R506 registra una implementación y
tests CPU sin ejecución experimental.

R507 verificó esta frontera sin abrir el origen sellado ni ejecutar el
validador integral de materialización. Las comprobaciones leyeron JSON,
informes, blobs Git y los siete sources de implementación; no dereferenciaron
artefactos de truth. Esto es suficiente para el scope de amendment, pero no
anticipa la obligación de R508: congelar una config nueva, publicarla con su
auditoría exclusiva y obtener un preflight real contra ese HEAD antes de
cualquier ejecución.

## Comprobaciones CPU-only

Todas las órdenes usaron `CUDA_VISIBLE_DEVICES=''` y
`PYTHONDONTWRITEBYTECODE=1`; pytest deshabilitó plugins externos y cache. No se
usó ni consultó GPU.

| Check | Resultado | wall | max RSS | swaps |
|---|---:|---:|---:|---:|
| checker read-only de canonicalidad, Git, R502–R506, hashes y partición | PASS | 2.02 s | 815.416 KiB | 0 |
| `test_hard_set_r504_resolution_supplement_authenticates_separate_layers` | 1 PASS | 3.15 s | 841.356 KiB | 0 |
| `git diff --check` del target | PASS | — | — | — |

El test focal ejercitó la composición de las capas separadas y los rechazos de
deriva de autoridad base, findings R504 y binding de resolución. No se corrió
una suite que materializara árboles ni el experimento. La configuración pedida
al launcher fue Codex `gpt-5.6-sol`, effort `high`; esta instancia no expone
una interfaz independiente para verificar el identificador efectivo del
modelo/esfuerzo, por lo que no simulo esa confirmación.

## Riesgo residual

No queda un riesgo material dentro del scope R507. Permanece una frontera de
fase deliberada: el amendment no es todavía una config ejecutable y R507 no
puede demostrar el preflight de una config que aún no existe. La futura R508
debe validar esa composición completa en su propio HEAD; si falla, el plan
exige preservar el terminal documental sin crear `attempt_v4`.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R507",
  "scope": "HARD_SET_AUTHORITY_R504_RESOLUTION_AMENDMENT",
  "target": {
    "amendment_sha256": "51f3972a5f73178b1594376a92ac252d416736901870c44205a025ebac09d16b"
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
