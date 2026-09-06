# R499 — Auditoría independiente del amendment estático Wave 60 v3

**Dictamen técnico: `PASS` — 0 HIGH / 0 MEDIUM / 0 LOW.**

La amendment autentica de forma cerrada la recuperación del único falso
positivo observado en el guard de identidad del intento v2. El terminal físico
continúa siendo pre-truth y recuperable, el draw se preserva sin redraw, el
ledger firmado hereda exactamente 215,36700256168842 segundos y el cambio de
implementación queda limitado a runner, preparer y test. El módulo científico y
el worker permanecen byte-exactos. No encontré un camino por el cual la
excepción estática pueda relajar la identidad de otro archivo, un hardlink, la
igualdad primaria/replay, el manifest, la autoridad de sources o el presupuesto.

## Target, forma y autoridad documental

El target auditado es el commit
`17efd12c5a9247e1dc5c27b61315bc3b3bb3298c`, hijo directo de R498
`5839bcdb94ee9c401a5ceeb65700c3553d63f83c`. Su único path es:

```text
Biblioteca/Geometria_Proporcional_Ground_Truth/waves/
  WAVE_60_STATIC_PROTOCOL_IDENTITY_GUARD_RECOVERY_V3_AMENDMENT.json
```

El blob Git, el archivo físico y el binding esperado comparten SHA-256
`c56c4e22c93ec73ff9f5ce2ee31316769bf15e21d57d7047a9ab172fbfcbdee6`.
El archivo tiene 21.869 bytes, parsea como un único objeto, contiene exactamente
las 18 claves del schema
`wave60-static-protocol-identity-guard-recovery-amendment-v1` y sus bytes se
reproducen mediante JSON canónico con claves ordenadas, indentación de dos
espacios y newline final.

Leí completos la amendment, el validador vigente y las autoridades enlazadas.
La cadena real y exclusiva es:

```text
R495 config v2 PASS da9a9bf…
  -> plan inicial 4d29440…
  -> R496 REVISE eb674e0… (0/0/1)
  -> plan corregido bca3d2f…
  -> R497 PASS ff708b0… (0/0/0)
  -> implementación 24fbb4a…
  -> R498 PASS 5839bcd… (0/0/0)
  -> amendment 17efd12…
```

Cada commit documental o de auditoría tiene parent directo, path exclusivo y
blob SHA coincidente. R496 sólo objetó dos afirmaciones operacionales no
preservadas; el plan corregido sustituyó los ceros de swap por `no preservado`
y separó el transcript no archivado de la autoridad durable. R497 verifica esa
corrección sin cambiar el diseño. R498 autentica la implementación real.

## Origen v2, draw y población

`prior_attempt_container` resuelve canónicamente a
`data/geometria_proporcional/wave60_frozen_policy_transport_attempt_v2`, sin
symlink ni escape. `origin_inventory` tiene 53 entradas y coincide exactamente
con `physical_tree_inventory(attempt_v2/primary)`: dominio, tipos, tamaños,
hashes, modos y ownership. La comparación closed-world rechaza nodos extra,
symlinks y sustituciones físicas.

`preserved_draw_sha256` contiene exactamente 17 archivos: escrow, freeze,
manifest y los catorce miembros declarados por el manifest. Su dominio coincide
con ese conjunto cerrado y cada SHA coincide con el inventario físico. Escrow,
freeze y manifest reproducen además los bindings de `escrow_origin`. La config
snapshot física es la config v2 auditada por R495, con SHA-256
`b965ec2d414df0dc6ce3460dd3cb7cc741537971aac92000795dd3265a18f626`.

La población se recomputó desde los tres JSONL sellados. Train, val y lockbox
coinciden en 4.992 filas, 1.152 pair tokens únicos totales, 768 elegibles, 384
out-of-catalog, 192 no canónicos y 192 en la intersección elegible/no canónica.
El predicado conserva filtrado previo de filas, `is_out_of_catalog=false` y
población `canonical_preserving`.

## Terminal firmado y presupuesto durable

Los quince hashes de `prior_terminal` coinciden con los artefactos físicos de
pair, primary y replay. La validación Ed25519 completa reconstruye:

```text
pair terminal       PAIR_ABORTED_PRE_TRUTH
primary terminal    INVALID_NEW_DRAW_IDENTITY
replay terminal     PEER_ABORTED_PRE_TRUTH
any_truth_accessed  false
recovery_allowed    true
```

`recovery_pair_durable_elapsed()` vuelve a autenticar el paquete antes de sumar
el ledger. El resultado coincide exactamente con `prior_durable_budget`:

```text
preparation_seconds    215.36700256168842
source_binding_seconds   0.0
score_apply_seconds      0.0
phase_seconds            0.0
durable_seconds        215.36700256168842
```

El acumulado es menor que el presupuesto de 900 segundos y no incorpora un
segundo débito v1. No se abrió truth, no se ejecutó scoring y la amendment no
decide `GO/NO-GO` científico.

## Sources y frontera científica

`prior_source_sha256` es idéntico al mapa de ocho sources de la config v2, no al
mapa histórico embebido en el escrow v1. El commit de implementación cambia
exactamente tres paths y ningún cuarto:

```text
preparer  85069cd9e889e8dd5124ed0107c9e8fba69ab79a5edf52f3e25fee9e34ea4999
runner    b35cd563f715bdff9b6e7489ac04712c728673563898d4a6aebf0144d4a50261
test      ca344305754669d15a86456e148a22d45af8e9d0f79ada0e98fd7a37034e2bef
```

Para los tres, `old_sha256` coincide con la config v2 y `new_sha256` coincide
con el blob Git del commit `24fbb4a…` y el archivo físico. Los dos sources
declarados invariantes coinciden entre config v2, implementación y filesystem:

```text
módulo científico 46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65
worker             c6c5c832633d9ed8a99079dd4fc4f5677073f7e1883a204fa6bb41f6d646bac7
```

No cambian modelos, features, thresholds, policies, estimandos, bootstrap,
penalty, seeds ni acceso a truth.

## Excepción estática y resistencia adversarial

La allowlist contiene una sola entrada:
`benchmark/protocol_config.json`. Sólo omite la colisión de bytes de ese path
contra antecedentes. El archivo sigue sujeto al manifest, a igualdad byte-exacta
primaria/replay y a separación de `(device, inode)` respecto de replay y de cada
antecedente. Todos los commitments y los otros miembros siguen requiriendo
novedad.

El guard corregido aceptó read-only el par v2 real: 22 archivos compartidos por
bytes entre primaria/replay, todos físicamente separados, y cinco antecedentes
sin colisiones no autorizadas. Ocho tests focales pasaron en 20,16 s. Cubren E2E
v2→v3 con ledger continuo, config/source partition, provenance cerrada, hardlink
con replay o antecedente, drift de manifest/protocolo, desigualdad del par y
colisión byte-exacta de un archivo no estático. La corrida alcanzó 859.520 KiB
de RSS máximo y 0 swaps.

## Validación productiva integral

Para resolver únicamente la dependencia legítima de la amendment con R499,
construí mediante plumbing Git un commit object hijo directo de `17efd12…` que
añade sólo este informe. Un índice y gitdir efímeros bajo `/mnt/m2-1TB`
expusieron ese objeto como HEAD a los subprocess Git sin mover HEAD, main ni refs
del repositorio real.

Sobre esa vista construí en memoria una config v3 válida: namespace v3, doce
bindings de recovery hacia v2/amendment/R499, auditoría final R500, self-binding
y los tres hashes de sources autorizados. `execution_contract.git_commit` quedó
ligado al objeto R499. `validate_recovery_amendment()` ejecutó el validador
productivo completo sobre `attempt_v2/primary` en modo recovery y aceptó
lineage, auditorías, inventarios, firmas, terminal, ledger, source map, delta y
contrato científico. No se materializó v3.

Todas las validaciones usaron `CUDA_VISIBLE_DEVICES=''` y
`PYTHONDONTWRITEBYTECODE=1`; no se usó ni consultó GPU. Los basetemps, índice y
gitdir se inventariaron y eliminaron por sus paths exactos. No modifiqué código,
config ni datos; este informe es el único archivo creado.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R499",
  "scope": "STATIC_PROTOCOL_IDENTITY_GUARD_RECOVERY_AMENDMENT",
  "target": {
    "amendment_sha256": "c56c4e22c93ec73ff9f5ce2ee31316769bf15e21d57d7047a9ab172fbfcbdee6"
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
