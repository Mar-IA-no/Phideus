# R496 — Auditoría independiente del plan de corrección del guard estático de Ola 60

**Dictamen técnico: `REVISE` — 0 HIGH / 0 MEDIUM / 1 LOW.**

El plan identifica correctamente un falso positivo estrictamente pre-truth y
propone la relajación mínima compatible con el contrato congelado: únicamente
la igualdad de bytes de `benchmark/protocol_config.json` deja de contar como
identidad con un antecedente. El archivo continúa dentro del manifest, ligado
al SHA upstream, igual por bytes entre primaria y replay y separado por inodo
de replay y de los cinco antecedentes. No se relaja ningún commitment, ningún
otro archivo ni ninguna regla de topología física.

El sellado del intento v2 antes de modificar el runner es además el único orden
correcto: conserva como autoridad el comportamiento que produjo el aborto y
evita que código posterior reinterprete retrospectivamente una root `PREPARED`.
Con el terminal físico pre-truth esperado, la vía v2→v3 es compatible con el
validador de recovery y con el ledger acumulado ya existente. La cadena
R496–R500 y los tres sources autorizados bastan para implementar, auditar,
autorizar y ejecutar esa recuperación sin redraw ni acceso semántico a truth.
El único finding no afecta ese diseño: dos observaciones operacionales del
preámbulo carecen de un artefacto durable que permita auditarlas.

## Finding

### LOW R496-01 — Dos observaciones operacionales no tienen fuente durable

El plan afirma `swap del proceso = 0` para primaria y replay (líneas 23–24) y
narra el primer comando ejecutado con la root agregada (líneas 39–44). Los
receipts firmados preservan duración, acumulado, RSS, presupuesto y
`cuda_visible_devices`, pero no poseen un campo de swap; tampoco existe un
journal o transcript archivado del comando que falló antes de cruzar la
transacción. Por tanto pude confirmar duraciones, RSS, ausencia de truth, draw
canónico y ausencia de débito adicional, pero no esos dos hechos históricos.

**Corrección mínima:** reemplazar los dos ceros de swap por `no preservado` (o
ligar una medición durable identificada) y calificar el comando errado como
`según transcript operacional no archivado`, sin usarlo como autoridad. No se
requiere cambiar la excepción, el orden de sellado, la superficie de código ni
la cadena R496–R500.

## Target y alcance

El target auditado es exactamente:

- plan commit:
  `4d29440b745407206053de3119344c6eeb05ffd3`;
- parent directo:
  `da9a9bf1f7c03ef2e062d874dd9f9f1a6f8d67bd`;
- path único:
  `Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_STATIC_PROTOCOL_IDENTITY_GUARD_RECOVERY_PLAN.md`;
- SHA-256 del blob Git y del archivo físico:
  `7c6d4472eea18662a677b887267134c0e9ab876145762275cf9cddfc7d590929`.

El commit es exclusivo y `git diff --check` no reporta defectos. La auditoría
leyó el plan vigente y el plan original completos, la config v2, los
validadores del runner y del preparador, el worker congelado y los tests
pertinentes. Los paquetes v2 fueron inspeccionados sólo por su superficie
física, hashes y metadatos públicos/firmados; no se interpretó contenido
sellado.

## Guardia de identidad: excepción cerrada

`opaque_draw_fingerprint()` incorpora cada miembro declarado por el manifest,
comprueba su SHA/tamaño y conserva device e inode
(`run_wave60_frozen_policy_transport.py:1479-1545`). Después,
`validate_new_draw_pair()` exige igualdad de commitments y bytes entre primaria
y replay, inodos distintos entre ambas, y novedad de commitments, bytes e
inodos frente a antecedentes (`run_wave60_frozen_policy_transport.py:1548-1615`).
Así se localiza el exceso: la comparación de bytes de todos los paths no
distingue entre material generativo y protocolo estático.

La allowlist unitaria de las líneas 65–83 del plan es suficiente porque:

1. la excepción se aplica sólo a `bytes:benchmark/protocol_config.json` en la
   comparación con antecedentes;
2. `validate_prepared_root()` mantiene presencia closed-world, consistencia
   SHA/tamaño con el manifest y la attestation firmada para ese path
   (`run_wave60_frozen_policy_transport.py:1217-1369`);
3. las comparaciones primaria/replay ocurren antes del bucle de antecedentes y
   por tanto no quedan afectadas por la excepción;
4. el check de `(st_dev, st_ino)` es una rama independiente del check de bytes,
   de modo que tampoco debe quedar allowlisted;
5. commitments completos, escrow, semantic root, bundles y todos los demás
   miembros del manifest continúan bajo la regla de novedad.

Las pruebas obligatorias del plan cubren el caso positivo y los ataques
relevantes: coincidencia de cualquier path no estático, hardlinks con replay o
antecedente, manifest divergente y par primaria/replay no exacto. También
corrigen el hueco del fixture actual: `_fake_draw()` sólo declara
`visible/train.jsonl`, por lo que el test vigente de identidad nunca ejercita
`protocol_config.json` (`tests/test_wave60_frozen_policy_transport.py:6778-6846`).

## Estado físico de attempt_v2

La reproducción read-only dio los siguientes resultados:

| comprobación | primaria | replay |
|---|---:|---:|
| `validate_prepared_root()` | PASS | PASS |
| entradas del fingerprint físico | 22 | 22 |
| commitment groups | 5 | 5 |
| `truth_accessed` firmado | `false` | `false` |
| `fit_operations` firmado | `false` | `false` |
| duración propia de preparación | 77,44117898866534 s | 77,92582357302308 s |
| tiempo previo | 60,0 s | 137,44117898866534 s |
| acumulado durable | 137,44117898866534 s | 215,36700256168842 s |
| RSS máximo | 951.365.632 B | 956.383.232 B |
| swap durable preservado | no | no |

El guard vigente devuelve `INVALID_NEW_DRAW_IDENTITY`. Frente a cada una de las
cinco roots Wave 59 configuradas, la descomposición produjo una lista de una
sola entrada: `bytes:benchmark/protocol_config.json`. Las siete copias medidas
—primaria, replay y cinco antecedentes— tienen 1.079 bytes y SHA-256
`c45a7fb245950521ceac4c6de75b51e746152f506522c52697d05bdc30673468`,
pero siete inodos distintos y `st_nlink=1`. Esto confirma tanto el diagnóstico
como la suficiencia de la excepción mínima.

El binding de `decision_select.npz` también es físicamente trazable en
`preparation_freeze.json`: path
`wave55_policy_bridge_fresh_v1/bundles/decision_select.npz` y SHA-256
`f07f6c25527501b2ea85cba809b3e70f744b8b081b23d6da6b05b02c9f37e9ee`.
La mención del primer comando equivocado no dejó artefacto durable por no haber
cruzado la transacción; el plan debe explicitar que procede de un transcript no
archivado y que no integra la autoridad de recovery ni el ledger.

## Sellado v2 antes del cambio

El orden de las líneas 85–100 es seguro y necesario. En el runner congelado,
`execute_prepared_pair()` valida primero ambas roots, reconstruye el ledger y
ejecuta `validate_new_draw_pair()` antes de source binding, score o evaluate
(`run_wave60_frozen_policy_transport.py:4284-4368`). Ante esta excepción:

- sella primaria como `INVALID_NEW_DRAW_IDENTITY`;
- sella replay como `PEER_ABORTED_PRE_TRUTH` con binding direccional;
- publica `PAIR_ABORTED_PRE_TRUTH` de forma atómica;
- conserva `any_truth_accessed=false` y `recovery_allowed=true`.

El plan no trata esa expectativa como hecho consumado: el terminal físico manda
y cualquier terminal distinto o acceso a truth detiene la vía. Ejecutar el
sellado después de cambiar el guard sería incorrecto porque el mismo par ya no
produciría el failure que debe autorizar el recovery.

## Recovery v2→v3 y ledger

La infraestructura vigente ya demuestra el mecanismo general. El validador
`recovery_pair_durable_elapsed()` exige un pair failure firmado,
`PAIR_ABORTED_PRE_TRUTH`, `any_truth_accessed=false` y
`recovery_allowed=true`, y suma preparación más fases pre-truth
(`run_wave60_frozen_policy_transport.py:4141-4184`). El preparador vuelve a
validar los dos ledgers firmados, exige continuidad primaria→replay y devuelve
el acumulado durable como débito inicial del nuevo intento
(`prepare_wave56_fresh.py:7538-7590`).

Para el fallo de identidad esperado no existen fases source/score que sumar;
por tanto v3 hereda exactamente 215,36700256168842 s antes de su propia
preparación. La nueva amendment puede reutilizar la rama tipada ya existente
para autenticar el terminal, el inventario closed-world, el escrow/draw
preservado, el config audit previo y el delta de sources. El plan acierta al
crear un schema específico porque la rama genérica actual prohíbe cambios de
sources; esa extensión queda confinada al preparador, mientras el runner sólo
cambia el clasificador de la colisión.

El test integral existente materializa y ejecuta un recovery v2→v3 con ledgers
firmados acumulados, mismo draw por bytes, namespace nuevo e inodos distintos
(`tests/test_wave60_frozen_policy_transport.py:3348-4230`). Pasó junto con el
test vigente del guard:

```text
2 passed in 17.67s
```

La recuperación propuesta no redibuja: toma la primaria v2 sellada como
autoridad, copia escrow y benchmark por bytes a roots v3 nuevas, prohíbe
`--force` y conserva v2 inmutable. Tampoco abre truth: la validación del origen
usa firmas, hashes, manifests, inventarios y metadatos; el primer acceso
semántico sigue detrás de la barrera score de ambas roots.

## Superficie y cadena R496–R500

La superficie de tres archivos es completa y mínima:

- runner: única clasificación de igualdad estática;
- preparador: schema, lineage, inventario y delta de sources del recovery;
- tests: positivo realista, ataques y regresión.

El módulo científico y el worker permanecen byte-exactos. La autoridad source
law observada contiene 13 modelos usados, 3 no usados, 1.300 tree keys y 3.900
arrays de estado; nada de esa superficie necesita cambiar para corregir el
guard.

La secuencia propuesta también cierra la autoridad sin circularidad:

```text
plan -> R496 -> terminal v2 -> implementación -> R497
     -> amendment v3 -> R498 -> config v3 -> R499
     -> preparación/ejecución v3 -> R500
```

Cada documento queda en un commit exclusivo y cada auditoría es hija directa
de su target. La amendment liga el terminal v2, R495, plan/R496 e
implementación/R497; R498 la autentica. La config liga amendment/R498, su
self-binding y los tres hashes cambiados; R499 la autentica antes de cualquier
preparación. R500 se limita a los resultados físicos. Ningún eslabón otorga
promoción científica ni decide `GO/NO-GO`.

## Condición de reauditoría

La corrección de R496-01 es puramente documental y admite reauditoría focal. Una
vez resuelta, deben mantenerse literalmente las condiciones ya escritas en el
plan: sellar v2 con los bytes congelados antes de cambiar código; abortar si el
terminal físico no es recuperable o si truth fue accedida; aplicar la excepción
sólo a igualdad de bytes del path unitario; y exigir en R497 pruebas explícitas
de manifest, pair exactness, inodos, antecedentes, lineage y presupuesto.

No modifiqué código, config ni datos. No usé ni consulté GPU.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R496",
  "scope": "STATIC_PROTOCOL_IDENTITY_GUARD_RECOVERY_PLAN",
  "target": {
    "plan_commit": "4d29440b745407206053de3119344c6eeb05ffd3",
    "plan_sha256": "7c6d4472eea18662a677b887267134c0e9ab876145762275cf9cddfc7d590929"
  },
  "technical_verdict": "REVISE",
  "findings": {
    "high": 0,
    "medium": 0,
    "low": 1
  },
  "files_modified": false,
  "gpu_used_or_queried": false
}
```
