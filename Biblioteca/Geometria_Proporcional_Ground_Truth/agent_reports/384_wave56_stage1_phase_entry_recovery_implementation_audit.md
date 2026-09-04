# R384 — Auditoría técnica independiente de la implementación de recovery para la entrada a fase de Wave 56 Stage 1

**Implementation commit:** `7b37b5381b0c7540e86de2d53001903475d321ab`
**Preparer SHA-256:** `e4c79cc8129ae827da28cfa719ebbd3e31ccc9ca926ae6a15f48f75b2c236624`
**Runner SHA-256:** `a9f2cd4e1826b9d1290d48faa0d5ead5cd48468488164462b1cce7c859ffde30`
**Test SHA-256:** `8c245e2a881649335ce5894502c4174fc377864af082216bcb9b1c05df0e64a4`
**Result:** `REVISE`

## Dictamen ejecutivo

El parche corrige correctamente el fallo de entrada a fase: conserva los tres
hashes de split del freeze, agrega todos los miembros `visible/` autenticados
por el manifest, exige concordancia entre ambas fuentes cuando se solapan y
mantiene una igualdad cerrada contra el inventario físico. El primario oficial
reprodujo el fallo anterior por un único miembro omitido,
`calibration_null.jsonl`, y el guard nuevo aceptó el paquete público intacto.
La focal completa terminó con `34 passed` en CPU-only; los negativos físicos
rechazaron calibración faltante, mutada y un visible extra, y un probe adicional
rechazó el conflicto entre manifest y freeze.

La implementación no puede aprobarse todavía porque conserva un bypass de
autoridad documental que el plan P3 prohíbe explícitamente. El validador exige
una cabecera canónica con `**Result:** `PASS`` pero no inspecciona la decisión
final del informe. Un reporte con esa cabecera y una sección final cuyo dictamen
es `REVISE` fue aceptado por `_require_report_fields`. Por tanto, después de
introducir A3, J3 y F3, la ejecución podría superar el preflight aunque una de
las auditorías concluyera materialmente `REVISE`. P3 establece que ese estado no
autoriza ejecución aunque la cabecera accidental diga otra cosa y exige
coherencia entre cabecera, cuerpo y decisión.

El resultado es `REVISE`. El finding no afecta la corrección del inventario,
pero sí el cierre fail-closed de la nueva cadena de autorización y debe
resolverse antes de crear el amendment v3.

## Alcance y restricciones

Leí completos P3, R381, R382, R383 y las versiones vigentes del preparador, el
runner y el test focal. Inspeccioné el diff exacto `parent..I3`, los objetos Git,
parents, paths, blobs y hashes. Sobre el primario oficial `PREPARED` usé sólo
metadata y artefactos públicos; sustituí explícitamente el inventario de fuentes
por el congelado y el inventario de inferencia por su mapa público para aislar
el guard sin abrir logits. No abrí escrow, secretos, truth, miembros sellados,
labels, oracle ni resultados oficiales; no usé web o GPU y no ejecuté recovery,
replay ni fases oficiales.

Todas las pruebas y probes se ejecutaron con:

```text
CUDA_VISIBLE_DEVICES=''
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
PYTHONDONTWRITEBYTECODE=1
```

## Identidad, diff y blobs

`I3` tiene un único parent,
`ad0359d548f7e7c8f820f2c7183231f4b76e47ca`, que introduce exclusivamente
R383 y desciende directamente de P3. El plan mantiene SHA-256
`2beeded884380087dc850f410da4516cc805e54ae12aefc0678c2489c4aa5e49`
tanto en P3 como en el parent de I3.

El diff de I3 contiene exactamente los tres paths autorizados:

- `experiments/geometria_proporcional/prepare_wave56_fresh.py`;
- `experiments/geometria_proporcional/run_wave56_contextual_gate.py`;
- `tests/test_wave56_preoracle_recovery.py`.

El diff suma respectivamente `31/7`, `13/0` y `67/8`
inserciones/eliminaciones, y `git diff --check` no reporta errores. Los hashes
del worktree y de los blobs resueltos directamente desde I3 coinciden con los
tres valores de la cabecera. `HEAD == I3` y el worktree estaba limpio antes de
crear este informe.

El path canónico
`experiments/geometria_proporcional/configs/wave56_stage1_phase_entry_recovery_amendment_v3.json`
y los paths previstos de R384 y R385 estaban ausentes y tenían cero commits de
introducción. El preparador usa el nuevo path, el plan P3 y el schema separado
`wave56-stage1-phase-entry-recovery-amendment-v1`.

## Composición del visible y cierre físico

En `validate_prepared_package`, las entradas iniciales de
`expected_visible` siguen viniendo de `preparation_freeze.visible_sha256` para
`train`, `val` y `lockbox`. Luego cada miembro del manifest cuyo path comienza
con `visible/` aporta su path relativo y SHA-256. Si una entrada compartida no
coincide, el guard aborta; finalmente exige igualdad exacta con
`hash_inventory(benchmark/visible)`.

La reproducción pública sobre el primario oficial produjo:

```text
old_equals_physical = false
old_missing_vs_physical = ["calibration_null.jsonl"]
shared_split_conflicts = {}
new_equals_physical = true
actual_visible_names = [
  "calibration_null.jsonl", "lockbox.jsonl", "train.jsonl", "val.jsonl"
]
```

Con monkeypatch explícito de `execution_source_hashes`, del inventario de
inferencia y de los frozen rows ajenos al guard, la validación pública completa
del primario terminó `PASS`. No se leyó ningún logit. Un fixture sintético
intacto también pasó; al cambiar sólo el hash de `train` en el freeze, manteniendo
el miembro físico y el manifest concordantes, abortó con
`visible split differs between manifest and preparation freeze`.

La focal física confirma además que `calibration_null.jsonl` existe en el
paquete generado, acepta el paquete íntegro, rechaza un archivo visible extra,
rechaza calibración ausente y rechaza una mutación same-size antes de cualquier
transacción de fase. `hash_inventory` conserva el rechazo de symlinks y la
igualdad final conserva el rechazo de cualquier archivo no manifestado.

## Dos deltas de fuente, schema y DAG futura

El freeze público del origen contiene 26 fuentes y el conjunto de paths coincide
exactamente con `required_execution_sources`. Frente al árbol actual hay dos y
sólo dos cambios:

```text
experiments/geometria_proporcional/prepare_wave56_fresh.py
experiments/geometria_proporcional/run_wave56_contextual_gate.py
```

Los hashes old del contrato coinciden con los blobs del commit de origen
`51aae0715dfe8318f5333c568429c8e9af59f866`; los hashes new coinciden con I3.
`_validate_contract_delta` exige igualdad de campos y conjunto de fuentes,
rechaza cualquier conjunto de deltas distinto de `{preparador, runner}` y liga
los triples `path/old_sha256/new_sha256` de ambos. El test permanece ligado al
blob de I3 pero no se confunde con una fuente del contrato.

El validador v3 conserva el parent directo `I3 → A3 → J3 → F3`, la introducción
única de los tres artefactos posteriores, commits documentales exclusivos,
`HEAD == F3`, limpieza global, el diff post-I3 cerrado, blobs exactos de
preparador/runner/test y el hash del runner en la cabecera de A3. También
mantiene los paths de auditoría limitados a Markdown directo bajo el directorio
canónico.

## Finding material

### F1 — Alto — Una cabecera `PASS` puede ocultar una decisión final `REVISE`

P3 dice en sus líneas 180–183 que un `REVISE` no autoriza ejecución aunque la
cabecera accidental diga otra cosa, y en 215–218 exige que ambas auditorías sean
coherentes en cabecera, cuerpo y decisión. Sin embargo,
`_require_report_fields` sólo valida el layout inicial y la unicidad de las
líneas de cabecera (`prepare_wave56_fresh.py:755-789`). Las llamadas para A3 y
F3 proporcionan el campo literal `**Result:** `PASS``
(`prepare_wave56_fresh.py:1020-1030` y `1054-1062`), pero ninguna comprueba la
sección terminal de decisión.

El probe adversarial creó un Markdown canónico con la cabecera completa
esperada y:

```markdown
## Decision

`REVISE`. Material blocker remains.
```

`_require_report_fields` lo aceptó (`contradictory_body_probe=ACCEPTED`). Los
negativos existentes no cubren este caso: cambian la propia cabecera a
`REVISE` y agregan otra línea `**Result:** `PASS`` o esconden toda la
attestation en fence/comentario (`tests/test_wave56_preoracle_recovery.py:276-290,
369-383`). Ninguno conserva una cabecera válida mientras contradice el dictamen
final.

El costo de corrección es acotado: definir y validar una decisión terminal
canónica para ambos reportes, ligada al mismo resultado de la cabecera, y agregar
negativos end-to-end para A3 y F3 con cabecera `PASS` y decisión `REVISE`. El
detalle sintáctico debe evitar inferir el dictamen por menciones incidentales en
el cuerpo; una sección final explícita y parseable es preferible a buscar la
palabra `REVISE` en todo el documento.

## Pruebas ejecutadas

- Focal completa CPU-only:
  `venv/bin/python -m pytest -q -p no:cacheprovider tests/test_wave56_preoracle_recovery.py`
  — `34 passed in 15.65s`, sin fallos ni skips.
- Probe público del primario `PREPARED` con inventarios monkeypatcheados de
  forma explícita — fallo previo reproducido por una única calibración omitida;
  composición nueva exacta y guard actual `PASS`.
- Probe sintético de conflicto split manifest/freeze — rechazado.
- Probe adversarial de decisión documental contradictoria — aceptado, confirma
  F1.
- Git — parent único, tres paths exactos, hashes de blobs/worktree, freeze de P3,
  paths v3 libres, `git diff --check` y source delta exacto.

## Preservación de no-redraw y límites

La implementación conserva que un amendment no puede autorizar un primary
fresco, que recovery/replay rechazan `keys_override`, que las claves provienen
exclusivamente del escrow durable y que el origen se revalida antes de extraer
claves y después de regenerar. La focal mantuvo el monkeypatch que hace fallar
cualquier llamada a `secrets.token_bytes`, verificó replay exacto y conservó
los negativos de whitelist físico y manifest del primary. No encontré otro
bypass material en el guard de visible, los dos deltas, blobs, paths, DAG o
no-redraw.

Este dictamen es técnico y pre-oráculo. No valida contenido secreto ni resultados
oficiales, no autoriza ejecutar recovery o fases y no constituye `GO/NO-GO`
científico.

## Decisión

`REVISE`. La corrección de `calibration_null.jsonl` y el schema v3 están bien
implementados, pero la cadena no es fail-closed frente a una auditoría cuya
cabecera diga `PASS` y cuya decisión final diga `REVISE`. Debe cerrarse F1 y
repetirse la focal y el probe adversarial antes de introducir el amendment v3.
