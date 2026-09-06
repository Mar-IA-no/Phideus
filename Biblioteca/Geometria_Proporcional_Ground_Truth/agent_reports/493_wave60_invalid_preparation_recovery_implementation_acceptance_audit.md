# R493 — Auditoría independiente de aceptación de la implementación `INVALID_PREPARATION`

**Dictamen técnico: `PASS` — 0 HIGH / 0 MEDIUM / 0 LOW.**

El sucesor resuelve el finding MEDIUM de R491 sin ampliar la frontera
autorizada. El guard productivo de la auditoría de amendment exige ahora R494,
y una prueba pública con repositorio Git sintético real acepta R494 y rechaza
R490 y R492. El lineage incorpora la implementación rechazada y la auditoría
R491, el plan R491 y R492; reserva R493 para esta aceptación, R494 para la
amendment, R495 para config y R496 para resultados. Las matrices cuantitativas,
la procedencia positiva y la invariancia científica coinciden con el plan
R491 aprobado por R492.

## Identidad y alcance

El target auditado es exactamente:

- commit: `7d8143deb283928faa82e99e390f79d16f301867`;
- parent directo R492: `9eeb7b0a54ad1bd73ab0df0ee768ef7c24433deb`;
- árbol Git: `b17963be4cf1c42eb67af31872f107af19b4b1e2`;
- diff exclusivo: `579` inserciones y `60` eliminaciones en:
  - `experiments/geometria_proporcional/prepare_wave56_fresh.py`;
  - `tests/test_wave60_frozen_policy_transport.py`.

El plan R491 fue introducido exclusivamente por
`e7af03ac590e90098ff94adb09cd0181f5a083e0`, hijo directo de R491, con
SHA-256 Git/físico
`d699814fa56de9ed05ce4b457ee4a231af379ee3c80c815c55b9c1a2c6cc1ce5`.
R492 fue introducido exclusivamente por el parent del target, es hijo directo
del plan y su SHA-256 Git/físico es
`1b0daedf7ff27c32f0fa56f36d5fbda34fe2713cd4d7e4be56ba82076cfc7ab2`.

Leí completos el plan R491, R492, el diff y las secciones vigentes afectadas.
`git diff --check` quedó limpio. Los hashes del blob target coinciden con los
archivos físicos:

| Path | SHA-256 target/físico |
|---|---|
| `experiments/geometria_proporcional/prepare_wave56_fresh.py` | `3d0532cd840461fa07ce1fccc81d5c1b085b8c00397f6162c88f435e0da23ac9` |
| `tests/test_wave60_frozen_policy_transport.py` | `12611bd90e13a5e1ff7d601b4b654802e021f90b9f3c4be8fe9f38183e178e9e` |

No modifiqué código, config, amendment ni fuentes científicas; este informe es
el único archivo creado y no forma parte del target.

## Cierre del finding R491

`validate_wave60_invalid_preparation_amendment_audit()` es una función pública
del preparer que delega en el validador canónico completo. Fija scope
`INVALID_PREPARATION_RECOVERY_AMENDMENT`, target `amendment_sha256`, parent
directo de la amendment y `expected_audit_id="R494"`
(`prepare_wave56_fresh.py:510-528`). El ramal productivo la invoca en lugar del
guard anterior, de modo que no quedan dos políticas divergentes.

`test_invalid_preparation_amendment_audit_requires_r494()` crea una amendment y
auditorías en un repositorio Git temporal, comprueba commit exclusivo y parent
directo, acepta R494 y reconstruye desde el mismo parent casos R490 y R492 que
deben fallar por `audit id drifted`
(`test_wave60_frozen_policy_transport.py:4549-4631`). No usa mocks,
`monkeypatch`, introspección ni sustitución del parser.

## Lineage, matrices y partición de autoridades

El validador del sufijo autentica ahora, después de R490, la implementación
`4e50425b`, R491 como `REVISE 0/1/0`, el plan R491 y R492 como `PASS 0/0/0`;
recién entonces admite la implementación sucesora auditada por R493
(`prepare_wave56_fresh.py:1047-1213`). La amendment futura es closed-world con
43 claves: añade exactamente los cuatro bindings de esa historia y los liga a
commits, paths y hashes concretos (`prepare_wave56_fresh.py:5035-5083,5668-5755`).

La prueba de sufijo cubre exactamente:

- seis implementaciones, con negativos de path, hash viejo, hash nuevo y
  cruces contra cada otra implementación;
- trece auditorías por cinco campos semánticos, para 65 negativos;
- precondición explícita `blob_sha256 == physical_sha256 == binding_sha256`
  antes de cada rechazo semántico;
- 26 etapas en `alternate_steps` y 26 en `suffix_steps`, incluidas las ramas de
  parent saltado y path adicional.

Estas cantidades están afirmadas en el propio test
(`test_wave60_frozen_policy_transport.py:5382-5418,5450-5758,6043-6254,6494-6538`).
Las tres mutaciones de source law reconstruyen el sufijo completo y son
rechazadas sin confundir cambio científico con cambio del preparer.

La positiva de transaction/provenance usa exclusivamente R493 y afirma que
R489 y R491 no aparecen en el objeto positivo. El objeto completo queda igual
en `generation_receipt.json`, `preparation_freeze.json` y
`preparation_receipt.json`, y la attestation Ed25519 conserva el binding
transitivo por path, bytes y SHA-256
(`test_wave60_frozen_policy_transport.py:2034-2042,2136-2171`).

La autoridad final queda particionada correctamente: R493 autentica sólo la
implementación de recuperación, R494 la amendment, R495 la config v2 y R496
permanece reservado a resultados. La fixture futura materializa R493/R494/R495
y comprueba que source law sigue anclada en R475 mientras preparer y test
provienen de R493 (`prepare_wave56_fresh.py:1295-1339`;
`test_wave60_frozen_policy_transport.py:6631-6759`).

## Frontera científica

Módulo, runner y worker no cambiaron entre parent y target, y sus blobs target
coinciden con los archivos físicos y con los hashes congelados por R491:

- módulo: `46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65`;
- runner: `1c778c3e60c1bbcebeb5c83430601a7c0b148e447528195f1dec4296322825aa`;
- worker: `c6c5c832633d9ed8a99079dd4fc4f5677073f7e1883a204fa6bb41f6d646bac7`.

El cambio no altera source law, draw, modelo, features, políticas, thresholds,
roster, estimando, presupuesto, schema de amendment, las doce claves de
`attempt.recovery` ni la ausencia canónica de `hard_set_tau`.

## Pruebas, recursos y limpieza

Todo se ejecutó CPU-only con `CUDA_VISIBLE_DEVICES=''`, sin usar ni consultar
GPU, y con `/usr/bin/time -v`:

```text
Wave 60 completa:
148 passed in 387.87s
wall 6:29.13; max RSS 920744 KiB; process swaps 0

Regresión explícita Waves 56–59 (nueve archivos):
336 passed, 1 skipped in 413.71s
wall 6:55.01; max RSS 1065644 KiB; process swaps 0
```

Ambas suites usaron basetemps propios bajo
`/mnt/m2-1TB/wave60-r493-audit.jS89uPc0`. Antes de eliminarlo, el root fue
inventariado como directorio `0700`, owner `0:0`, tamaño `5.9G`, 15.035
archivos y 7.038 directorios. Se eliminó por ese path exacto y se verificó su
ausencia.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R493",
  "scope": "INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
  "target": {
    "implementation_commit": "7d8143deb283928faa82e99e390f79d16f301867"
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
