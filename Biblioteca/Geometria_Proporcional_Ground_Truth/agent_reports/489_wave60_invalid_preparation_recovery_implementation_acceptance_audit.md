# R489 — Auditoría independiente de aceptación de la implementación `INVALID_PREPARATION`

**Dictamen técnico: `REVISE` — 0 HIGH / 1 MEDIUM / 0 LOW.**

El candidato autentica la historia completa R481–R488, conserva las tres
implementaciones rechazadas, reserva R489 para la autoridad positiva, amplía el
keyset de amendment a 35 claves y renumera amendment/config/resultados como
R490/R491/R492. Las matrices de parents, paths extra, divergencia Git/físico,
deltas cruzados y sources científicos están materializadas, y las suites focal
y regresiva pasan. Sin embargo, los 45 negativos semánticos de auditoría no
comprueban una de las precondiciones aislantes exigidas por el plan: igualdad
entre blob Git del commit alternativo, archivo físico y SHA del binding. Como
aceptan cualquier `RuntimeError`, todavía pueden pasar antes de alcanzar el
parser semántico. Por eso la evidencia no satisface íntegramente el contrato de
aceptación.

## Identidad y alcance

El target auditado es exactamente:

- commit: `fb0248f0430c640099f97e7e7c69eedc271fc9d3`;
- parent directo R488: `bb6a80d4aae74683f0a4b3513cd5d44bb9fa5c6b`;
- árbol Git: `37d41f7d3925c7b88c8e14292262d51f1177783e`;
- diff exclusivo: `1465` inserciones y `125` eliminaciones en:
  - `experiments/geometria_proporcional/prepare_wave56_fresh.py`;
  - `tests/test_wave60_frozen_policy_transport.py`.

El parent R488 es hijo directo del plan R487
`126ee2ca160d2bb3301a146b9850709c7aa78d1f`, introduce exclusivamente
`488_wave60_invalid_preparation_recovery_r487_resolution_plan_audit.md` y su
blob/archivo coincide en
`f8e7b5d1c979a8b9af9de3411137300dd7a3afbd8eb72af83378ec72b5500f01`.

Leí completos los planes vigentes R485, R486 y R487, R488 y el diff íntegro de
los dos archivos modificados; contrasté además las funciones productivas
alcanzadas por las pruebas. Los blobs target coinciden con el filesystem:

| Path | SHA-256 parent | SHA-256 target/físico |
|---|---|---|
| `prepare_wave56_fresh.py` | `b21d89af10021904347563997b3ec1e13292558cfc30519bd2c5f4d8cc8d7f32` | `0b52f06012c92feef02bf7a93892bb35c34c7196574d94b465c3796cbb6e16a4` |
| `test_wave60_frozen_policy_transport.py` | `979054582ca1a61e6834f19d65e8d39954b431e1482f2c72a14af830c0363fdd` | `f1d3d9668d5d09bd5fa87618fb7db42d7ac49550cf763f820d39d345cdf7f80a` |

`git diff --check` quedó limpio. No se modificó código, config, amendment ni
otro artefacto durante esta auditoría.

## Frontera científica y lineage productivo

Módulo, runner y worker son byte-exactos entre R475
`9f1a229d9c0ccb5e46b921e6c92281becc317139`, el target y el filesystem:

- módulo: `46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65`;
- runner: `1c778c3e60c1bbcebeb5c83430601a7c0b148e447528195f1dec4296322825aa`;
- worker: `c6c5c832633d9ed8a99079dd4fc4f5677073f7e1883a204fa6bb41f6d646bac7`.

`validate_wave60_invalid_preparation_implementation_suffix()` incorpora los
ocho objetos posteriores a las 27 claves fijadas antes de R485, valida las tres
implementaciones rechazadas y sus auditorías, los tres planes sucesivos, R488
PASS y el sucesor final R489. Cada implementación exige parent directo, delta
exclusivo preparer/test y `old_sha256` calculado contra R475; la final exige
además igualdad de cada source científico contra R475
(`prepare_wave56_fresh.py:510-969`).

El validador de amendment exige exactamente 35 claves, hardcodea correctamente
R485, el commit real R486 `2049eff3b411024e6b4fd444f2b975ae76c27f3e`,
R487 y R488, y pasa cada objeto al helper ampliado
(`prepare_wave56_fresh.py:4774-5380`). La autoridad positiva de implementación
es R489; amendment y config se autentican como R490 y R491 respectivamente
(`prepare_wave56_fresh.py:1030-1102,5395-5425`). R492 permanece como numeración
reservada a la futura auditoría de resultados; no existe todavía un artefacto
de resultados que deba ligarse desde este commit. La secuencia es lineal y no
circular.

## Cobertura que sí quedó implementada

La fixture positiva construye las nueve auditorías consecutivas R481–R489 con
sus scopes, targets, verdicts y conteos propios. R481, R483, R485, R486 y R487
permanecen `REVISE`; R482, R484, R488 y R489 son `PASS`. El helper `audit_specs`
enumera las nueve y genera cinco reportes alternativos por cada una, uno por
`audit_id`, `target`, `scope`, `technical_verdict` y `findings`
(`tests/test_wave60_frozen_policy_transport.py:5158-5391`).

Los bindings externos commit/path/SHA se alteran independientemente para las
nueve auditorías (`tests/test_wave60_frozen_policy_transport.py:5825-5846`). La
matriz `suffix_steps` cubre las 18 transiciones desde la primera implementación
rechazada hasta R489: para cada una crea un commit con exactamente los paths
esperados más un único path adicional, y otro con parent saltado pero delta
correcto (`tests/test_wave60_frozen_policy_transport.py:5877-6105`).

La divergencia real blob↔físico se prueba separadamente para un documento y una
auditoría: el HEAD temporal contiene bytes físicos nuevos y su SHA, mientras el
commit ligado conserva el blob anterior; las precondiciones se afirman antes
del rechazo por `blob differs`
(`tests/test_wave60_frozen_policy_transport.py:5848-5875`).

Los deltas `old_sha256`, `new_sha256` y `path` se alteran para preparer y test en
las tres implementaciones rechazadas y la final. Cada binding también recibe el
`new_sha256` de cada una de las otras tres implementaciones, comprobando primero
que no coincidan (`tests/test_wave60_frozen_policy_transport.py:5100-5130`).

Los tres sources científicos tienen negativos independientes con cadenas
secundarias reales. Cada cadena muta un source después de R475 y reconstruye
todo el sufijo con parents directos y deltas exclusivos; el commit final sigue
cambiando sólo preparer/test y el rechazo llega a `crossed scientific source`
(`tests/test_wave60_frozen_policy_transport.py:5397-5823`). Esto ya no se
sustituye por alterar la lista declarativa.

La positiva de transaction/provenance usa únicamente R489, excluye R483 y R485,
y exige igualdad del objeto completo en `generation_receipt.json`,
`preparation_freeze.json` y `preparation_receipt.json`. Verifica la firma
Ed25519 real y el record exacto del receipt mediante path, bytes y SHA físico
(`tests/test_wave60_frozen_policy_transport.py:2034-2040,2132-2171`). La
attestation conserva su schema: la provenance queda firmada transitivamente por
el receipt y no se duplica dentro del payload.

## Finding MEDIUM — falta probar blob Git = físico = binding en cada negativo semántico

El plan R486 exige que, antes de invocar el validador para cada campo de cada
auditoría, el test compruebe parent, paths cambiados, hash físico/blob y
diferencia semántica única
(`WAVE_60_INVALID_PREPARATION_RECOVERY_R486_RESOLUTION_PLAN.md:88-120`). La
implementación comprueba:

- diferencia de una sola clave semántica;
- delta exclusivo del path de reporte;
- parent directo correcto;
- path externo correcto;
- `sha256(archivo físico) == binding.sha256`.

Pero no contiene la aserción restante:

```text
git_blob_sha256(repo, bad_commit, relative)
  == file_sha256(repo / relative)
  == case[binding][sha_field]
```

El hueco está en el loop común a los 45 casos
(`tests/test_wave60_frozen_policy_transport.py:5315-5391`). La construcción
normal mediante `git add`/`git commit` hace probable esa igualdad, pero no la
convierte en evidencia testada. El bloque termina con
`pytest.raises(RuntimeError)` sin restringir mensaje ni fase; por tanto una
divergencia accidental blob/físico también haría pasar el negativo antes de que
`parse_wave60_audit_report()` o `parse_wave60_revise_audit_report()` evalúen el
campo semántico. Los dos casos dedicados de divergencia prueban justamente el
camino opuesto y no sustituyen esta precondición por auditoría/campo.

La corrección es local a la suite: añadir dentro del loop la igualdad explícita
blob del `bad_commit` = archivo físico = binding, antes del `pytest.raises`.
No hace falta modificar producción salvo que esa nueva precondición revele otra
divergencia. Después debe repetirse Wave 60 y la regresión Waves 56–59.

## Pruebas, recursos y limpieza

Todo se ejecutó CPU-only con `CUDA_VISIBLE_DEVICES=''`, sin usar ni consultar
GPU, bajo un único root temporal dedicado en `/mnt/m2-1TB`:

```text
Wave 60 completa:
147 passed in 311.61s
wall 5:12.89; max RSS 915768 KiB; process swaps 0

Regresión explícita Waves 56–59 (nueve archivos test_wave56..test_wave59):
336 passed, 1 skipped in 407.77s
wall 6:49.04; max RSS 1065592 KiB; process swaps 0
```

El root `/mnt/m2-1TB/.wave60-r489-audit.pcn5Iy` fue inventariado antes de
borrarlo: modo `0700`, owner `0:0`, `5.9 GiB` y `14,662` archivos. Se eliminó
por su path exacto y se verificó su ausencia. El swap global del host se mantuvo
en aproximadamente `22 GiB`; `/usr/bin/time` registró cero swaps para ambas
corridas. Este informe es el único archivo creado y no forma parte del target.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R489",
  "scope": "INVALID_PREPARATION_RECOVERY_IMPLEMENTATION",
  "target": {
    "implementation_commit": "fb0248f0430c640099f97e7e7c69eedc271fc9d3"
  },
  "technical_verdict": "REVISE",
  "findings": {
    "high": 0,
    "medium": 1,
    "low": 0
  },
  "files_modified": false,
  "gpu_used_or_queried": false
}
```
