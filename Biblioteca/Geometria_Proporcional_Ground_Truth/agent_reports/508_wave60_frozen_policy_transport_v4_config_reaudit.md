# R508 — Reauditoría final de configuración Wave 60 v4

## Dictamen técnico: PASS

La configuración congelada en
`b156f6857eaa36edc8bda9e7687de7b8e1ea9721` corrige la composición de
autoridad que bloqueó R504 sin cambiar el experimento. El refreeze liga el
amendment suplementario R504/R507, reserva R508 como auditoría final y actualiza
únicamente el self-binding y los hashes de preparer/test autorizados por R506.
El roster permanece cerrado en ocho sources y la transición v3→R506 conserva
la partición exacta de dos sources cambiados y cinco invariantes.

No encontré findings altos, medios ni bajos. Esta auditoría no ejecutó el
experimento, no creó `attempt_v4`, no abrió truth para scoring o evaluación y
no usó ni consultó GPU. El dictamen es estrictamente técnico y pre-ejecución;
no constituye una decisión científica ni un `GO/NO-GO`.

## Identidad, canonicalidad y closed world

- Target: `b156f6857eaa36edc8bda9e7687de7b8e1ea9721`, hijo directo único de R507
  `1bfc0de3ebed7d66a8378e4ec24c38bed52aa37a`.
- El commit modifica exclusivamente
  `experiments/geometria_proporcional/configs/wave60_frozen_policy_transport.json`.
- El archivo físico y el blob Git son byte-exactos, tienen 11.599 bytes y
  SHA-256 `191483d2909c3a95a1e82488e1834b55c849763f55c72549f07f2c4cf81d6416`.
- La serialización es exactamente `json.dumps(..., indent=2, sort_keys=True,
  ensure_ascii=True) + "\n"`.
- `validate_pre_draw_config()` y `validate_prospective_config()` aceptan el
  objeto con keysets exactos. No existe `hard_set_tau` en la config canónica.
- El self-binding, calculado después de sustituir sólo su propia entrada por 64
  ceros y hashear el JSON canónico compacto, es
  `eab40e2d34cfcd532437c5e7567ac94b7988a46afb865bab84728fa90735e810`.

## Delta mínimo respecto de la config rechazada

La comparación estructural contra la config R504 rechazada, commit
`201f9257c1e016b925719f6917dd1a9298ca493b`, produce exactamente diez cambios
leaf y ningún otro:

1. cinco bindings de `attempt.recovery` pasan del amendment inicial/R503 al
   suplemento `fd1270b`/R507: path, SHA-256, audit commit, audit path y audit
   SHA-256;
2. `final_audit` pasa de R504 a R508 en sus dos campos;
3. `source_sha256` actualiza self-binding, preparer y test.

Todo el resto es byte-estructuralmente igual a la candidata R504: namespace
v4, draw preservado, origen v3, source law, contrato `hard_set_tau=0.5`,
features, políticas mean/tail, penalty, bootstrap, seeds, batch size, splits,
CPU/threads y presupuesto acumulativo máximo de 900 s. En particular, no se
introduce un nuevo draw, scoring, fit, selección adaptativa, recalibración ni
apertura anticipada de truth.

## Binding suplementario y cadena de autoridad

La config liga exactamente:

- amendment suplementario `fd1270b084d4ebcbd6b4b675169665ce5f9c207b`,
  SHA-256 físico/Git
  `51f3972a5f73178b1594376a92ac252d416736901870c44205a025ebac09d16b`;
- R507 `1bfc0de3ebed7d66a8378e4ec24c38bed52aa37a`, SHA-256 físico/Git
  `ad0ebd32f1c2df415fcdf12786e92577a9cb54077a4e2f60ad041a7f1e90fea2`;
- schema suplementario cerrado
  `wave60-hard-set-authority-r504-recovery-amendment-v1`, estado `APPROVED` y
  recovery kind `HARD_SET_AUTHORITY`.

La validación recompuso la cadena completa sin sustituir terminales previos:
R502 REVISE `00b433a6` con `0/1/0`, correcciones `410a9189` y `e29069f6`,
implementación aceptada `7eba44b9`, R502 PASS `7182d975`, amendment inicial
`3a21c039`, R503 PASS `2b98cfa8`, config rechazada `201f9257`, R504 REVISE
`b70ee7e8` con `1/0/0`, plan `0e2768b6`, R505 PASS `e84bfe70`, corrección R506
`464ceb59`, R506 PASS `fb5dfa64`, suplemento `fd1270b0`, R507 PASS `1bfc0de3`
y refreeze `b156f685`. Los commits documentales son exclusivos; los commits de
implementación modifican sólo los sources declarados por su capa.

## Roster, partición y hashes físicos/Git

`required_execution_sources` contiene ocho paths únicos y su conjunto coincide
exactamente con las ocho claves de `source_sha256`:

| Clase | Source | SHA-256 final |
|---|---|---|
| self-bound | config | `eab40e2d34cfcd532437c5e7567ac94b7988a46afb865bab84728fa90735e810` |
| changed | preparer | `0dd0f3389b2db1011ce95c916a37faf4c3898460c2d30fba8f8339c5075b92c8` |
| changed | test | `d8ca7d06848eb17091e743aaf025abcefa9cc333489b2c6641cd6bab9cab6960` |
| invariant | runner | `b35cd563f715bdff9b6e7489ac04712c728673563898d4a6aebf0144d4a50261` |
| invariant | módulo científico | `46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65` |
| invariant | worker | `c6c5c832633d9ed8a99079dd4fc4f5677073f7e1883a204fa6bb41f6d646bac7` |
| invariant | R475 | `e5c49ca16506469ac5099c2a3f1992819c1a54fed790f1638a82e93b9b1d9996` |
| invariant | R476 | `497bb87f7e3677c4dae11a0281e0362fa4d1ef4d6ba60252f01cdc1f2c0a8a30` |

El universo no autorreferente queda cerrado como `2 changed + 5 invariantes`,
sin faltantes, extras, cruces ni intersección. Para preparer/test, `old_sha256`
coincide con la config v3/R500 y `new_sha256` coincide simultáneamente con el
blob R506, la config final y el filesystem. Para los cinco invariantes, el hash
v3 coincide simultáneamente con `prior_source_sha256`, el blob R506, la config
final y el filesystem. Esto preserva byte-exactos módulo, runner, worker, R475
y R476 y evita la regresión al runner pre-v3 que causó R504-01.

## Frontera de intento y truth

El path
`data/geometria_proporcional/wave60_frozen_policy_transport_attempt_v4` no
existe como archivo, directorio ni symlink antes de congelar este informe. Las
comprobaciones estáticas y pytest leyeron config, código, informes, blobs Git y
fixtures sintéticos; no ejecutaron el experimento ni scoring. El preflight
normativo posterior al commit de este informe es read-only, no crea output y
sólo autentica las fuentes y artefactos históricos permitidos por el protocolo.

## Comprobaciones CPU-only

Todas las corridas usaron `CUDA_VISIBLE_DEVICES=''`,
`PYTHONDONTWRITEBYTECODE=1`; pytest deshabilitó plugins externos y cache y usó
un basetemp propio con parent `0755` bajo `/mnt/m2-1TB`.

| Check | Resultado | wall | max RSS | swaps |
|---|---:|---:|---:|---:|
| checker de config/Git/schema/delta/lineage/partición | PASS | 2.11 s | 826.628 KiB | 0 |
| `pytest -k 'hard_set_v4 or r504_resolution'` | 20 PASS / 154 deselected | 88.48 s | 887.628 KiB | 0 |
| preflight real contra HEAD auditado | PASS | 3.22 s | 957.680 KiB | 0 |

El informe se congeló primero porque el preflight exige que R508 sea el `HEAD`
exclusivo hijo de la config. La llamada directa a `preparation_preflight()`
contra ese HEAD retornó contrato y no entró en `run_preparation_transaction()`.
Resultado exacto: `state=PREFLIGHT_PASS`, config SHA-256
`191483d2909c3a95a1e82488e1834b55c849763f55c72549f07f2c4cf81d6416`,
`source_count=8`, `upstream_count=9`, `historical_preflight.status=PASS` y
`n_tokens=384`; para seeds 17, 29 y 43, los cinco campos `array_exact`
(`choice_logits`, `choice_pair_token`, `pair_token`, `set_logits`, `target`)
fueron `true`. El exit status fue `0`.

El preflight autenticó once inputs históricos declarados: visible val,
authorized labels val, protocol, normalizer, split manifest, tres checkpoints
y tres arrays val-monitor. No ejecutó scoring, fit, draw ni evaluación, y no
creó output. Esta publicación se enmienda, sin binding descendiente, para
incorporar el resultado. Una reconfirmación idéntica contra el HEAD final
enmendado es condición de cierre; ante cualquier diferencia se corrige este
dictamen antes de entregar.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R508",
  "scope": "CONFIG",
  "target": {
    "config_commit": "b156f6857eaa36edc8bda9e7687de7b8e1ea9721",
    "config_sha256": "191483d2909c3a95a1e82488e1834b55c849763f55c72549f07f2c4cf81d6416"
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
