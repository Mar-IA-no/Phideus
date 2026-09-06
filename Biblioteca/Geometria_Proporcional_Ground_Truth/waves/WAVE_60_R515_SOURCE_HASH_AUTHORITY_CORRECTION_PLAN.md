# Ola 60 — corrección de autoridad del hash fuente observado después de R515

> **Estado:** `PRE-IMPLEMENTATION / R515-PASS-SUPERSEDED-FOR-AUTHORIZATION / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-06
> **R514:** commit `574f79810e5283dec0478c8c1c53e3496c808e97`, SHA-256 `9f96084160627a4c9102949ff413d98b6c28a22a20bf2b6c4e115e44cc888b78`
> **R515:** commit `8b5fb79cb992034021698e7f36da8779d50824d2`, SHA-256 `e81fbdd8b5067d2a9bcb2c0aedbac792bd36beadd8813860e0da32c215a6fd3e`, `PASS 0/0/0`

## 1. Finding posterior a R515

El primer test del borrador de implementación comparó el mapa normativo de
R514 contra `config.source_sha256` y abortó. R514 transcribió el hash de
`src/geometria_proporcional/wave60_frozen_policy_transport.py` como:

```text
46e31fa1096ab4e5a0c5115fc4f16922e165a21dbe47016c42953bf345116c65
```

La autoridad triple —config R508, archivo físico y blob Git ligado— coincide
en cambio en:

```text
46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65
```

La diferencia es exactamente un carácter (`fc4f` frente a `fc4e`). R515
declaró `8/8 PASS` sin detectar esa divergencia; por ello su informe se conserva
como evidencia histórica, pero su PASS no autoriza R516 bajo el plan anterior.
No se reescriben R514 ni R515.

## 2. Autoridad corregida

Este documento reemplaza únicamente el valor erróneo. Los otros siete source
bindings, los seis target hashes, la config física/self-binding, la tabla de
self-manifests, la normalización semántica, los keysets y la identidad física
de R514 permanecen normativos sin cambios.

El mapa exacto de ocho sources que debe aceptar el checker es el mapa físico de
`experiments/geometria_proporcional/configs/wave60_frozen_policy_transport.json`
en commit `b156f6857eaa36edc8bda9e7687de7b8e1ea9721`:

| Path | SHA-256 / self-binding |
|---|---|
| `src/geometria_proporcional/wave60_frozen_policy_transport.py` | `46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65` |
| `experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py` | `b35cd563f715bdff9b6e7489ac04712c728673563898d4a6aebf0144d4a50261` |
| `experiments/geometria_proporcional/_wave60_phase_worker.py` | `c6c5c832633d9ed8a99079dd4fc4f5677073f7e1883a204fa6bb41f6d646bac7` |
| `experiments/geometria_proporcional/prepare_wave56_fresh.py` | `0dd0f3389b2db1011ce95c916a37faf4c3898460c2d30fba8f8339c5075b92c8` |
| `tests/test_wave60_frozen_policy_transport.py` | `d8ca7d06848eb17091e743aaf025abcefa9cc333489b2c6641cd6bab9cab6960` |
| `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/475_wave60_source_law_recovery_implementation_reaudit.md` | `e5c49ca16506469ac5099c2a3f1992819c1a54fed790f1638a82e93b9b1d9996` |
| `Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/476_wave60_source_law_recovery_authority_audit.md` | `497bb87f7e3677c4dae11a0281e0362fa4d1ef4d6ba60252f01cdc1f2c0a8a30` |
| `experiments/geometria_proporcional/configs/wave60_frozen_policy_transport.json` | `eab40e2d34cfcd532437c5e7567ac94b7988a46afb865bab84728fa90735e810` |

El test debe exigir simultáneamente igualdad del mapa completo con la config,
hash físico de los siete sources no autorreferentes y self-binding canónico de
la config. Una divergencia de un carácter, aunque aparezca en un plan auditado,
es rechazo.

## 3. Cadena definitiva revisada

```text
R515 PASS histórico insuficiente
  -> R516 este plan correctivo, exclusivo, parent R515
  -> R517 auditoría del plan, exclusiva, parent R516
  -> R518 implementación checker+test, exclusiva, parent R517
  -> R519 auditoría de implementación, exclusiva, parent R518
  -> R520 corrección candidata JSON, exclusiva, parent R519
  -> R521 auditoría de artefacto, exclusiva, parent R520
  -> documentación y wiki
```

Paths futuros exactos:

```text
R517: Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/517_wave60_r515_source_hash_authority_correction_plan_audit.md
R518: experiments/geometria_proporcional/adjudicate_wave60_v4_result.py
      tests/test_wave60_v4_result_adjudication.py
R519: Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/519_wave60_r509_replay_normalization_implementation_audit.md
R520: Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_V4_REPLAY_NORMALIZATION_CORRECTION.json
R521: Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/521_wave60_v4_replay_normalization_correction_audit.md
```

R517 usa scope `R515_SOURCE_HASH_AUTHORITY_CORRECTION_PLAN` y target exacto
`{plan_commit,plan_sha256}`. R519 usa scope
`R509_REPLAY_NORMALIZATION_RESOLUTION_IMPLEMENTATION` y target exacto
`{implementation_commit,files}`, con los dos paths R518. R521 usa scope
`WAVE60_V4_REPLAY_NORMALIZATION_CORRECTION` y target exacto
`{artifact_commit,artifact_path,artifact_sha256}`.

Todos los informes conservan el keyset de autoridad R514, un único bloque
JSON, `PASS 0/0/0`, `files_modified=false` y
`gpu_used_or_queried=false`. Parent, pathset, blob y hash se validan
fail-closed.

## 4. Cambios exactos en R520

El `authority_chain` de R520 añade a las nueve claves de R514 estas dos claves
exactas:

```text
r516_source_hash_correction_plan
r517_source_hash_correction_plan_audit
```

El plan R516 se liga mediante `{commit,path,sha256}` y R517 mediante
`{commit,path,sha256,authority_json}`. Las claves de implementación y auditoría
se renombran conforme a su número real: `r518_implementation` y
`r519_implementation_audit`. El resto de autoridades históricas R509–R515 se
preserva, incluida R515 como PASS histórico luego supersedido para autorizar
implementación.

El top-level y los demás subobjetos conservan los keysets de R514. Sólo cambian
los valores de fase:

- `artifact_status=CANDIDATE_PENDING_R521_AUDIT`;
- `activation_condition.required_audit_id=R521`;
- `activation_condition.required_audit_path` es el path R521 anterior;
- `authority_effect=ACTIVATES_CONDITIONAL_CORRECTED_VIEW` permanece;
- la vista continúa condicional y la documentación no la adopta antes de R521.

R520 no afirma que R521 ya ocurrió. R521 repite la derivación, autentica el
commit/path/hash de R520 y decide si activa la vista.

## 5. Estado del borrador y pruebas

El borrador de checker/tests que descubrió el error no se integra bajo R515.
Se conserva fuera de la cadena hasta R517. R518 debe introducir exactamente
los dos paths declarados y contener el hash corregido, además de todas las
guardas de R514.

La suite añade un regression test literal para el mapa de ocho sources y
mantiene los ataques de R514: autoridades Git/JSON, hardlinks/inodos,
self-manifest metadata, roster, hashes target, enlaces locales, segundo
mismatch, roles, keysets, publicación externa exclusiva y preservación de la
ciencia.

## 6. Autoridad científica y cómputo

La corrección de un carácter documental no cambia el intento, la config, los
sources, las métricas ni los patrones. Todo sigue siendo CPU-only; no se usa ni
consulta GPU. La arquitectura no se promueve y no se declara `GO/NO-GO`.
