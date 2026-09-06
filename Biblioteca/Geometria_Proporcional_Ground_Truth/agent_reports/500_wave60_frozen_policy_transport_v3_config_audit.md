# R500 — Auditoría de config Wave 60 v3

## Dictamen: PASS

La config congelada de Wave 60 v3 queda técnicamente aceptada para preparar el
nuevo par bajo la recovery aprobada en R499. La auditoría no ejecutó la
preparación, no materializó `attempt_v3`, no abrió truth y no emitió una decisión
`GO/NO-GO` científica.

## Identidad y parentage

El target auditado es el commit
`0cc349c10b91cffe0eb53ae6669fcffbfaf8c756`, hijo directo de R499
`2e602c290c0271c7f3f6de637b081d43cf2e8480`. El commit cambia un único path:

`experiments/geometria_proporcional/configs/wave60_frozen_policy_transport.json`

El blob Git y el archivo físico son byte-exactos, JSON canónico y tienen
SHA-256
`8d8edeed1d4c7943655a2824bd906001f711581d079ec79da9b9704739f94ae2`.
R499, a su vez, es un commit exclusivo de informe, hijo directo del commit
exclusivo de la amendment v3. El informe autentica la amendment con scope
`STATIC_PROTOCOL_IDENTITY_GUARD_RECOVERY_AMENDMENT`, target SHA-256
`c56c4e22c93ec73ff9f5ce2ee31316769bf15e21d57d7047a9ab172fbfcbdee6`
y findings `0/0/0`.

## Delta v2 → v3

La comparación estructural contra la config v2 del commit
`ef4a620ae79f5eb4502eea9cca1547da86216f93` encuentra sólo el delta autorizado:

- namespace y versión de attempt: v2 → v3;
- doce bindings de recovery hacia el attempt v2, su auditoría R495, la
  amendment v3 y R499;
- reserva de auditoría final R500;
- paths de output primario, replay y parent al namespace v3;
- self-binding de la config;
- hashes del runner, preparer y test cambiados por la implementación R498.

No cambia el dominio de ocho `required_execution_sources`. Tampoco cambian
schema, status, dispositivo CPU, threads, penalty, bootstrap, presupuesto,
políticas, features, seeds, batch size, splits físicos, benchmark fresco,
source bindings, autoridad de source law, implementación científica ni plan.
Por tanto el contrato científico es idéntico al de v2.

El self-binding recalculado es
`b62c6948a262befbd294da9f43892d4e02a93d6c40f70c9a5cc8a7c0807bb3f1`.
Los ocho bindings físicos coinciden con la config. La partición autorizada por
R498 es exactamente:

```text
runner    b35cd563f715bdff9b6e7489ac04712c728673563898d4a6aebf0144d4a50261
preparer  85069cd9e889e8dd5124ed0107c9e8fba69ab79a5edf52f3e25fee9e34ea4999
test      ca344305754669d15a86456e148a22d45af8e9d0f79ada0e98fd7a37034e2bef
```

Los sources científicos declarados invariantes coinciden entre config v2,
implementación R498, config v3 y filesystem:

```text
módulo científico 46e31fa1096ab4e5a0c5115fc4e16922e165a21dbe47016c42953bf345116c65
worker             c6c5c832633d9ed8a99079dd4fc4f5677073f7e1883a204fa6bb41f6d646bac7
```

## Estado físico heredado

La amendment liga 17 archivos preservados del draw, 53 entradas del inventario
de origen y 15 hashes terminales. El validador productivo volvió a autenticar
firmas, manifests, inventarios y terminales del attempt v2:

```text
pair terminal       PAIR_ABORTED_PRE_TRUTH
primary terminal    INVALID_NEW_DRAW_IDENTITY
replay terminal     PEER_ABORTED_PRE_TRUTH
any_truth_accessed  false
recovery_allowed    true
```

El ledger acumulado autenticado coincide exactamente con la amendment:

```text
preparation_seconds    215.36700256168842
source_binding_seconds   0.0
score_apply_seconds      0.0
phase_seconds            0.0
durable_seconds        215.36700256168842
```

## Validación ejecutable

`validate_pre_draw_config()` y `validate_prospective_config()` aceptaron el JSON
físico. `validate_recovery_amendment()` recorrió el lineage completo, R496–R499,
la partición R498, el mapa de sources, los artefactos preservados, terminal,
firmas, ledger y el delta de contrato. Después, una vista HEAD detached efímera
que contiene este informe como único hijo del commit de config permitió ejecutar
la autoridad final y `preparation_preflight()` con los paths canónicos y la
fuente de recovery v2, sin crear outputs.

Ocho tests focales cubrieron el flujo v2→v3 con ledger continuo, autoridad final
de los tres sources cambiados, delta contractual, provenance cerrada, igualdad
del par, separación de inodes, integridad del manifest/protocolo y rechazo de
colisiones no estáticas. Pasaron en 20,47 s; la corrida alcanzó 861.076 KiB de
RSS máximo y 0 swaps. La validación integral preliminar alcanzó 827.848 KiB de
RSS máximo y 0 swaps; la corrida integral de calificación sobre la primera vista
sellada alcanzó 956.964 KiB y 0 swaps.

Todas las corridas usaron `CUDA_VISIBLE_DEVICES=''` y
`PYTHONDONTWRITEBYTECODE=1`, con basetemps bajo `/mnt/m2-1TB`. No se usó ni
consultó GPU. Los recursos efímeros se retiraron por sus paths exactos. El
repositorio real conservó HEAD, main y refs sin movimiento.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R500",
  "scope": "CONFIG",
  "target": {
    "config_commit": "0cc349c10b91cffe0eb53ae6669fcffbfaf8c756",
    "config_sha256": "8d8edeed1d4c7943655a2824bd906001f711581d079ec79da9b9704739f94ae2"
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
