## R468 — REVISE

Resultado: **1 HIGH, 0 MEDIUM, 0 LOW**. No emito bloque normativo `wave60-audit-authority-v1`.

### HIGH — Una recuperación posterior a v2 sigue siendo imposible

La continuidad temporal del ledger quedó corregida para:

```text
primary v1 → primary v2 → replay v2
```

El e2e ahora usa el camino real de transacción, monitor de presupuesto y firma, sin reescribir manualmente los receipts. También preserva y verifica `cumulative_duration_seconds`.

Sin embargo, una recuperación posterior —por ejemplo v2 abortado → v3— falla antes de poder reutilizar ese ledger.

[`validate_wave60_final_config_authority()`](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/prepare_wave56_fresh.py:449) establece correctamente que:

- una config v1 debe descender directamente de la auditoría source-law;
- una config recuperada vN debe descender directamente de su `amendment_audit_commit`.

Pero [`_validate_wave60_recovery_amendment()`](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/prepare_wave56_fresh.py:3843) vuelve a exigir incondicionalmente que la config del intento previo descienda directamente de la auditoría source-law:

```python
require_direct_parent(
    repo_root,
    prior_config_commit,
    prior_config["source_law_authority"]["audit_commit"],
    "Wave 60 prior config",
)
```

Eso sólo es cierto para v1. Para una config v2 válida, su parent directo es la auditoría del amendment v2.

Reproducción sobre el repositorio temporal creado por el propio e2e focal:

```text
v2_config_commit = ac8bb3862650296e324e50aef5f142e893e881be
actual_parent     = 7b03b19510b02affa1cf90eb591ab73ba3610ab3
source_law_audit  = 4375b3362520ecb0f5fd0e5a9e24611d5a12fdf9
is_source_parent  = False

RuntimeError:
Wave 60 prior config commit must directly descend from its frozen predecessor
```

Por tanto, v2 es ejecutable, pero no puede convertirse en antecedente de v3. Esto contradice el namespace general `attempt_v{N}` y el énfasis contractual en recuperaciones posteriores.

Corrección requerida:

- al validar la config previa, elegir su predecessor según su propia clase:
  - v1/no recovery → `source_law_authority.audit_commit`;
  - vN/recovery → `attempt.recovery.amendment_audit_commit`;
- agregar un e2e `v2 abortado → v3 primary → v3 replay` que use transacción, monitor y firma reales y pruebe continuidad acumulativa desde v1, sin edición manual de receipts.

### Estado de R467 y no regresiones R466

- **Ledger v1→primary v2→replay v2:** corregido y cubierto por el e2e real.
- **Receipts:** el test ya no los reconstruye manualmente; usa `run_preparation_transaction()`, `wave59_coordinator_budget()` y `finalize_preparation_budget_authority()`.
- **Aliases en finalize exitoso:** corregido para `attempt`, `primary` y `replay`; los tres casos tienen rechazo focal.
- **Cadena Git v1/v2 de R466:** corregida; el finding nuevo aparece al intentar encadenar una recuperación posterior.
- **Paquete pair-failure con dos roots físicas:** sin regresión.
- **Traversal/symlink del intento previo:** sin regresión.
- **Finalize reanudado 899+2:** sin regresión; el test específico pasa.

### Identidad y alcance Git

```text
commit: 257275d9496f32a441aa4abce51c7f4f0d49e214
parent: 630ed485273c75433d1f5cbbec0937f5e6edb620
tree:   bc21e72eba491a679f646e57a7eb4636cace70dc
```

Diff exclusivo:

```text
3 files changed, 166 insertions(+), 79 deletions(-)
```

Paths modificados:

```text
experiments/geometria_proporcional/prepare_wave56_fresh.py
experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py
tests/test_wave60_frozen_policy_transport.py
```

Los otros dos paths contractuales permanecen byte-invariantes. Los cinco archivos físicos coinciden con sus blobs en `HEAD`.

### SHA-256 físicos

```text
40688554e7d97de9d065b34930303324fdbad6f2f55e4745536751ebac588da0  src/geometria_proporcional/wave60_frozen_policy_transport.py
23fc2029db88c25b8f5b74e3768e4345fb6e8fd8c181e70573e0ae0f54ccba53  experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py
c2ffafbda6234e2d7c92cc829b8f5634c9f5085a7b965f8e280243c31ef5591b  experiments/geometria_proporcional/_wave60_phase_worker.py
66be7d8ee187733c7e83bf718e01002b9dd2601afc70153ab4a4d99683f2204b  experiments/geometria_proporcional/prepare_wave56_fresh.py
075e1a37d55b49b3c46b8935688ddbad24c1a5283bcba958259fa3080322a497  tests/test_wave60_frozen_policy_transport.py
```

### Verificación ejecutada

Focal Wave 60:

```text
46 passed in 70.19s
maximum RSS: 873348 KiB
process swaps: 0
```

Regresión completa Wave 56–60:

```text
382 passed, 1 skipped in 458.04s
maximum RSS: 1065904 KiB
process swaps: 0
```

Además:

```text
py_compile: PASS
git diff --check: PASS
worktree inicial/final: limpio
```

Snapshot final del host:

```text
RAM usada: 10618413056 bytes
RAM disponible: 22747025408 bytes
swap usada global: 25070653440 / 34359734272 bytes
```

La swap global preexistente no fue atribuible a estas corridas; `/usr/bin/time` registró `0` swaps para ambos procesos.

No se usó ni consultó GPU/CUDA, Colab o Mendieta. Todas las pruebas llevaron `CUDA_VISIBLE_DEVICES=''` y cuatro threads. Se eliminó exclusivamente el temporal propio `.r468-audit-uaB9EW` —aproximadamente 3.0 GiB— y no se modificaron archivos ni commits.

**Decisión final: REVISE**
