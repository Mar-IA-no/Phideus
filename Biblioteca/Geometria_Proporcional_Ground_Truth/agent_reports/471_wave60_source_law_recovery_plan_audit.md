## Auditoría técnica independiente R471

**Veredicto: REVISE — 0 HIGH, 2 MEDIUM, 0 LOW**

Objetivo auditado:

- Commit: `0d85607b901686f77f5e3cc8a83708882fb618bc`
- Parent: `ab60d325075996fbfde14d685c7fdf5c1d0909af`
- SHA-256 del plan: `0ee1cc62953bedbe043ea62b96cd8f377e4a34fe7cb371f62cc78be0038cbb50`
- Commit exclusivo: un único archivo agregado.
- Worktree inicial y final: limpio.

### Findings

#### MEDIUM 1 — La dirección normativa de tres relaciones Git está invertida

El plan exige que la auditoría sea “parent directo” del plan en `WAVE_60_SOURCE_LAW_RECOVERY_PLAN.md:135-140`, y repite la misma dirección invertida para source audit e implementación/config en `WAVE_60_SOURCE_LAW_RECOVERY_PLAN.md:258-267`.

Eso contradice:

- la secuencia correcta declarada en `WAVE_60_SOURCE_LAW_RECOVERY_PLAN.md:304-316`;
- la formulación explícita “child directo” de `WAVE_60_SOURCE_LAW_RECOVERY_PLAN.md:311-313`;
- el validador vigente, que comprueba que el commit de auditoría tenga como parent al commit auditado en `prepare_wave56_fresh.py:407-428`;
- la cadena histórica efectiva `implementation → implementation audit`, confirmada por `d694d803... → 93c3b3f7...`.

Seguir literalmente las cláusulas invertidas produciría una genealogía imposible o haría que el validador rechazara la auditoría legítima.

**Corrección requerida:** expresar todas las aristas sin ambigüedad:

- parent de la auditoría del plan = commit del plan;
- parent de la implementación = auditoría del plan;
- parent de la auditoría de implementación = implementación;
- parent de la auditoría source law = auditoría de implementación;
- parent de la config = auditoría source law;
- parent de la auditoría de config = config.

#### MEDIUM 2 — No queda definida la transición que publica `SOURCE_LAW_INVALID` cuando falla el preflight v1

El plan ordena que cualquier mismatch del terminal v1 produzca un nuevo terminal v2 inválido en `WAVE_60_SOURCE_LAW_RECOVERY_PLAN.md:83-102`. Sin embargo, también establece que el staging v2 sólo se crea después de validar íntegramente §§3–5 en `WAVE_60_SOURCE_LAW_RECOVERY_PLAN.md:175-187`.

Falta distinguir cómo se publica atómicamente el terminal inválido si precisamente falla §3 o §4. La implementación actual crea el staging antes de validar request/output y luego convierte cualquier excepción en terminal firmado (`run_wave60_frozen_policy_transport.py:1414-1425`, `run_wave60_frozen_policy_transport.py:1504-1512`). Al reordenar la canonicalización como exige el nuevo plan, ese comportamiento deja de estar determinado.

La ambigüedad también afecta la frontera entre:

- errores de path/namespace inseguros, que deben rechazar sin escribir;
- errores de autenticación del terminal v1 o del request recovery, que según el plan deben producir `SOURCE_LAW_INVALID`;
- fallos posteriores del worker, que también deben cerrar en ese terminal.

**Corrección requerida:** predeclarar la máquina de publicación:

1. canonicalizar y validar el namespace v2 sin mutaciones;
2. si el output es inseguro, está fuera del repo o ya existe, rechazar dejando el target intacto y sin terminal nuevo;
3. una vez probado que el namespace v2 es canónico y libre, cualquier fallo de validación v1/request/autenticación debe usar un staging de fallo explícito y publicar atómicamente `SOURCE_LAW_INVALID`;
4. sólo un preflight completo habilita el staging científico y el worker;
5. agregar tests que comprueben target ausente para rechazos de namespace y terminal inválido completo para fallos de autoridad.

### Controles sin findings

- El plan deriva de un antecedente auténtico: el terminal v1 contiene exactamente cinco archivos, modos `0444`, roots `0700`, ownership `root:root`, `nlink=1`, inventario cerrado y firma Ed25519 válida.
- Sus cinco hashes coinciden con §1.
- El hash del error se reprodujo a partir del `ValueError` causado por `Path.relative_to()` entre output relativo y `REPO_ROOT` absoluto.
- No existen attempt, draw ni staging Wave 60 residuales.
- La recuperación está permitida por el contrato base pre-truth en `WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md:487-494`.
- No se detectaron cambios del estimando, ley fuente, modelos, thresholds, controles, bootstrap o autoridad `GO/NO-GO`.
- El binding transitivo request → v1 → plan/auditoría → freeze → attestation → manifest es prospectivamente viable.
- El descuento durable previo, el límite acumulado `<900 s`, el RSS por proceso y la prohibición de GPU están correctamente planteados.
- Los cinco paths autorizados alcanzan para implementar request tipado, canonicalización, validación v1, path v2 y consumo posterior.
- No se modificaron archivos, no se consultó ni utilizó GPU, Colab o Mendieta.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R471",
  "scope": "SOURCE_LAW_RECOVERY_PLAN",
  "target": {
    "plan_commit": "0d85607b901686f77f5e3cc8a83708882fb618bc",
    "plan_sha256": "0ee1cc62953bedbe043ea62b96cd8f377e4a34fe7cb371f62cc78be0038cbb50"
  },
  "technical_verdict": "REVISE",
  "findings": {
    "high": 0,
    "medium": 2,
    "low": 0
  },
  "files_modified": false,
  "gpu_used_or_queried": false
}
```
