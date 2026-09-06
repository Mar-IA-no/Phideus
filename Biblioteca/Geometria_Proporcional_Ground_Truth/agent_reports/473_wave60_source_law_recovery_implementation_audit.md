# R473 — Auditoría de implementación Wave 60

## Dictamen técnico: REVISE

Commit auditado: `53c383284e222cd60890a87857d66a8245f1a148`.

Conteo: **1 HIGH, 2 MEDIUM, 0 LOW**.

### HIGH — R473 no puede autenticar la ejecución canónica

El auditor de implementación exige literalmente `scope="IMPLEMENTATION"` en [run_wave60_frozen_policy_transport.py](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py:362), especialmente en la comparación de la línea 420. Sin embargo, la autoridad R473 requerida para este commit usa `scope="SOURCE_LAW_RECOVERY_IMPLEMENTATION"`.

La ruta de recuperación llama ese validador sin distinguir el nuevo scope en la línea 1949. Por ello, un request v2 correctamente ligado a R473 será rechazado después de reservar el staging y sellará la única root v2 como `SOURCE_LAW_INVALID`, `recovery_allowed=false`.

La incompatibilidad reaparece en [prepare_wave56_fresh.py](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/prepare_wave56_fresh.py:1120), que también fija `scope="IMPLEMENTATION"` para la futura config. Ese archivo estaba autorizado por el plan, pero no fue modificado.

Acción requerida:

- Parametrizar el scope esperado.
- Conservar `IMPLEMENTATION` exclusivamente para autenticar el audit histórico de v1.
- Exigir `SOURCE_LAW_RECOVERY_IMPLEMENTATION` para `53c3832`/su sucesor y propagarlo al preparador.
- Añadir una prueba con un reporte R473-shaped exacto tanto en publisher como en validación de config.

### MEDIUM — El output v2 incorrecto escribe fuera del namespace canónico

El publisher crea el staging antes de determinar si la invocación es recovery: líneas 1917–1936 del runner. La identidad recovery se deriva de `output == SOURCE_AUTHORITY_DEFAULT`, y el desacuerdo con el schema se procesa luego como fallo semántico, líneas 1942–1944 y 2068–2077.

Reproducción CPU realizada: un request recovery dirigido a otro path repo-relative seguro produjo:

- `published_wrong_output=true`
- `terminal=SOURCE_LAW_INVALID`
- `recovery_allowed=true`

Esto contradice el gate sin mutaciones del plan, [WAVE_60_SOURCE_LAW_RECOVERY_PLAN.md](/mnt/m2-1TB/Phideus/Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_SOURCE_LAW_RECOVERY_PLAN.md:181), y la regla de que cualquier fallo v2 sea no recuperable, línea 257.

Acción requerida:

- Separar explícitamente el entrypoint recovery del publisher legacy.
- Validar el output v2 exacto antes de crear staging.
- Ante recovery con output distinto, levantar excepción sin escribir target ni staging.
- Añadir regresión que compruebe además que nunca se publica `recovery_allowed=true`.

### MEDIUM — Falta cobertura positiva integrada del publisher recovery

Los tests nuevos verifican por separado:

- request y v1 auténticos: [test_wave60_frozen_policy_transport.py](/mnt/m2-1TB/Phideus/tests/test_wave60_frozen_policy_transport.py:1523);
- tampering y forma física: línea 1544;
- sellado de fallo v2: línea 1637;
- rechazos de namespace: línea 1681;
- worker recovery aislado: línea 3046.

No existe una prueba positiva que atraviese `publish_source_law_authority` con una auditoría recovery válida, ni pruebas CLI/API relativas y absolutas. Tampoco se comprueba integradamente el journal acumulativo, attestation, manifest y rename final. Esa ausencia permitió que los dos defectos anteriores coexistieran con una suite focal verde.

Acción requerida: construir una cadena Git temporal o inyectar una autoridad autenticable y probar publisher API + CLI de extremo a extremo, incluyendo lineage, presupuesto restante, v1 inmutable, ausencia de attempt y terminal v2 completo.

## Aspectos conformes

- Lineage: `53c3832` es hijo directo de `346e9fc`; el commit modifica exclusivamente los cuatro paths declarados, todos permitidos por el plan.
- Plan `a8e8932` y R472 `346e9fc` están autenticados por path, blob, hash, exclusividad y parent directo.
- v1 conserva exactamente cinco archivos, firma válida, root/journal `0700`, archivos `0444`, `root:root`, `nlink=1` y los cinco hashes canónicos.
- La validación v1 cubre closed world, metadata, hashes, request, Git, journal, failure, inventory y attestation.
- El worker recovery conserva UID/GID 65534, capabilities vacías, `NoNewPrivs`, CUDA invisible y probes denegados sobre v1/attempt.
- El presupuesto descuenta `0.00401783362030983 s`, limita el worker al remanente y exige acumulado `<900 s`.
- La config tipa exclusivamente la autoridad v2.
- Al cierre seguían ausentes v2, attempt y staging canónicos; los hashes Wave 59 declarados permanecían válidos.
- `git diff --check` pasó y `git status --short` quedó vacío.

## Pruebas ejecutadas

- Suite Wave 60: **81 passed** en 83,99 s; pico RSS 872.820 KiB.
- Regresión Wave 56–60: **414 passed, 1 skipped**, con tres fallos inducidos por ubicar `tmp_path` dentro del repo; reejecutados bajo sus precondiciones normales: **3 passed**. Resultado efectivo: **417 casos ejecutables pasaron, 1 skip**.
- Los primeros intentos en `/tmp` encontraron `EXDEV` y luego `ENOSPC`; no se atribuyeron al commit.
- Todas las corridas usaron CPU, `CUDA_VISIBLE_DEVICES=''`, cuatro threads en la regresión, y registraron cero swaps.
- No se consultó GPU, Colab ni Mendieta.
- Los directorios temporales propios fueron eliminados. No se modificaron archivos del repositorio.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R473",
  "scope": "SOURCE_LAW_RECOVERY_IMPLEMENTATION",
  "target": {
    "implementation_commit": "53c383284e222cd60890a87857d66a8245f1a148"
  },
  "technical_verdict": "REVISE",
  "findings": {
    "high": 1,
    "medium": 2,
    "low": 0
  },
  "files_modified": false,
  "gpu_used_or_queried": false
}
```
