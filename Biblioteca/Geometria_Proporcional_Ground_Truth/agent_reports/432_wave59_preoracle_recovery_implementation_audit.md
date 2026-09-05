# Wave 59 pre-oracle recovery implementation audit R432

**Implementation commit:** `d02422f132ebfc7e07fa0e508be80a46cd76ff93`  
**Preparer SHA-256:** `ef210e3fd02965dc3e0d270d19623ca0fac35cd56a57d61c51d15e36230b0881`  
**Runner SHA-256:** `58fcd7c3d23facbf68a26a073f0fd7dfe51b6364b2fe539e4a9047ea6512ae73`  
**Prospective test SHA-256:** `d728e535cd867c6ebc86c7fcad9d0b6470166e7445b28e6bc8ea0acea701abdb`  
**Recovery test SHA-256:** `40f85b07d6c4e36539b4fb745a6265f8c84c424218c22e0a31ee02265d867c04`  
**Result:** `REVISE`

## Finding

### P1 — El runner acepta como autoridad un paquete recuperado autodeclarado

El plan exige que el runner autentique el paquete recuperado y quede ligado simultáneamente a config, amendment, provenance, preparation freeze y HEAD ([plan:242](/mnt/m2-1TB/Phideus/Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_59_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:242), [plan:299](/mnt/m2-1TB/Phideus/Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_59_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:299)). La implementación no autentica que `recovery_root` sea un output válido del preparador:

- [`validate_execution_bindings()`](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave59_hgb_guard_bracket.py:379) acepta cualquier directorio que contenga una copia del amendment público y un `preparation_freeze.json` cuyos campos autodeclarados reproduzcan HEAD, config, hashes físicos y `amendment_sha256`.
- No valida schema, phase, claves exactas, receipt/journal, manifest, owner ni modos del freeze o del directorio.
- [`_ensure_preparation_authority()`](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave59_hgb_guard_bracket.py:2399) sólo compara hashes de bundles y cuatro campos copiables del freeze. Tampoco autentica su procedencia.
- La ejecución considera canónico cualquier `prepared == output` que posea un subdirectorio `prepared/` ([runner:2512](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave59_hgb_guard_bracket.py:2512)).

Reproduje dos casos:

1. `_ensure_preparation_authority()` aceptó cinco archivos arbitrarios y un freeze con `schema_version` y `phase` iguales a `forged-and-unchecked`, siempre que sus hashes autodeclarados coincidieran.
2. `validate_execution_bindings()` aceptó un `recovery_root` modo `0777` con freeze autodeclarado, manteniendo activos los checks reales de clean HEAD. Sólo se sustituyó el retorno de la autoridad pública futura, todavía no publicada en R432, para simular su éxito posterior.

Una vez publicada la cadena pública, cualquiera puede copiar el amendment y fabricar esos campos. Por tanto, la rama relajada distingue “directorio con declaraciones consistentes” de “sin recovery”, pero no “paquete producido y sellado por el preparador” de “paquete fabricado”.

Además, falta la prueba positiva exigida: el único llamado directo al runner en el test nuevo comprueba el rechazo sin `recovery_root` ([test:656](/mnt/m2-1TB/Phideus/tests/test_wave59_preoracle_recovery.py:656)). No hay prueba directa de la rama recuperada ni de rechazo de freeze, provenance, receipt, owner o modos falsificados.

## Corrección requerida

Antes de publicar R432 como PASS:

- autenticar el paquete preparado mediante una autoridad no autofirmada: como mínimo identidad física root-owned, modos esperados, estructura canónica, schema/phase y forma exacta;
- ligar freeze, preparation receipt, journal de preparación, amendment y hashes de bundles mediante una cadena verificable emitida por el preparador;
- rechazar directorios alternativos, world-writable, freezes incompletos o con campos adicionales y bundles ligados únicamente por hashes autodeclarados;
- añadir una prueba positiva de paquete recuperado auténtico y pruebas negativas de paquete forjado, amendment copiado con provenance falso, freeze inválido, owner/modos incorrectos y receipt/journal desligados.

## Verificaciones realizadas

- Commit, parent directo, conjunto de cuatro paths y los cuatro SHA-256 coinciden.
- `git diff --check` limpio y worktree sin cambios.
- Lectura completa del preparer, runner, ambos tests, plan vigente y R431.
- La cadena pública, los tres deltas, content-blind, inventario/atestación de fallo, segundo inventario TOCTOU y separación de etapa semántica no mostraron otro bloqueo.
- Regresión focal independiente con CUDA vacío: `60 passed in 43.28s`.
- No se abrió semánticamente el origen real ni se ejecutó recovery.
- Los basetemps propios fueron eliminados después de verificar su cierre.

## Machine-verifiable decision

**Final decision:** `REVISE`
