# Wave 59 signed preparation implementation audit R434

**Implementation commit:** `9f9c64a4e5e80849f20bfe8e5cf02628a8d4823b`  
**Preparer SHA-256:** `d3e7f5f27a34aebc5c47baeff2a5da4da738f9508e4155748804d0cac856bdd6`  
**Runner SHA-256:** `e1f601a5653cbbe0d0598ab6de61a6437b0c7f8119d5f021f2ec9574d6580e4c`  
**Prospective test SHA-256:** `dd055598df51c51c4873fb43f04da6a28302b0145f41f33d5108b0e81451cccc`  
**Recovery test SHA-256:** `d102f825de8af3a3238217d591267ac5fca497c5b4db6f918b02e8b9ef270fd1`  
**Result:** `REVISE`

## Alcance verificado

El parent directo es exactamente `463af3e30302aded6f97c9b6ec63e71755eedaa8`. El commit modifica exclusivamente los cuatro paths autorizados y el worktree estaba limpio. Leí completos el plan de recuperación, R432, R433, el diff y los cuatro archivos modificados.

La regresión focal CPU-only terminó con `71 passed in 45.15s`, usando CUDA invisible, cuatro threads y un `basetemp` propio posteriormente eliminado. No se ejecutó recovery ni hubo acceso semántico a escrow, secrets o truth.

## Findings graduados

### P1 — Un fallo de la firma puede impedir su propio archivado y dejar viva la raíz canónica

La cuarta condición de R433 no queda satisfecha.

El preparador firma al final y, si esa operación falla, entra correctamente al `except` exterior ([prepare_wave56_fresh.py:3922](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/prepare_wave56_fresh.py:3922), [prepare_wave56_fresh.py:3928](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/prepare_wave56_fresh.py:3928)). Sin embargo, ese handler llama a `archive_failed_attempt()` con la misma clave privada cuya lectura, correspondencia o disponibilidad pudo causar el fallo.

El archivador vuelve a exigir esa clave y firma `failure_attestation.json` antes de renombrar la raíz ([run_wave59_hgb_guard_bracket.py:2303](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave59_hgb_guard_bracket.py:2303), [run_wave59_hgb_guard_bracket.py:2316](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave59_hgb_guard_bracket.py:2316)). Si la clave desaparece, queda ilegible o deja de corresponder a la pública entre el preflight y la firma final, la segunda firma falla por la misma causa y `os.replace(run_dir, archived)` nunca ocurre. No existe fallback que retire físicamente la raíz canónica.

Tampoco se agregó una prueba que inyecte un fallo de firma/publicación final y compruebe que la raíz desaparezca y el intento archivado permanezca inventariado. Las pruebas nuevas sólo cubren validación de paquetes ya firmados.

Corrección requerida: desacoplar el retiro físico fail-closed de la disponibilidad de la firma de fallo, y probar al menos fallos de firma y publicación atómica verificando ausencia de la raíz canónica.

### P1 — La firma liga el manifest, pero el runner no liga los archivos físicos declarados por ese manifest

`preparation_attestation.json` firma el registro del propio `benchmark/manifest.json`, pero no los archivos del benchmark ([prepare_wave56_fresh.py:3336](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/prepare_wave56_fresh.py:3336)). Eso sería suficiente si el runner verificara que el mapa `manifest["files"]` continúa coincidiendo con los bytes físicos.

`validate_signed_preparation_package()` sólo exige las claves superiores, schema/generator y el SHA del manifest ([run_wave59_hgb_guard_bracket.py:587](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave59_hgb_guard_bracket.py:587), [run_wave59_hgb_guard_bracket.py:600](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave59_hgb_guard_bracket.py:600)). No llama a `validate_manifest()`, no compara cada registro `bytes + sha256` con su archivo y no exige igualdad entre el inventario físico de `benchmark/` y el mapa firmado.

Por ello, alterar después de la firma un archivo visible, commitment o atestación semántica deja intactos el manifest firmado y toda la cadena actualmente comprobada por esta función. También pueden aparecer archivos adicionales bajo `benchmark/` sin rechazo en la autorización de preparación. El cierre posterior mediante `artifact_manifest.json` llega después de las fases analíticas y no sustituye una autenticación fail-closed previa.

La corrección puede seguir siendo content-blind: validar el inventario del benchmark por `lstat`, tamaño y SHA-256 contra el manifest, sin interpretar escrow, commitments ni truth.

### P2 — La comprobación anti-symlink de la raíz queda anulada por resolución prematura

`validate_signed_preparation_package()` intenta inspeccionar con `lstat` la ruta no resuelta ([run_wave59_hgb_guard_bracket.py:468](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave59_hgb_guard_bracket.py:468)). Pero los callers productivos ya resolvieron esa ruta:

- `_execute_once()` en [run_wave59_hgb_guard_bracket.py:2688](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave59_hgb_guard_bracket.py:2688);
- `execute()` en [run_wave59_hgb_guard_bracket.py:3056](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave59_hgb_guard_bracket.py:3056);
- `main()` en [run_wave59_hgb_guard_bracket.py:3098](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave59_hgb_guard_bracket.py:3098);
- y nuevamente `validate_execution_bindings()` en [run_wave59_hgb_guard_bracket.py:380](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave59_hgb_guard_bracket.py:380).

Un alias symlink hacia la raíz canónica llega al validator como el directorio real y el `lstat` ya no puede detectar el alias. No permite fabricar una raíz alternativa, pero contradice la identidad física predeclarada y vuelve inefectivo el guard específico.

## Condiciones R433

1. Firma posterior a mutaciones de receipt/presupuesto: satisfecha.
2. Introducción histórica del test y blob nuevo: satisfecha mediante `introduced_commit`, ancestralidad, blob del implementation commit y path incluido en el commit.
3. Verificación criptográfica individual más invariantes primary/replay: satisfecha; ambas firmas se verifican antes de retirar únicamente rol, modo y los tres records operativos declarados.
4. Fallos de firma/publicación archivados: no satisfecha por el P1 anterior.

La autoridad Git/source conserva parents directos, paths exclusivos, hashes de blobs, introducciones históricas, auditorías parseables, HEAD final y worktree global limpio. No observé otro bloqueo en esa cadena.

**Final decision:** `REVISE`
