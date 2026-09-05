# Wave 59 closed inventory implementation audit R439

**Implementation commit:** `51ceb02a93f23a6dd97fc75ce677a425a2dbf5ab`  
**Preparer SHA-256:** `b3999b95133aef4184cda3614b15ac30b5ff67cdaac224d752754a991c7c24a7`  
**Runner SHA-256:** `9dcd0020606b01cb90d10e1db7db6b2f418f2c6f3428e3c18b456af4541bfad9`  
**Prospective test SHA-256:** `4e5c8f91cda1692648eeff2d445a63244a98d5706be72fa27f6203abc5ffba39`  
**Recovery test SHA-256:** `5b67a2326962517cc2bff71922b672d952765552860d548c4d5f5177157a80d0`  
**Result:** `REVISE`

## Findings

### P0/P1 — ninguno

El commit es hijo directo de `2e7f16f`, modifica exactamente los cuatro paths autorizados y pasa `git diff --check`. El plan vigente coincide con SHA-256 `73e6e2c0403154bef6cd989d341d6778026644886af1653c6ddd0c3f1b3d52d9`; R436 y R438 también coinciden con sus hashes documentados.

El runner ahora cierra correctamente archivos y directorios físicos contra `manifest.files`: los probes focales rechazaron archivo extra, directorio vacío, FIFO, socket Unix, miembro symlink, archivo ausente y mutación de bytes. El preparador rechaza el FIFO antes de invocar la firma. Los cierres anteriores de archivo fail-closed y guard de ruta cruda permanecen sin regresión.

### P2 — El pre-sign del preparador sigue un symlink usado como raíz de `benchmark/`

`validate_wave59_closed_benchmark_inventory()` ejecuta `benchmark_root.resolve(strict=True)` antes de inspeccionar la raíz mediante `lstat`. Por eso rechaza symlinks internos, pero acepta que el propio `root/benchmark` sea un symlink a otro árbol físico.

Un probe focal construyó una preparación físicamente ordinaria salvo por `run/benchmark -> external-benchmark`; el helper la aceptó y `publish_wave59_preparation_attestation()` llegó a invocar el firmante y publicar `preparation_attestation.json`.

Esto no abre autoridad analítica: `validate_signed_preparation_package()` conserva `_require_root_owned_mode(resolved / "benchmark", ..., directory=True)` y rechaza ese paquete. Sin embargo, contradice el cierre físico pre-sign incorporado por este commit y permite emitir una atestación sobre un benchmark cuya raíz no es el directorio físico declarado.

Corrección acotada: aplicar `lstat()` a `benchmark_root` antes de `resolve()`, rechazando symlink o tipo distinto de directorio, y agregar un probe directo del publicador que confirme que la firma no se invoca para una raíz `benchmark/` symlink.

## Verificación focal

`21 passed in 2.87s`, cubriendo aceptación auténtica, negativas del benchmark, alias del caller y ambos fallbacks de archivo sin firma. Los probes auxiliares fueron CPU-only, no accedieron material semántico y limpiaron sus directorios temporales. El worktree terminó limpio.

**Final decision:** `REVISE`
