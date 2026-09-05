# Wave 59 second correction implementation audit R436

**Implementation commit:** `c716e684689c216ee471007a66383c3bd588a1d7`  
**Preparer SHA-256:** `405f9b1d927bc7546e175e34a62989d85b607ee1d5e2f919b3f22d102c8c0e2b`  
**Runner SHA-256:** `b2f12ebf2e38a96b475c5572e0c5297f1433f2cb1e539c6f82792a7c4bda19cb`  
**Prospective test SHA-256:** `e742994eed67008abb95bd40ce45e003ba03cdebe492955c0a45995d26a920c7`  
**Recovery test SHA-256:** `4088ad0af890e914aa7270892ef5ad5fe54603ed580f6c1cdb5df4e9153349ce`  
**Result:** `REVISE`

## Alcance verificado

Leí completos los cuatro blobs modificados y su diff contra `32633086d9a3a8e2cf7de12544559812c538802d`, además del plan canónico, R434 y R435. Los enlaces declarados coinciden:

- plan SHA-256: `665f0e73ef2f75aa2f14adaa6c4b17b47e3a501b7f47e6650f399adb060ab9a1`;
- R434 SHA-256: `4f055b9b61f99abdf1297fef2bb44c8c4da7bf42c54d1cac502cc334d2eb3c66`;
- R435 SHA-256: `97b9a216d0763fa0ebedbd1fbdae8b8b41f7aa97babab01a8f8a1db166e3aabc`.

La cadena es lineal: `2b6aeee → 4104848 → 3263308 → c716e68`. El implementation commit tiene como parent directo a R435, modifica exclusivamente los cuatro paths autorizados y pasa `git diff --check`.

No ejecuté recovery, inferencia ni acceso semántico a escrow, secrets, commitments o truth. No usé GPU, web ni Mendieta.

## Findings graduados

### P0 — ninguno

No encontré una vía de redraw, apertura anticipada de truth ni promoción científica no autorizada.

### P1 — El inventario del benchmark todavía acepta objetos físicos no declarados

`validate_signed_preparation_package()` recorre `benchmark/`, rechaza symlinks y agrega al conjunto `physical` únicamente archivos regulares. Todo objeto que no sea symlink ni archivo regular queda ignorado (`run_wave59_hgb_guard_bracket.py:601-610`).

Reproduje adversarialmente un paquete sintético auténticamente firmado, agregué `benchmark/extra.pipe` mediante `mkfifo` y el validator devolvió:

```text
EXTRA_FIFO_RESULT=recovery
```

El FIFO no figura en `manifest.files`, pero tampoco entra en `physical`; por eso la igualdad de conjuntos y `validate_manifest()` pasan. La misma lógica no cierra el conjunto de directorios físicos adicionales. Esto contradice el inventario físico cerrado exigido por el plan y aprobado por R435: la autoridad acepta una raíz con material físico fuera del manifest.

Corrección requerida: durante el recorrido, rechazar explícitamente cualquier entrada que no sea directorio permitido o archivo regular declarado, y cerrar también el conjunto de directorios contra los padres derivados de `manifest.files`. Agregar al menos probes para FIFO/special node y directorio vacío no declarado.

### P2 — ninguno adicional

La mutación byte-exacta de archivos regulares sí queda cubierta por tamaño y SHA-256; symlinks y archivos regulares extra son rechazados.

## Cierres correctos de R434

### Archivo fail-closed sin clave

El archivador captura tanto fallo de firma como fallo de publicación, publica `failure_attestation_error.json` con `status=UNATTESTED_SIGNING_FAILURE` y `authoritative=false`, actualiza `failure_inventory.json`, mueve la raíz mediante `os.replace()` dentro del `finally` y retorna el archivo retirado. Los callers vuelven a propagar mediante `raise` el error original de preparación o ejecución.

El marker no confiere autoridad de recovery: la validación vigente continúa exigiendo `failure_attestation.json` firmado.

### Argumento crudo antes de resolver

`execute()`, `_execute_once()` y `main()` preservan el argumento mediante `absolute()` y ejecutan el guard basado en `lstat` antes de `resolve()`. El alias symlink aportado por el caller es rechazado también en el entrypoint público.

### Regresión

La regresión CPU-only afectada terminó:

```text
296 passed in 382.46s
```

Incluyó las suites Wave 56, 57, 58 y 59, junto con los nuevos casos de fallo de firma/publicación, alteración/extra/symlink del benchmark y alias symlink del caller. Se usó CUDA invisible, cuatro threads y un `basetemp` propio eliminado al finalizar.

**Final decision:** `REVISE`
