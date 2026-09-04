# R403 — Reauditoría de implementación de recovery pre-oracle de Ola 57

**Implementation commit:** `5b335cb47c00cb9770bfcacd815459c5109a1213`
**Preparer SHA-256:** `a4de65cdccc33b7709ce79952848116ff6d0a13705612231adaa29710b821253`
**Test SHA-256:** `7ee02f622c61633509a58973966cdafd586fe497bd4c287793ac0775c2e84594`
**Result:** `PASS`

## Dictamen

El candidato cierra los dos findings de R402. Su commit y padre directo son exactos, el preparador conserva el blob ya auditado y el único cambio respecto de la implementación anterior está en la cobertura de `tests/test_wave57_prospective.py`. El delta desde el padre de la cadena activa sigue limitado al preparador compartido y al test Wave 57; `git diff --check` termina limpio.

El validator real de autoridad Wave 57 se ejecuta ahora sobre una cadena Git sintética completa, primero en un caso positivo y después en dieciséis fallos independientes. El probe content-blind contra el origen real también intercepta las rutas de lectura e interpretación que faltaban. No encontré falsos positivos en los negativos ni una regresión de la ruta histórica Wave 56.

R402 fue leído verbatim como antecedente de esta reauditoría. No fue modificado ni tratado como commit o eslabón del DAG activo, cuyo candidato desciende directamente de R401.

## Findings priorizados

### Alta

Ninguno.

### Media

Ninguno.

### Baja

Ninguno que requiera revisar la implementación.

## Cierre del finding alto de R402

El fixture crea un repositorio temporal con origen, plan, auditoría de plan, implementación, auditoría de implementación, amendment y auditoría final, conservando blobs old/new y contratos diferenciados (`tests/test_wave57_prospective.py:137-329`). Las constantes de paths se sustituyen de forma acotada al fixture y la llamada entra en `_validate_wave57_recovery_amendment` real; sólo se mockean las dos etapas de inspección física/semántica posteriores y `read_escrow`, registrando el orden `content-blind → semantic → escrow` (`tests/test_wave57_prospective.py:332-381`).

El positivo cruza la cadena completa y comprueba schema, commit de implementación y orden de accesos (`tests/test_wave57_prospective.py:417-424`). La matriz negativa cubre path alternativo, schema, status, assertion, población, hash new, hash old de origen, commits mezclados de plan e implementación, ruptura del parent directo, resultados `REVISE` en cada una de las tres auditorías, auditoría de implementación ausente, HEAD posterior y worktree sucio (`tests/test_wave57_prospective.py:427-491`).

Para descartar que el `pytest.raises` amplio ocultara defectos del fixture, reejecuté los dieciséis casos capturando su excepción concreta. Cada uno alcanzó la guardia prevista: path canónico; schema; aprobación; assertions; población; blob nuevo; blob viejo del origen; paths exclusivos de plan e implementación; parent directo; parser canónico de cada auditoría; artefacto de auditoría ausente; HEAD exacto; o limpieza global. Ningún caso falló por una causa colateral.

Esto ejercita la implementación de autoridad en `experiments/geometria_proporcional/prepare_wave56_fresh.py:1649-1966`, incluyendo el cierre Git previo al preflight content-blind y la transición posterior a validación semántica. El primer finding de R402 queda cerrado.

## Cierre del finding medio de R402

El probe real es path-aware para el escrow y todo `benchmark/sealed/`. Intercepta `Path.read_text`, `Path.read_bytes`, `builtins.open` e `io.open`; impide `json.loads` sobre bytes; mantiene bloqueados los helpers semánticos; y usa profiling de llamadas C para impedir `bytes.fromhex` (`tests/test_wave57_prospective.py:508-617`). Un probe independiente confirmó que el perfilador observa efectivamente `bytes.fromhex` con el nombre esperado.

Los descriptores sensibles sólo pueden originarse mediante `os.open` llamado desde `_secure_file_record` con `O_NOFOLLOW`. Sus `fstat` y lecturas se ligan al mismo helper, se exige que `fstat` anteceda al streaming y el cierre sólo completa descriptores ya registrados con lecturas realizadas (`tests/test_wave57_prospective.py:527-572`). Al final, el test exige cobertura de escrow, tres secret files y cuatro truth JSONL, ausencia de descriptores abiertos y correspondencia exacta entre aperturas y cierres (`tests/test_wave57_prospective.py:622-654`).

El probe pasó contra el origen canónico sin alcanzar parsing, conversión o helpers semánticos. Conserva así la única lectura pre-authority permitida por el plan: `lstat`, `O_NOFOLLOW`, `fstat` y bloques dirigidos al SHA-256 implementados en `prepare_wave56_fresh.py:295-338`. El segundo finding de R402 queda cerrado.

## Regresiones y contrato preservado

- La separación efectiva sigue siendo autoridad Git y content-blind antes de semántica y escrow (`prepare_wave56_fresh.py:1649-1952`).
- El delta de contrato continúa admitiendo sólo `git_commit` y exactamente los hashes del preparador y test Wave 57 (`prepare_wave56_fresh.py:957-999`).
- Permanecen congelados por split `4992` rows, `1152` tokens totales y `768` elegibles, además de OOD, no canónicos e intersección (`prepare_wave56_fresh.py:1711-1721`).
- Recovery sigue extrayendo claves del escrow reutilizado; `secrets.token_bytes` permanece fuera de la rama con `recovery_context` (`prepare_wave56_fresh.py:2375-2414`).
- Replay conserva comparación exacta de escrow, freeze, manifest, visibles, logits, poblaciones y provenance (`prepare_wave56_fresh.py:2310-2372`).
- El dispatcher mantiene validators separados por schema y la ruta Wave 56 continúa usando el parser estricto por defecto (`prepare_wave56_fresh.py:834-909,1969-1995`).

## Ejecución de auditoría

- Subsuite específica de los dos cierres: 22 pruebas pasadas, 29 deseleccionadas, en 6,03 s.
- Regresión focal completa de `test_wave57_prospective.py`, `test_wave56_preoracle_recovery.py` y `test_wave56_prospective.py`: 164 pruebas pasadas en 211,27 s.
- Dieciséis negativos del validator reejecutados como probes diagnósticos: dieciséis rechazos por la causa pretendida y cero aceptaciones inesperadas.
- `git diff --check` sin errores; blobs del preparador y test coincidentes con el encabezado.

Las pruebas usaron `CUDA_VISIBLE_DEVICES=''` y un thread para BLAS, OpenMP y MKL. No se ejecutó recovery físico, inferencia oficial, oracle ni GPU. No se repitió la suite global ya pasada sobre el mismo blob del preparador: la regresión focal incluye íntegramente el archivo modificado y las suites históricas relevantes.

## Machine-verifiable decision

**Final decision:** `PASS`
