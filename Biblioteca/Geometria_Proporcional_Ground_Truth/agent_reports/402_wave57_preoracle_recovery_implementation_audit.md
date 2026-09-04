# R402 — Auditoría de implementación de recovery pre-oracle de Ola 57

**Implementation commit:** `4c5137cdeac8991108fd5594e8f3aaa517b1887e`
**Preparer SHA-256:** `a4de65cdccc33b7709ce79952848116ff6d0a13705612231adaa29710b821253`
**Test SHA-256:** `275ad4c46889e4902d2bf778925d445a4bfcfc758696c3fc1818aefa139e871d`
**Result:** `REVISE`

## Síntesis

El candidato auditado tiene la identidad y el delta correctos: su padre directo es `9e9a71493ea99c2dfe466bc7adc37db3a5df666b`, modifica solamente el preparador compartido y `tests/test_wave57_prospective.py`, y ambos blobs coinciden con los hashes declarados. La inspección del código no encontró una inversión actual de la frontera content-blind: la cadena Git, HEAD, limpieza, contrato público y delta se resuelven antes del primer acceso semántico al escrow o a truth sellada. El lector físico usa `lstat`, apertura binaria con `O_NOFOLLOW`, `fstat` y SHA-256 por bloques.

El candidato debe revisarse, sin embargo, porque su cobertura ejecutable no prueba el nuevo validator de autoridad de Ola 57. Además, el probe sobre el origen real no bloquea todas las rutas de interpretación enumeradas por el plan. Es una brecha de aceptación material en un control cuyo propósito es demostrar ausencia de acceso prematuro, aunque las pruebas existentes y la suite completa terminen en verde.

## Findings priorizados

### Alta — el validator completo de autoridad de Ola 57 no tiene ejecución directa en tests

La rama nueva que autentica path canónico, schema, estado, assertions, poblaciones, blobs, DAG lineal, HEAD, worktree, origen físico, replay y transición a validación semántica ocupa `experiments/geometria_proporcional/prepare_wave56_fresh.py:1649-1966`. Ninguna prueba llama realmente a `_validate_wave57_recovery_amendment`. El único ejercicio del dispatcher sustituye ambos validators por lambdas y comprueba sólo la selección tipada (`tests/test_wave57_prospective.py:88-118`). Los otros tests nuevos invocan componentes aislados —preflight content-blind, delta, revalidación mockeada, provenance y parser— en `tests/test_wave57_prospective.py:121-328`.

Por eso quedan sin demostración ejecutable específica de Ola 57 los rechazos obligatorios de amendment en path alternativo, schema/estado/assertions/población alterados, commits no lineales o mezclados, hashes old/new falsos, auditorías ausentes o contradictorias, HEAD posterior y worktree sucio. Esas obligaciones están enumeradas en `Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_57_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:197-208`. La cobertura histórica de Ola 56 no cierra esta brecha: el despacho lleva a funciones distintas en `experiments/geometria_proporcional/prepare_wave56_fresh.py:1969-1995` y la nueva función contiene su propia implementación de aproximadamente trescientas líneas.

Corrección requerida: agregar un fixture Git temporal que construya la secuencia cerrada plan → R401 → implementación → R402 → amendment → auditoría final, invoque la función real y aplique mutaciones independientes a cada autoridad y vínculo. Debe existir al menos un caso positivo y casos negativos para cada familia de rechazo exigida por el plan.

### Media — el probe real no vigila todas las rutas semánticas que el plan exige prohibir

El probe real verifica `O_NOFOLLOW`, bloquea `Path.read_text` sobre paths sensibles y sustituye cinco helpers conocidos (`tests/test_wave57_prospective.py:135-168`). No intercepta `Path.read_bytes`, `open`/lecturas binarias distintas del lector autorizado, `json.load`/`json.loads` aplicado a bytes obtenidos por otra vía ni `bytes.fromhex`. Una regresión que leyera un archivo sensible con `read_bytes` y luego lo pasara directamente a `json.loads`, por ejemplo, no activaría los guards actuales.

El plan exige explícitamente que el test haga fallar `read_text`, `read_jsonl`, un parser JSON, `bytes.fromhex` o cualquier consumidor semántico sobre escrow, secrets o truth sellada (`WAVE_57_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:209-214`). El código actual sí mantiene una whitelist pública estrecha (`prepare_wave56_fresh.py:1030-1042`) y el recorrido observado usa sólo el hasher opaco para archivos sensibles (`prepare_wave56_fresh.py:295-338,1054-1234`); este finding registra una defensa de test incompleta, no una lectura semántica observada.

Corrección requerida: ampliar el oracle del probe para que cualquier adquisición o transformación semántica de bytes sensibles falle, permitiendo exclusivamente la secuencia registrada `os.open(O_NOFOLLOW) → os.fstat → os.read → hashlib.sha256`. El test debe demostrar de forma positiva que escrow, tres secrets y cuatro truth JSONL cruzaron sólo esa ruta.

## Controles que sí quedaron verificados

- La identidad Git es exacta: commit `4c5137cdeac8991108fd5594e8f3aaa517b1887e`, padre directo `9e9a71493ea99c2dfe466bc7adc37db3a5df666b` y delta limitado a los dos paths autorizados. `git diff --check` no reportó errores.
- El preflight físico calcula todos los hashes por streaming binario y rechaza symlinks o archivos especiales (`prepare_wave56_fresh.py:295-380`). El probe contra el origen real confirmó apertura opaca de escrow, tres secrets y cuatro truth JSONL sin alcanzar los helpers semánticos instrumentados.
- El delta de contrato conserva todos los campos salvo `git_commit` y exactamente las dos fuentes autorizadas, con bindings old/new completos (`prepare_wave56_fresh.py:957-999`).
- La secuencia efectiva termina la autoridad de repositorio y el preflight content-blind antes de llamar a la etapa semántica (`prepare_wave56_fresh.py:1649-1943`). La extracción de claves ocurre sólo después de una nueva revalidación autorizada (`prepare_wave56_fresh.py:2375-2414`).
- El contrato de población fija por split `4992` filas, `1152` tokens totales y `768` elegibles, además de los conteos OOD/no canónicos (`prepare_wave56_fresh.py:1711-1721`). La ejecución recuperada valida por `eligible_unique_pair_tokens`, conservando ambos conteos en receipts (`prepare_wave56_fresh.py:2484-2538`).
- No hay redraw bajo `recovery_context`: las claves provienen exclusivamente del escrow reutilizado y `secrets.token_bytes` queda en la rama sin escrow (`prepare_wave56_fresh.py:2391-2414`).
- Replay conserva comparación exacta de manifest, protocolo, visibles, escrow, freeze, amendment, logits, preparation freeze, poblaciones y provenance (`prepare_wave56_fresh.py:2310-2372`).
- La ruta histórica de Ola 56 conserva el validator separado y el parser estricto por default. El probe manual confirmó: Ola 57 acepta una o dos LF terminales y rechaza tres; Ola 56 acepta una y rechaza dos o tres (`prepare_wave56_fresh.py:834-909`).

## Ejecución de auditoría

- Probe real content-blind más caso canónico R401: 2 pruebas pasadas en 2,32 s.
- Suites focalizadas `test_wave57_prospective.py`, `test_wave56_preoracle_recovery.py` y `test_wave56_prospective.py`: 144 pruebas pasadas en 218,08 s.
- Suite completa: 495 pruebas pasadas en 269,49 s.
- Todas las ejecuciones usaron `CUDA_VISIBLE_DEVICES=''` y límites de un thread para BLAS/OpenMP/MKL. No se ejecutó recovery físico del origen, inferencia oficial, oracle ni GPU.

Los resultados verdes muestran que no hay regresiones observadas en la cobertura existente; no compensan las obligaciones de prueba ausentes descritas arriba.

## Machine-verifiable decision

**Final decision:** `REVISE`
