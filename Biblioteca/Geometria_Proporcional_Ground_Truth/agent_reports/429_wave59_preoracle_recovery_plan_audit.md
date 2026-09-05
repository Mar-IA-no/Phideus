## Auditoría independiente focal — Ola 59

**Plan commit:** `ee57624128991bfc3624e35f3e363b32a5ed3179`  
**Plan SHA-256:** `cc0073b3696c27459036040f1120ae98dfca42b44a0248345f44c88c12da558d`  
**Result:** `REVISE`

### Findings

#### P1 — El schema de dispatch escrito no existe

El plan enruta la recuperación con:

`prospective_config.schema_version=wave59-hgb-guard-bracket-v1`

en [WAVE_59_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md](/mnt/m2-1TB/Phideus/Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_59_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:158).

El contrato congelado usa exactamente `wave59-fresh-hgb-guard-bracket-v1` en [wave59_fresh_hgb_guard_bracket.json](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/configs/wave59_fresh_hgb_guard_bracket.json:2), igual que `WAVE59_CONFIG_SCHEMA` en [prepare_wave56_fresh.py](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/prepare_wave56_fresh.py:110).

Implementado literalmente, el nuevo dispatcher no reconocerá Wave 59 y caerá en otra ruta o rechazará el amendment.

**Corrección requerida:** reemplazar el identificador de la línea 158 por `wave59-fresh-hgb-guard-bracket-v1`. El schema propio del amendment puede conservarse si es deliberado.

#### P1 — Contradicción insoluble entre source binding y regresión obligatoria

El plan exige simultáneamente:

- conservar byte-exacto `tests/test_wave59_prospective.py` ([líneas 173–177](/mnt/m2-1TB/Phideus/Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_59_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:173));
- cambiar sólo preparador, runner y el test nuevo, con sólo dos deltas dentro de las 33 fuentes congeladas ([líneas 181–199](/mnt/m2-1TB/Phideus/Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_59_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:181));
- mantener rechazo estricto para paquetes no recuperados ([líneas 218–231](/mnt/m2-1TB/Phideus/Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_59_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:218));
- ejecutar la regresión completa Wave 59 ([línea 281](/mnt/m2-1TB/Phideus/Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_59_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:281)).

Pero `tests/test_wave59_prospective.py` pertenece a las 33 fuentes y su fixture físico construye un paquete **no-recovery** antes de invocar `runner.execute()` en [test_wave59_prospective.py](/mnt/m2-1TB/Phideus/tests/test_wave59_prospective.py:176); el replay vuelve a hacerlo en [línea 265](/mnt/m2-1TB/Phideus/tests/test_wave59_prospective.py:265). El runner autentica todas las fuentes contra los hashes viejos del config antes de inspeccionar el paquete ([run_wave59_hgb_guard_bracket.py](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave59_hgb_guard_bracket.py:349), llamada inicial en [línea 2431](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/run_wave59_hgb_guard_bracket.py:2431)).

Tras cambiar preparador y runner, esos tests normales necesariamente fallarán por los dos hashes nuevos. No existe una implementación que satisfaga a la vez todas las cláusulas actuales sin introducir un bypass oculto.

**Corrección requerida:** declarar una estrategia de tests realizable. La opción más trazable es:

1. autorizar también la modificación de `tests/test_wave59_prospective.py`;
2. incorporar su hash old/new como tercer delta de las 33 fuentes y reducir de 31 a 30 las fuentes inmutables;
3. adaptar el fixture físico para presentar provenance de recovery sintético autenticado;
4. conservar pruebas explícitas de que un paquete normal/no-recovery rechaza los hashes nuevos;
5. ligar ambos tests —el adaptado y el nuevo— en la auditoría de implementación.

No debe resolverse mediante detección de pytest, un bypass genérico ni debilitando la validación productiva.

#### P2 — El schema del inventario físico está descrito de forma ambigua

El plan exige tipo, modo, uid, gid, tamaño y hash para las 26 entradas, incluidas seis carpetas ([líneas 57–61](/mnt/m2-1TB/Phideus/Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_59_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:57)). La primitiva segura existente sólo asigna `bytes` y `sha256` a archivos; las carpetas tienen path, tipo, modo, uid y gid ([prepare_wave56_fresh.py](/mnt/m2-1TB/Phideus/experiments/geometria_proporcional/prepare_wave56_fresh.py:365)).

**Corrección recomendada:** explicitar que tamaño y SHA-256 son obligatorios únicamente para archivos regulares.

#### P2 — Falta explicitar continuidad de identidad entre las dos etapas

La etapa content-blind usa correctamente `O_NOFOLLOW → fstat → streaming SHA-256`, pero el plan no exige una nueva comprobación de identidad inmediatamente antes del primer parseo semántico. Entre cerrar el descriptor opaco y reabrir el path queda una ventana TOCTOU.

**Corrección recomendada:** exigir que la etapa semántica comience revalidando el inventario físico completo mediante la misma primitiva content-blind —o que consuma snapshots/descriptores autenticados— y agregar una prueba que sustituya un archivo sensible entre etapas y demuestre rechazo antes del parseo.

### Aspectos validados

- **Mismo escrow:** el diseño prohíbe redraw, fija escrow, freeze y manifest exactos, impide `secrets.token_bytes` en recovery/replay y exige revalidación posterior a la regeneración. Es sólido una vez corregidos los findings.
- **Frontera content-blind/semántica:** la whitelist pública y la lista de paths sensibles son precisas; los spies y el probe contra el origen real son adecuados. Falta sólo cerrar explícitamente la continuidad entre etapas.
- **Chicken-and-egg de fuentes:** la aceptación provisional limitada a old/new hashes permite construir el contrato sin abrir escrow ni output; la autoridad completa posterior es conceptualmente correcta.
- **Continuidad del runner:** amendment copiado, provenance, preparation freeze, HEAD exacto y revalidación antes de cada fase forman un contrato suficiente.
- **Cadena Git:** confirmé una cadena limpia: el commit auditado tiene parent `1ef98a109413b55d8dcf4a5df2171d7017c7cd57`, introduce únicamente el plan de 329 líneas y actualmente es HEAD exacto con worktree limpio.
- **Evidencia pública del origen:** 26 entradas exactas —6 directorios y 20 archivos—, ownership y modos coincidentes. Los seis hashes principales y el manifest coinciden con el plan; inventario y atestación de fallo verifican; el manifest cubre exactamente sus 14 registros; los tres visibles tienen 4992 filas.
- **CPU:** el protocolo es viable dentro de 1800 s/8 GiB por corrida y 3600 s combinados. La regresión inmediatamente anterior sobre la misma implementación —el commit actual sólo agregó el plan— terminó con `222 passed` en aproximadamente 278 s. No hay justificación para GPU.

No abrí semánticamente `generation_escrow.json`, secret files, commitments ni truth sellada; sólo fueron objeto de inventario, metadatos y hashing binario opaco. No realicé ediciones ni commits.
