# Wave 56 Stage 1 — Plan de amendment para la entrada a fase del recovery

**Estado:** `FROZEN_IMPLEMENTATION_PLAN`
**Alcance:** corrección operativa mínima posterior a `PREPARED`, sin cambio de protocolo científico
**Decisión científica:** reservada a Mariano; este plan no declara `GO/NO-GO`

## 1. Motivo y evidencia observada

La recuperación pre-oráculo v2 fue aprobada por R381 y R382. El commit final
`2b36c6c8d9ebd77e0fd681d67f8ed1add0b5fc0c` permitió ejecutar el recovery
oficial desde el escrow preservado. El preparador terminó con:

```text
{"execution_mode": "recovery", "state": "PREPARED"}
```

El primer ingreso oficial a `fit` abortó antes de abrir labels u oracle con:

```text
RuntimeError: prepared visible inventory differs from preparation freeze
```

La comparación pública del paquete dejó estos hechos:

- claves esperadas por `validate_prepared_package`:
  `train.jsonl`, `val.jsonl`, `lockbox.jsonl`;
- claves físicas observadas bajo `benchmark/visible/`:
  `calibration_null.jsonl`, `train.jsonl`, `val.jsonl`, `lockbox.jsonl`;
- único path extra respecto del conjunto construido por el runner:
  `calibration_null.jsonl`;
- los hashes de `train`, `val` y `lockbox` coincidieron exactamente;
- `phases/` y `authorized_labels/` permanecieron ausentes;
- el escrow y el freeze público republicados fueron byte-exactos respecto del
  intento fallido original.

El paquete oficial alcanzó correctamente `PREPARED`; el error pertenece al
borde de validación que precede a la primera fase. El primario observado se
conserva intacto como evidencia y no se usa como fuente de nuevas claves.

## 2. Observación, hipótesis e inferencia

**Observación.** `validate_prepared_package` calcula el inventario físico de
todo `benchmark/visible/`, pero construye el inventario esperado sólo con los
tres splits de fase. El generador oficial incluye además la población visible
de calibración nula, autenticada por el manifest.

**Hipótesis causal.** La igualdad falla por una omisión del inventario esperado,
no por corrupción de los tres splits, divergencia del escrow, redraw, cambio de
fuentes ni materialización prematura de verdad.

**Inferencia acotada.** Debe corregirse la composición del inventario esperado
y agregarse una prueba que haga atravesar a un paquete físico generado la
validación previa a fase. Las suites anteriores no autorizan continuar mediante
un bypass: demostraron propiedades parciales, pero no cubrieron este layout
oficial completo.

## 3. Corrección mínima propuesta

### 3.1 Runner de fases

En
`experiments/geometria_proporcional/run_wave56_contextual_gate.py`,
`validate_prepared_package` debe:

1. conservar los hashes de `train`, `val` y `lockbox` fijados en
   `preparation_freeze.json`;
2. incorporar al inventario esperado todos los miembros `visible/` declarados
   por el `benchmark/manifest.json`, incluido `calibration_null.jsonl`;
3. exigir concordancia cuando un split aparece tanto en el freeze como en el
   manifest;
4. comparar el mapa completo contra `hash_inventory(benchmark/visible)` para
   seguir rechazando faltantes, mutaciones y archivos extra no manifestados.

No se elimina ni relaja la validación miembro por miembro que ya exige hash y
tamaño desde el manifest.

### 3.2 Validador del amendment

El contrato público del escrow fija los hashes de todas las fuentes requeridas.
Cambiar el runner crea necesariamente un segundo delta de fuente además del
preparador ya corregido. Por ello,
`experiments/geometria_proporcional/prepare_wave56_fresh.py` debe introducir un
nuevo path canónico y un schema de amendment v3 que:

- autorice exactamente dos deltas de fuente respecto del contrato de origen:
  el preparador y el runner;
- declare para ambos `path`, `old_sha256` y `new_sha256`;
- exija que el commit de implementación cambie exclusivamente preparador,
  runner y test focal de recovery;
- verifique los blobs de los tres paths desde el commit declarado;
- agregue el SHA-256 del runner al bloque canónico de la auditoría de
  implementación;
- preserve parents directos, introducciones únicas, `HEAD == F`, worktree
  limpio, reportes Markdown no ejecutables y todos los controles v2.

Hashes de origen ya congelados:

```text
prepare_wave56_fresh.py
7ff5919d2b0bdd607ca179180c4f94de3ff5be6e23e6024b21e748d22c61fb44

run_wave56_contextual_gate.py
304d27fa6ee2e6d511c5acef4f19c3990bd3af28cb207c5b43760f8d5efbda15
```

### 3.3 Test físico de regresión

`tests/test_wave56_preoracle_recovery.py` debe extender el recorrido físico
sintético de recovery/replay para comprobar, antes de abrir fases, que:

- el paquete generado contiene `visible/calibration_null.jsonl`;
- `validate_prepared_package` acepta el inventario íntegro no alterado;
- una mutación o un archivo visible extra no manifestado sigue siendo
  rechazado antes de iniciar una transacción de fase.

El test conserva los negativos existentes de no-redraw, origen físico,
cabeceras, DAG, manifest del primary y exactitud de replay.

## 4. Archivos permitidos en la implementación

El nuevo commit de implementación `I3` puede modificar exclusivamente:

1. `experiments/geometria_proporcional/prepare_wave56_fresh.py`;
2. `experiments/geometria_proporcional/run_wave56_contextual_gate.py`;
3. `tests/test_wave56_preoracle_recovery.py`.

No se modifican generador, config prospectiva, seeds, modelos, checkpoints,
estimando, selector, umbrales, workers, fuentes históricas ni criterios de
diagnóstico.

## 5. Nueva cadena de autorización

La cadena v2 queda preservada como evidencia histórica del recovery que llegó a
`PREPARED` y reveló el defecto. No se reescriben R381, R382 ni el amendment v2.

La continuación usa una cadena nueva:

```text
P3 → I3 → A3 → J3 → F3
```

- `P3`: commit exclusivo de este plan;
- `I3`: implementación exclusiva de los tres paths permitidos;
- `A3`: auditoría independiente de implementación, prevista como R384;
- `J3`: introducción única del JSON v3;
- `F3`: auditoría final independiente, prevista como R385.

Path canónico nuevo:

```text
experiments/geometria_proporcional/configs/wave56_stage1_phase_entry_recovery_amendment_v3.json
```

Informes previstos:

```text
Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/384_wave56_stage1_phase_entry_recovery_implementation_audit.md
Biblioteca/Geometria_Proporcional_Ground_Truth/agent_reports/385_wave56_stage1_phase_entry_recovery_final_package_audit.md
```

`A3`, `J3` y `F3` deben ser hijos directos sucesivos. Cada commit cambia un
único path, salvo `I3`, que cambia exactamente los tres declarados. El JSON v3
conserva el inventario físico y hashes del intento fallido original y agrega
las identidades exactas de `P3`, `I3` y `A3`.

## 6. Matriz mínima de auditoría

La auditoría de implementación debe leer completos plan, código y tests
vigentes y verificar:

- reproducción del fallo con el inventario oficial ya preservado;
- aceptación del paquete íntegro con calibración nula manifestada;
- rechazo de calibración mutada, faltantes y visibles extra;
- segundo delta de fuente limitado exactamente al runner;
- no ampliación de la autoridad del amendment a un primary fresco;
- no-redraw y reutilización exclusiva del escrow original;
- parents, paths, blobs, hashes y cabeceras canónicas;
- focal completa CPU-only.

La auditoría final debe repetir la focal y la suite amplia exacta Wave 49–56,
además de validar `P3 → I3 → A3 → J3` y el JSON v3. Un `REVISE` no autoriza
ejecución, aunque su cabecera accidental dijera otra cosa: se debe leer el
informe completo.

## 7. Reejecución oficial

Sólo después de `HEAD == F3` y worktree globalmente limpio:

1. ejecutar recovery desde el intento fallido original con el amendment v3;
2. usar `--force` para que el preparador archive de forma recuperable el
   primario v2 que quedó en `PREPARED` antes de publicar el nuevo primario;
3. confirmar `PREPARED` y ejecutar `fit`, `select`, `adjudicate` en orden;
4. preparar un replay nuevo desde el primary recuperado, con el mismo amendment
   y referencia canónica;
5. ejecutar sus tres fases y exigir replay exacto;
6. preservar checkpoints, arrays, receipts, manifests, freezes y estados crudos
   para reanálisis.

No se elimina el intento fallido original ni el primario v2 archivado. Si una
transacción nueva falla, se conserva bajo el mecanismo normal de archivos
`failed`/`superseded`.

## 8. Recursos y frontera GPU

El protocolo prospectivo declara `device: cpu` y sus workers de fase son CPU
aislados. Las pruebas focales y amplia también son CPU-only. No se usa ni se
consulta GPU para este amendment. Si apareciera una operación nueva cuya forma
correcta requiriese GPU, el trabajo se detiene antes de ejecutarla y se informa
a Mariano con objetivo, duración y VRAM estimada.

## 9. Criterio de cierre

El amendment queda procedimentalmente listo sólo si:

- el bug oficial queda cubierto por un test físico que falla antes del parche;
- focal y suite amplia terminan sin fallos ni skips materiales;
- las dos auditorías independientes concluyen `PASS` de forma consistente en
  cabecera, cuerpo y decisión;
- la DAG y los diffs exclusivos se verifican desde objetos Git;
- recovery y replay oficiales completan sus estados previstos;
- toda interpretación posterior separa observación, hipótesis e inferencia y no
  declara por sí sola `GO/NO-GO`.
