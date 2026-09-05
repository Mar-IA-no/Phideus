# Auditoría independiente — plan sucesor Wave 59 con replay normalizado

**Plan commit:** `4204ac6e29e00dee863e40f5a122ea97a5d67712`  
**Plan SHA-256:** `45de7f77898941c5241fd779c8f959f34b7c459f270b5da86655257f551c88d4`  
**Resultado:** `REVISE`

## Dictamen: REVISE

La normalización tipada propuesta es conceptualmente correcta, pero el protocolo no es ejecutable con el alcance de implementación declarado. Un primario realmente fresco volvería a fallar antes de preparación completa y, aun corrigiendo ese primer fallo, no podría publicar la atestación que el replay exige. Además, la auditoría final de config figura en la secuencia humana, pero no queda convertida en una autoridad fail-closed del preflight.

## Findings

### BLOQUEANTE — El draw fresco repite determinísticamente el defecto de conteo pre-oracle

El plan ordena un draw nuevo con claves nuevas y afirma que el único cambio de implementación será la semántica del checker de replay: `WAVE_59_REPLAY_NORMALIZATION_SUCCESSOR_PLAN.md:10-14`, `25-26`, `45-49`.

Sin embargo, la ruta primaria fresca vigente conserva exactamente el defecto que produjo el primer archivo fallido:

- la config congela `expected_eligible_pair_tokens_per_split=768`: `wave59_fresh_hgb_guard_bracket.json:242-246`;
- sin `recovery_context`, el preparador selecciona `total_unique_pair_tokens`: `prepare_wave56_fresh.py:3628-3634`;
- luego compara ese total con `768`: `prepare_wave56_fresh.py:3635-3639`;
- la ley generativa conservada produce `1152` tokens totales y `768` elegibles: `WAVE_59_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:17-33`.

El nuevo primario entra como modo `primary`, no como recovery: `prepare_wave56_fresh.py:673-689`. Tampoco puede usar un amendment como atajo, porque el preparador prohíbe expresamente autorizar un fresh primary mediante amendment: `prepare_wave56_fresh.py:654-656`.

Por tanto, cambiar sólo módulo, runner y test prospectivo —el alcance fijado en `WAVE_59_REPLAY_NORMALIZATION_SUCCESSOR_PLAN.md:150-154` y `177-178`— garantiza otro fallo antes de inferencia y antes del replay.

Corrección requerida:

- incluir `experiments/geometria_proporcional/prepare_wave56_fresh.py` en el commit de implementación;
- tipar en la config sucesora que el conteo contractual es `eligible_unique_pair_tokens`;
- validar ese campo de forma exacta y usarlo sólo para la identidad sucesora, sin alterar silenciosamente la semántica histórica;
- probar un primario fresco donde `total=1152` y `eligible=768` sea aceptado por el contrato correcto.

### BLOQUEANTE — La atestación de preparación no admite un primario fresco

Aun corregido el conteo, el primario sucesor fallaría al publicar su atestación:

- `wave59_preparation_attestation_payload()` sólo admite `recovery` o `replay`: `prepare_wave56_fresh.py:3356-3362`;
- exige `recovery_provenance`: `prepare_wave56_fresh.py:3363-3366`;
- incluye incondicionalmente `recovery_amendment.json` entre los records firmados: `prepare_wave56_fresh.py:3382-3405`;
- un primario fresco produce `provenance=None` y no publica amendment: `prepare_wave56_fresh.py:3566-3578`;
- el flujo Wave 59 llama de todos modos al publicador de la atestación al cerrar preparación: `prepare_wave56_fresh.py:3967-3977`.

Esto contradice tanto el draw fresco del plan como la exigencia de verificar realmente ambas preparation attestations: `WAVE_59_REPLAY_NORMALIZATION_SUCCESSOR_PLAN.md:76-77`, `110-113`, `219-221`.

Corrección requerida:

- ampliar el preparador con una unión tipada y cerrada:

  - sucesor fresco: `primary/replay`, sin provenance ni amendment;
  - recovery histórico: `recovery/replay`, con provenance y amendment;

- prohibir mezclas entre ambas formas;
- reconstruir el payload físico de cada forma antes de verificar la firma;
- añadir pruebas reales de firma para `primary ↔ replay`, además de conservar la regresión `recovery ↔ replay`.

El lugar natural para esa cobertura es también `tests/test_wave59_preoracle_recovery.py`, que ya contiene los fixtures reales de firma y paquete. En consecuencia, el alcance mínimo recomendable del commit pasa de tres a cinco archivos: módulo, preparador, runner, test prospectivo y test de preparación/recovery.

### ALTO — La auditoría final de config no queda ligada al preflight

La cadena declarada exige:

1. config congelada;
2. auditoría final independiente;
3. recién después, preflight y draw.

Eso figura en `WAVE_59_REPLAY_NORMALIZATION_SUCCESSOR_PLAN.md:180-184`. Pero la especificación del propio preflight sólo exige blobs de HEAD, ancestralidad del implementation commit, PASS de su auditoría y self-binding de config: `WAVE_59_REPLAY_NORMALIZATION_SUCCESSOR_PLAN.md:187-190`.

La implementación vigente hace exactamente eso y nada más:

- el preparador valida sources, implementation commit y implementation audit: `prepare_wave56_fresh.py:784-824`;
- el runner vuelve a comprobar sólo implementation commit/audit: `run_wave59_hgb_guard_bracket.py:410-428`.

Así, la config podría ejecutarse inmediatamente después de su commit, antes de la auditoría final. Esa auditoría también podría faltar o dejar de estar en HEAD sin que el preflight lo detecte.

Corrección requerida:

- agregar a la config una ruta predeclarada de auditoría final;
- exigir en preflight que:

  - su commit sea HEAD;
  - cambie únicamente el informe;
  - sea hijo directo del commit de config;
  - el informe nombre el commit de config, su hash bruto o self-binding y `PASS`;
  - el worktree esté globalmente limpio.

La relación Git dinámica evita el ciclo imposible de intentar almacenar dentro de la config el SHA de un informe que todavía no existe.

### MEDIO — La igualdad del contrato científico queda demasiado apoyada en revisión manual

El plan declara invariancia de features, targets, thresholds, controles, criterios y fuentes: `WAVE_59_REPLAY_NORMALIZATION_SUCCESSOR_PLAN.md:30-49`. Sin embargo, la config nueva se crea después del commit de implementación y no se define un proyector de diferencias permitidas respecto de la config original.

Los validators actuales cubren una parte importante del contrato: `wave59_hgb_guard_bracket.py:160-302` y `prepare_wave56_fresh.py:482-529`. No fijan exhaustivamente todos los campos result-affecting; por ejemplo, los thresholds de `pattern` están en `wave59_fresh_hgb_guard_bracket.json:186-205`, pero no forman parte de esas validaciones exactas.

La auditoría final podría detectar una deriva, pero el plan no le exige comparar ambas configs mediante una allowlist de diferencias.

Corrección requerida:

- definir exactamente qué campos pueden cambiar:

  - identidad y rutas de output;
  - config self-source;
  - revisión operacional, plan/auditorías e implementation binding;
  - el nuevo campo tipado que corrige el conteo elegible;

- exigir igualdad canónica de todo el resto contra la config original;
- incorporar esa comparación al validator o, como mínimo, convertirla en check reproducible obligatorio de la auditoría final.

### MEDIO — La matriz de pruebas permite no probar el modo que se ejecutará realmente

El plan permite que el caso integrado materialice “`recovery` o `primary` frente a `replay`”: `WAVE_59_REPLAY_NORMALIZATION_SUCCESSOR_PLAN.md:121-124`. Para este protocolo no son intercambiables: el primario sucesor es fresco, carece de provenance y amendment, y actualmente cae en las dos rutas bloqueantes anteriores.

La prueba acreditante debe exigir específicamente `primary ↔ replay`. La variante `recovery ↔ replay` puede mantenerse como regresión separada.

También faltan negativos explícitos para la nueva allowlist de config:

- ambas configs canónicas aceptadas individualmente;
- path con prefijo arbitrario y sufijo legacy rechazado;
- ambas configs presentes simultáneamente en `required_execution_sources` rechazadas;
- config ejecutada distinta de la única self-source declarada rechazada.

Para preservar el antecedente, conviene congelar hashes públicos concretos y comprobarlos antes y después del test sucesor. El estado actual confirma:

- primario anterior: `scientific_decision=null`, ambos `replay_exact=PENDING`, ambos `aggregate_with_replay=null`;
- `analysis.json`: `79a4c1eca78497d9fd6ea49177508ac9f879cabe0824b566b437d8b9f8e7eb96`;
- `artifact_manifest.json`: `ce30675744b27d656ff7eb177145ae5fac77d9f092ccc483d329425864185770`;
- replay fallido `FAILURE.json`: `4967c6108ffd07e1b8cf6c0f301e702722be592f0d678594dd89e659053aa3c4`;
- `failure_inventory.json`: `353afe6f61c476a82d5c061fadcdfa13bad5c27787629189cb80094d4769b9a9`;
- `failure_attestation.json`: `8cb19ce7d741a0caaa73c88d44c96fe10557dbba6283290f90e98bbec0cb93e3`.

## Aspectos conformes

- Preservar el draw anterior como diagnóstico no adjudicable es correcto. R445 y los artefactos públicos mantienen `replay_exact=PENDING`.
- Un draw nuevo es necesario: validation y monitor ya fueron abiertos, por lo que un fix posterior no puede promover el draw previo.
- El orden de la normalización tipada es correcto: primero hash bruto local, luego reconstrucción y firma, después equivalencia semántica.
- No agregar `generation_receipt_sha256` al omit-set global evita silenciar enlaces rotos.
- La allowlist exacta de paths de config es el diseño correcto frente al `endswith()` vigente.
- El presupuesto CPU es proporcionado. HGB de scikit-learn es CPU-native; no se justifica GPU.
- El plan no declara `GO/NO-GO` ni promueve arquitectura.

## Condición para PASS

La revisión debe, como mínimo:

1. incorporar preparador y su cobertura de pruebas al alcance de implementación;
2. corregir de forma tipada el conteo elegible del primario sucesor;
3. soportar y autenticar paquetes Wave 59 frescos sin recovery provenance;
4. hacer que la auditoría final sea una condición material del preflight;
5. cerrar las diferencias permitidas entre config original y sucesora;
6. exigir un test integrado real `primary ↔ replay`, sin monkeypatch de attestations.

No se editaron archivos, no se abrió material secreto o sellado y no se usó ni consultó GPU. El worktree permaneció limpio en `4204ac6e29e00dee863e40f5a122ea97a5d67712`.
