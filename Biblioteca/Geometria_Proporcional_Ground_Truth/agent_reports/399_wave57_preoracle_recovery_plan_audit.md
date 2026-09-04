# R399 — Auditoría independiente del plan de recuperación pre-oracle de Ola 57

**Plan commit:** `b5edbec5bcc1e48f13fb25e6f16fcf6a8e029335`
**Plan SHA-256:** `ccf5b6b010bfc9eba893d1a87f78a6606ef2142d8c0d75f4bd289af22e87f9d5`
**Result:** `REVISE`

## Dictamen ejecutivo

La recuperación del draw preservado es metodológica y materialmente defendible.
El campo prospectivo ya significaba `768` pair tokens elegibles por split antes
de extraer las claves, el intento falló porque la ruta primaria comparó ese
valor con los `1152` tokens totales y el árbol quedó detenido antes de
inferencia, oracle materializado y labels autorizados. No redibujar es la única
conducta compatible con el escrow existente.

El plan no debe implementarse todavía de manera literal. Su cadena de autoridad
exige validar el origen antes de extraer las claves y pide un probe real sin
abrir labels, pero no congela la separación técnica necesaria entre validación
pre-key y validación semántica. El helper vigente que el plan propone extender
ya abre el escrow, materializa sus tres claves, vuelve a abrir las mismas claves
desde el benchmark y carga truth sellada. Sin una obligación explícita y tests
negativos contra esas lecturas, una implementación aparentemente conforme puede
consumir las claves antes de que la cadena de autoridad haya terminado de
validarse. Este hueco afecta la frontera que legitima el recovery; no es una
observación cosmética.

## Findings priorizados

### ALTO — La validación pre-key no tiene una ruta cerrada que evite extraer las claves

**Observación.** El plan ordena que el nuevo validator compruebe la autoridad
"antes de extraer claves" y, dentro de ese mismo bloque, exige validar la
atestación semántica del origen
(`WAVE_57_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:95-114`). También exige que el
probe final opere contra el origen real sin abrir labels
(`WAVE_57_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:164-166`). Sin embargo, el
validator compartido vigente hace exactamente lo contrario:

- `validate_failed_recovery_origin()` llama a `read_escrow()` y luego a
  `validate_semantic_attestation()`
  (`experiments/geometria_proporcional/prepare_wave56_fresh.py:928-975`);
- `read_escrow()` lee el JSON completo y convierte los tres valores secretos
  desde hexadecimal antes de devolver
  (`experiments/geometria_proporcional/prepare_wave56_fresh.py:748-765`);
- `validate_semantic_attestation()` abre los tres archivos `*_secret.json`,
  convierte otra vez sus claves y carga todos los JSONL de truth sellada
  (`src/geometria_proporcional/wave49_checker.py:231-276`).

El test pedido para hacer imposible `secrets.token_bytes` sólo demuestra que no
se genera un draw nuevo
(`WAVE_57_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:145-162`). No demuestra que el
validator no abra o extraiga claves antes de autenticar plan, auditorías,
implementación, amendment, HEAD, inventario y contrato. Los tests de recovery
vigentes tampoco cierran esa propiedad: prueban no-redraw mediante monkeypatch
de `secrets.token_bytes`, pero el contexto sintético evita el validator físico
real (`tests/test_wave56_preoracle_recovery.py:862-929`).

**Impacto.** El orden de autoridad queda parcialmente autodeclarado. Una
regresión o una reutilización directa del helper actual podría leer las claves
del draw antes de verificar que el amendment, la cadena Git y el informe final
son los autorizados. Además, el supuesto probe "sin abrir labels" puede cargar
truth sellada mediante un helper cuyo nombre no revela ese efecto. La ausencia
de inferencia u oracle materializado en el árbol no corrige ese acceso
prematuro.

**Corrección obligatoria.** El plan debe congelar dos etapas distintas:

1. una validación de autoridad pre-key que no llame a `read_escrow()`,
   `keys_from_escrow()` ni a un helper que abra `benchmark/sealed/*_secret.json`;
2. recién después de que toda la cadena pase, una única extracción del escrow,
   seguida de comprobación de que contrato y commitments coinciden con el
   freeze público y con el inventario autenticado.

La etapa pre-key puede verificar el SHA-256 del escrow sin parsearlo, usar el
contrato secret-free de `pre_generation_freeze.json`, verificar la firma
detached de `semantic_root.json` con la clave pública y contrastar los
hashes/tamaños ya firmados contra el inventario físico. Si para computar las
poblaciones se permite leer sólo los campos de clasificación de los JSONL
sellados, esa excepción mecánica debe nombrarse con precisión y no debe abrir
valores de claves ni materializar targets analíticos.

La suite debe instrumentar y rechazar, durante la etapa pre-key, cualquier
llamada a `read_escrow()` o apertura de los tres `*_secret.json`; también debe
probar que el acceso único posterior sólo ocurre tras completar todos los
checks. Costo: medio y localizado en los dos archivos ya autorizados. Efecto:
hace ejecutable la frontera que el plan declara y evita que la separación
Wave 56/Wave 57 herede silenciosamente el orden de lectura legacy.

### MEDIO — Las aserciones de ausencia son más amplias que la evidencia física

**Observación.** El amendment propuesto exige literalmente
`no_oracle_in_origin` y `no_labels_in_origin`
(`WAVE_57_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:95-103`), mientras el propio
origen contiene truth sellada autenticada. El receipt público enumera
`sealed_truth` para `train`, `val`, `lockbox` y `calibration_null`
(`benchmark/attestations/semantic_root.json:15-37`), y las filas selladas
contienen, entre otros, `family_id`, `target_region`, `clean_y` y
`true_covariance_canonical`. Lo que sí está ausente es
`benchmark/sealed/oracle/` y el árbol `authorized_labels/`.

**Impacto.** La lectura literal vuelve falsa una aserción de autoridad; la
lectura contextual depende de saber que "labels" significa sólo labels
autorizados/materializados y que "oracle" significa sólo el oracle analítico
materializado. Esa ambigüedad importa porque el plan usa esas aserciones como
precondiciones machine-verifiable y como frontera de acceso.

**Corrección obligatoria.** Renombrar o definir inequívocamente las aserciones,
por ejemplo `no_materialized_oracle_in_origin` y
`no_authorized_labels_in_origin`, y ligarlas a la whitelist cerrada de paths.
Separar en el texto la existencia esperada de truth sellada de la ausencia de
sus derivados analíticos. Los negativos deben usar esas rutas y semánticas
exactas, no una búsqueda nominal abierta.

## Evidencia del origen revalidada

- El commit auditado es HEAD y su único path es el plan. El padre directo es el
  commit de origen del escrow `379229f1cae0f1b713fe5c293f303ed60ed7f187`.
- El SHA-256 del plan coincide exactamente con el valor del bloque inicial.
- El árbol fallido tiene `24` entradas: `6` directorios y `18` archivos
  regulares, `0` symlinks y `0` tipos especiales. Todo pertenece a `root:root`;
  los directorios son `0700`, los archivos son `0600` salvo
  `pre_generation_freeze.json`, que es `0644`.
- No existen `inference/`, `authorized_labels/`, `bundles/`, `phases/`,
  `preparation_freeze.json`, `generation_receipt.json` ni
  `benchmark/sealed/oracle/`. El error preservado es exactamente el guard de
  conteo y declara escrow presente
  (`wave57_contextual_tail_guard_fresh_v1.failed_20260904T215941566314Z/FAILURE.json:1-6`).
- Los hashes de contrato `57e727120346...`, escrow `d489f443a5c8...`, freeze
  `20c94171a053...`, failure `710b7d29de8...` y manifest
  `3d444db8a1e7...` coinciden con el plan
  (`WAVE_57_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:33-58`).
- Las `33` fuentes del contrato embebido coinciden con los archivos vigentes;
  no hubo ningún mismatch. En particular, el preparador conserva
  `796b5e8f580c...` y el test Wave 57 conserva `5238778afb00...`, los hashes
  anclados por el freeze público
  (`pre_generation_freeze.json:444-468`).
- Cada split contiene `4992` filas, `1152` pair tokens totales, `768`
  elegibles, `384` OOD, `192` no canónicos y una intersección
  elegible/no-canónica de `192`. Esto confirma que el filtrado debe ocurrir por
  fila antes de deduplicar, exactamente como implementa
  `sealed_population_counts()`
  (`experiments/geometria_proporcional/prepare_wave56_fresh.py:366-390`).

La inspección no mostró ni publicó valores de claves, no materializó labels, no
ejecutó inferencia, training, oracle ni GPU/CUDA.

## Controles del plan que sí son suficientes

### Validez del recovery y prohibición de redraw

La config pre-draw fija `expected_eligible_pair_tokens_per_split=768` y
`no_redraw_after_escrow=true`
(`experiments/geometria_proporcional/configs/wave57_contextual_tail_guard_fresh.json:186-190`).
El código fallido usa `total_unique_pair_tokens` fuera de recovery y el campo
elegible sólo cuando existe `recovery_context`
(`experiments/geometria_proporcional/prepare_wave56_fresh.py:1768-1795`). El
origen preservado reproduce exactamente la discrepancia `768 != 1152`; no hay
evidencia de una realización anómala. La ruta de invocación ya prohíbe un nuevo
primary cuando existe un archive con escrow y exige recovery desde ese archive
(`experiments/geometria_proporcional/prepare_wave56_fresh.py:538-590`).

### Inventario, hashes y poblaciones

El plan congela basename, hashes, inventario físico, ownership, modos, conteos y
doble revalidación del origen antes y después de regenerar
(`WAVE_57_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:33-81,95-114,168-190`). También
preserva explícitamente el solapamiento de las categorías mediante el conteo de
intersección; no repite la ambigüedad que afectó la primera recuperación de
Ola 56.

### Cadena Git y source contract

La secuencia de seis commits liga directamente plan, plan-audit,
implementación, implementation-audit, amendment y final-audit, exige commits de
un solo propósito, HEAD exacto y worktree globalmente limpio
(`WAVE_57_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:83-136`). La cadena evita los
bypasses históricos de commit ancestro con blob distinto y de informe
autorreferencial.

El contrato original ya incluye como fuentes tanto
`prepare_wave56_fresh.py` como `tests/test_wave57_prospective.py`
(`pre_generation_freeze.json:284-313,444-468`). Por eso es posible mantener
idénticos config, bindings, inventario de fuentes y todos los demás campos,
cambiando sólo `git_commit` y esos dos hashes. La implementación puede hacerse
sin modificar el config científico.

### Separación Wave 56/Wave 57 y alcance de dos archivos

El dispatch por `prospective_config.schema_version` es la frontera correcta:
ambas configs ya se validan por schema y Wave 57 ya enlaza el preparador
compartido como interfaz versionada
(`experiments/geometria_proporcional/prepare_wave56_fresh.py:422-525`;
`experiments/geometria_proporcional/prepare_wave57_fresh.py:14-23`). No hace
falta cambiar wrapper, generator, worker de inferencia, materializador,
coordinador, modelos ni config. Es factible concentrar implementación y tests en
los dos paths propuestos, siempre que los negativos obligatorios prueben tanto
la ruta histórica Wave 56 como la nueva autoridad Wave 57 y se cierre el finding
pre-key anterior.

### Replay

La matriz exigida cubre source/reference canónicos, misma copia de amendment,
escrow y freeze byte-exactos, manifest regenerado, visibles, logits, poblaciones
y provenance (`WAVE_57_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:109-114,145-186`).
El comparador compartido ya propaga esos controles y el manifest autentica sus
miembros sellados; el coordinador Wave 57 añade los freezes y artefactos de las
fases. No se encontró un hueco nuevo de replay que requiera ampliar el delta de
archivos.

## Condiciones concretas para una reauditoría PASS

1. Especificar la validación pre-key metadata-only y el único punto posterior de
   extracción del escrow.
2. Añadir negativos que fallen ante lectura del escrow o de cualquier
   `*_secret.json` durante la validación de autoridad, no sólo ante invocación de
   `secrets.token_bytes`.
3. Definir las aserciones como ausencia de oracle materializado y labels
   autorizados, distinguiéndolas de la truth sellada esperada.
4. Mantener sin cambios las restantes garantías de hashes, inventario,
   poblaciones, DAG Git, source-contract dual, replay y separación Wave 56/57.

## Veredicto

**REVISE.** No corresponde redibujar ni abandonar el draw; la población, hashes
y estado pre-oracle sostienen la recuperación del mismo escrow. La revisión se
limita a cerrar la secuencia real de acceso a claves y a volver literales las
aserciones de ausencia. No cambia config científico, estimando, modelos,
features, seeds, mínimos, criterios ni políticas, y no constituye un GO/NO-GO
científico.

## Machine-verifiable decision

**Final decision:** `REVISE`

