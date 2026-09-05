# Ola 59 — protocolo sucesor con normalización tipada del replay

> **Estado:** `R446-REVISE-INCORPORATED / PRE-IMPLEMENTATION / NEW-PROTOCOL / NEW-DRAW / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-05
> **Antecedente no adjudicable:** `wave59_fresh_hgb_guard_bracket_v1`
> **Auditoría causal:** `445_wave59_postmonitor_replay_failure_audit.md`

## Propósito

Este protocolo repite el bracket HGB congelado de Ola 59 sobre un draw nuevo,
después de corregir antes del freeze un defecto estrictamente operacional en la
comparación primary/replay. No reabre ni reinterpreta el draw anterior, no
modifica sus receipts y no transforma su replay fallido en una adjudicación
prospectiva.

El primario anterior alcanzó `COMPLETE`; el replay ejecutó también todas las
fases y fue archivado después de `monitor_evaluate`. La comparación independiente
mostró igualdad en `22/22` artefactos científicos exactos, `21/21` arrays,
`28/28` checks de preparación, los 16 estados portables y todos los secretos
comparados por hash. El único desacuerdo normalizado fue
`preparation_receipt.generation_receipt_sha256`. R445 lo clasificó como falso
negativo operacional y confirmó que, después de abrir validation y monitor, un
delta de código no puede adjudicar ese draw.

La nueva ejecución conserva el contrato científico. En la frontera operacional
cambia la comparación tipada del replay y agrega sólo la mecánica necesaria para
que un primario fresco pueda satisfacer el conteo y la atestación ya previstos.
El resultado seguirá separando observación, interpretación y decisión;
`GO/NO-GO` y promoción de arquitectura pertenecen a Mariano.

## Invariantes científicos

El protocolo sucesor conserva byte por byte o por igualdad estructural
predeclarada:

- arquitectura HGB, proposer ridge y controles;
- features, targets, utility matrix y penalty;
- roster factorial de 36 celdas y políticas principales;
- seeds de modelos, controles, bootstrap y sharding;
- cuantiles, comparación estricta, mínimos y criterios;
- splits físicos y fronteras FIT/CALIBRATE/VALIDATE/MONITOR;
- límites CPU, presupuesto de tiempo y RSS;
- fuentes upstream y sus hashes;
- ausencia de GPU y de decisión científica automática.

Se crea un draw fresco con tres claves nuevas bajo una config nueva y rutas
nuevas. No se reutilizan el escrow, claves, manifest ni truth del intento
anterior. La semántica generativa y la población esperada permanecen iguales;
los conteos observados del nuevo draw deberán verificarse, no copiarse como
resultado desde el draw anterior.

## Contrato tipado de población fresca

La config sucesora agrega dentro de `fresh_benchmark` el campo exacto:

```text
pair_token_count_basis = eligible_unique_pair_tokens
```

El módulo valida que ese valor aparezca sólo en la identidad de config sucesora.
El preparador selecciona entonces `eligible_unique_pair_tokens` antes de comparar
contra `expected_eligible_pair_tokens_per_split=768`. La config histórica, si se
lee como artefacto, conserva su comportamiento y sus bytes; no se le injerta una
semántica nueva. El test acreditante materializa por split `1152` tokens totales
y `768` elegibles y exige que el primario sucesor acepte el segundo conteo sin
alterar ninguno de los otros valores poblacionales.

No se permite un selector libre ni otra cadena. Falta del campo, valor diferente,
uso bajo la config histórica o selección de `total_unique_pair_tokens` para la
sucesora causan rechazo antes de inferencia.

## Causa que debe cerrarse

`generation_receipt.json` incluye `execution_mode`; por diseño el primario y el
replay declaran modos distintos. Cada `preparation_receipt.json` enlaza el SHA
bruto de su propio generation receipt. Aunque `_normalize_operational()` elimina
el modo al comparar el receipt de generación, deja intacto el hash derivado al
comparar el receipt de preparación. Dos receipts semánticamente equivalentes
producen así hashes brutos diferentes y `compare_runs()` los convierte en una
divergencia global.

La corrección no puede consistir en agregar
`generation_receipt_sha256` al omit-set global. Ese gesto perdería contexto y
podría silenciar un enlace roto en otro artefacto. La normalización debe ser
específica del receipt de preparación y posterior a comprobar su integridad
local.

## Normalización tipada

El runner incorporará un helper cerrado para comparar cada par
`generation_receipt.json + preparation_receipt.json`:

1. cargar ambos artefactos públicos y exigir sus schemas/formas ya congelados;
2. verificar en cada raíz que
   `preparation_receipt.generation_receipt_sha256` sea exactamente el SHA-256
   bruto de su propio `generation_receipt.json`;
3. verificar la preparation attestation de cada paquete contra la clave pública
   confiada y reconstruir su payload desde bytes físicos;
4. normalizar por separado los generation receipts mediante las reglas
   operacionales existentes;
5. exigir igualdad de esos generation receipts normalizados;
6. normalizar cada preparation receipt y reemplazar únicamente el valor del
   hash derivado por una representación tipada de la semántica normalizada del
   generation receipt —un digest canónico común o un marcador cerrado—;
7. exigir igualdad de los preparation receipts resultantes.

La representación tipada sólo se construye después de validar el enlace bruto.
No se ignoran hashes ausentes, mal formados o apuntados a otro archivo. Campos
semánticos nuevos, diferencias no autorizadas en generation receipt o cualquier
otro desacuerdo de preparation receipt siguen causando rechazo.

`compare_runs()` conserva las categorías separadas:
`scientific_exact`, `scientific_array_exact`, `secret_sha256_exact`,
`functional_state_exact` y `operational_semantic`. El cambio alcanza sólo la
entrada operacional `preparation_receipt.json`. Las firmas se verifican por
paquete antes de comparar invariantes; nunca se comparan firmas como sustituto
de validar cada una.

## Unión cerrada de atestaciones de preparación

La atestación vigente nació para el recovery histórico y presupone amendment y
provenance. El protocolo sucesor agrega una segunda forma tipada, sin relajar la
primera:

- `recovery/replay` histórico: schema vigente, `recovery_provenance` obligatorio
  y record obligatorio de `recovery_amendment.json`;
- `primary/replay` sucesor fresco: schema nuevo, sin provenance, sin amendment y
  con identidad de config sucesora firmada.

Schema, modo y presencia de provenance/amendment forman una unión exclusiva.
Un paquete que mezcle schema fresco con amendment, schema recovery sin amendment,
modo `primary` con provenance o modo no permitido se rechaza. El payload fresco
continúa ligando pre-generation freeze, manifest, generation receipt,
preparation freeze, preparation receipt, config snapshot, source bindings,
journal, logits y bundles. El runner reconstruye cada record desde bytes físicos,
verifica cada firma y sólo después compara invariantes primary/replay.

El preparador y su test específico entran por ello en el alcance de
implementación. La regresión histórica debe demostrar que la forma recovery
anterior conserva aceptación y todos sus rechazos.

## Cobertura de pruebas previa al draw

La implementación debe agregar, al menos:

- aceptación de primary/replay cuyos generation receipts difieren sólo en
  `execution_mode` y cuyos preparation receipts enlazan correctamente cada
  hash bruto;
- rechazo si uno de los hashes brutos no corresponde a su receipt local;
- rechazo si los generation receipts difieren en un campo que no normaliza la
  política operacional;
- rechazo si otro campo del preparation receipt difiere;
- rechazo de claves adicionales, schemas o formas no canónicas;
- verificación real de ambas preparation attestations, sin monkeypatch en el
  caso integrado que acredita la corrección;
- integración `compare_runs()` con cadena `primary ↔ replay` realista y
  `all_exact=true` sólo después de pasar todas las categorías;
- preservación del test que detecta amendment distinto, arrays alterados,
  modelo no portable y artefactos científicos divergentes;
- regresión completa de Wave 56, 57, 58 y 59 afectada, con CUDA invisible y
  cuatro threads;
- prueba negativa que demuestre que el draw anterior conserva
  `replay_exact=PENDING` y no es mutado por el protocolo sucesor.
- primario fresco sucesor con `total=1152`, `eligible=768` aceptado por la base
  tipada, y rechazo del total como sustituto;
- firma real de paquetes sucesores `primary ↔ replay` sin amendment/provenance,
  más regresión separada `recovery ↔ replay`;
- aceptación individual de las dos rutas canónicas de config, rechazo de path
  que sólo comparta sufijo, rechazo si ambas configs figuran como self-source y
  rechazo si la config ejecutada no es la única self-source declarada;
- hashes públicos congelados del primario anterior y sus tres failure records
  comprobados antes y después de los tests sucesores.

Los tests sintéticos no pueden fabricar placeholders idénticos para ambos
receipts si con ello eliminan la diferencia de modo que causó el defecto real.
El test integrado debe materializar específicamente `primary` frente a `replay`,
calcular los dos hashes brutos distintos y conservar equivalencia semántica.
`recovery ↔ replay` permanece como regresión distinta y no puede acreditar el
modo nuevo.

## Config y rutas nuevas

La config canónica será:

```text
experiments/geometria_proporcional/configs/
  wave59_fresh_hgb_guard_bracket_replay_normalized.json
```

Las raíces serán:

```text
data/geometria_proporcional/
  wave59_fresh_hgb_guard_bracket_replay_normalized_v1/
  wave59_fresh_hgb_guard_bracket_replay_normalized_v1_replay/
```

La config conserva `schema_version=wave59-fresh-hgb-guard-bracket-v1` para
reutilizar el mismo contrato científico validado. `plan` y
`accepted_plan_audit` continúan ligados al plan científico original y R420,
cuyas constantes valida el módulo Wave 59. Esta revisión operacional y su
auditoría se agregan como execution sources adicionales, no reemplazan ni
reescriben ese plan.

`implementation_binding` apuntará al commit que modifique únicamente:

- `src/geometria_proporcional/wave59_hgb_guard_bracket.py`;
- `experiments/geometria_proporcional/prepare_wave56_fresh.py`;
- `experiments/geometria_proporcional/run_wave59_hgb_guard_bracket.py`;
- `tests/test_wave59_prospective.py`;
- `tests/test_wave59_preoracle_recovery.py`.

El módulo de contrato debe reemplazar el sufijo único
`CONFIG_SOURCE_SUFFIX` por una allowlist de rutas relativas exactas que contenga
la config original y la sucesora. Tanto `validate_pre_draw_config()` como
`validate_execution_bindings()` deben aceptar sólo pertenencia exacta a esa
allowlist. Un path arbitrario que meramente termine en el nombre legacy sigue
rechazado. Este cambio no altera features, modelos ni criterios; hace alcanzable
la nueva identidad de config sin relajar su canonicalidad.

Su auditoría será una fuente requerida y contendrá el commit, hashes de los cinco
archivos, evidencia de tests y un encabezado `## Dictamen: PASS` consumible por
el validator existente. La config nueva reemplaza en su source map la config
anterior y la auditoría R426 por sus equivalentes sucesores; agrega este plan,
su auditoría y `tests/test_wave59_preoracle_recovery.py`, que pasa a ser fuente
porque su blob forma parte del cierre de compatibilidad. El resto de fuentes
debe coincidir con los blobs de HEAD. El self-binding de config se calcula con
su propio digest normalizado a ceros.

### Delta cerrado respecto de la config original

Un validator reproducible carga la config original y la sucesora y rechaza toda
diferencia fuera de esta allowlist:

- `primary_output` y `primary_output_name`;
- `replay_output` y `replay_output_name`;
- reemplazo del único path de config en `required_execution_sources` y
  `source_sha256`;
- reemplazo de R426 por la nueva auditoría de implementación;
- agregado de este plan y su auditoría como execution sources;
- agregado de `tests/test_wave59_preoracle_recovery.py` como execution source;
- `implementation_binding` completo;
- bloque cerrado `successor_authority`, con plan, auditoría de plan y path
  predeclarado de auditoría final;
- `fresh_benchmark.pair_token_count_basis=eligible_unique_pair_tokens`.

Todo otro campo, incluida la totalidad de `pattern`, features, targets, modelos,
factorial, controles, cuantiles, mínimos, bootstrap, shards, presupuestos,
referencias, source bindings upstream y artifact classes, debe ser canónicamente
igual a la config original. El validator exige además que exista exactamente un
self-source de config y que coincida con el path efectivamente ejecutado.

### Auditoría final como autoridad material

`successor_authority.final_config_audit_path` predeclara el informe final, pero
no su hash futuro. Antes de crear escrow, tanto preparador como runner exigen:

- que la config haya sido introducida por un commit que modifica sólo su path;
- que el informe haya sido introducido por el commit HEAD;
- que HEAD modifique sólo ese informe y sea hijo directo del commit de config;
- que el bloque canónico del informe nombre commit de config, SHA-256 bruto de
  config y `PASS` una sola vez;
- que el worktree esté globalmente limpio;
- que no exista ningún commit posterior.

El informe final no integra `required_execution_sources`, evitando circularidad;
su autoridad proviene de la relación dinámica config commit → informe/HEAD. El
validator corre antes de cualquier output, escrow o acceso a datos del draw.

## Cadena de autoridad

La secuencia previa a cualquier escrow nuevo es lineal:

1. este plan y ningún otro path;
2. auditoría independiente del plan y ningún otro path;
3. implementación en módulo de contrato, preparador, runner, test prospectivo y
   test de preparación/recovery, exactamente esos cinco paths;
4. auditoría independiente de implementación y ningún otro path;
5. config congelada nueva, con paths, source map, implementación y auditorías;
6. auditoría final independiente de config/cadena y ningún otro path;
7. preflight completo en HEAD limpio;
8. draw primario fresco;
9. replay desde el escrow primario y comparación exacta;
10. fases analíticas primary/replay y auditoría de artefactos.

Ningún commit posterior a la auditoría final puede existir antes de las corridas.
El preflight debe comprobar que cada archivo es el blob de HEAD, que el
implementation commit es ancestral, que su auditoría contiene
`## Dictamen: PASS`, que el config self-binding cierra, que el delta contra la
config original pertenece a la allowlist y que la auditoría final es HEAD
exclusivo e hijo directo de la config.

## Preservación del antecedente

Se conservan sin edición:

- `wave59_fresh_hgb_guard_bracket_v1/` con primario `COMPLETE` y condiciones
  de replay pendientes;
- `wave59_fresh_hgb_guard_bracket_v1_replay.failed_20260905T102929791142Z/`
  con `last_state=COMPLETE` y failure records firmados;
- R445 como auditoría causal y delimitación de claims.

Los tests y el preflight sucesor fijan como sentinels públicos del antecedente:

| Artefacto | SHA-256 |
|---|---|
| primario `analysis.json` | `79a4c1eca78497d9fd6ea49177508ac9f879cabe0824b566b437d8b9f8e7eb96` |
| primario `artifact_manifest.json` | `ce30675744b27d656ff7eb177145ae5fac77d9f092ccc483d329425864185770` |
| replay fallido `FAILURE.json` | `4967c6108ffd07e1b8cf6c0f301e702722be592f0d678594dd89e659053aa3c4` |
| replay fallido `failure_inventory.json` | `353afe6f61c476a82d5c061fadcdfa13bad5c27787629189cb80094d4769b9a9` |
| replay fallido `failure_attestation.json` | `8cb19ce7d741a0caaa73c88d44c96fe10557dbba6283290f90e98bbec0cb93e3` |

El sentinel verifica además `scientific_decision=null`, ambos
`replay_exact=PENDING` y ambos `aggregate_with_replay=null`. Un cambio en
cualquiera de estos bytes o estados aborta el nuevo preflight; no se regenera ni
se “corrige” el antecedente.

El antecedente puede usarse como diagnóstico post hoc de equivalencia, nunca
como adjudicación prospectiva ni como fuente de claves del nuevo draw.

## Presupuesto y dispositivo

Todo el protocolo es CPU-native. La preparación observada demoró alrededor de
dos minutos por paquete y la fase analítica menos de medio minuto por paquete;
la regresión amplia duró cerca de siete minutos. No se justifica GPU: HGB de
scikit-learn es CPU y migrarlo cambiaría la arquitectura comparada.

Si una contingencia futura exige CUDA para avanzar con eficiencia material, el
trabajo se detendrá antes de usarla, se preservará el estado y se avisará a
Mariano por Telegram con objetivo, duración y VRAM estimadas. Colab no forma
parte del protocolo.

## Criterio de cierre

El protocolo sucesor puede producir evidencia sólo si la auditoría final de la
config pasa, primary y replay completan sus máquinas de estados, las firmas se
verifican individualmente y `compare_runs()` devuelve `all_exact=true` sin
excepciones post hoc. Si vuelve a fallar, el intento se archiva y el alcance se
reporta tal cual es. Ninguna condición de este plan declara `GO/NO-GO`.

## Resolución de R446

R446 emitió `REVISE` porque el primer diseño alcanzaba el comparador pero no el
fresh primary: conservaba la comparación legacy del total contra `768`, la
atestación exigía recovery provenance/amendment y la auditoría final no era una
condición material. También faltaban un delta mecánico cerrado entre configs y
un test acreditante inequívocamente `primary ↔ replay`.

Esta revisión incorpora los cinco cierres. El conteo se tipa en la config
sucesora y sólo ella usa `eligible_unique_pair_tokens`; la atestación se convierte
en una unión exclusiva recovery/fresh; el preflight autentica config commit,
informe final y HEAD; el delta permitido se enumera campo por campo; y la prueba
integrada reproduce el modo fresco real con firmas, mientras recovery queda como
regresión separada. El alcance de implementación pasa a cinco paths y el nuevo
tramo de autoridad debe comenzar con una auditoría independiente de esta
revisión, hija del commit exclusivo de R446.
