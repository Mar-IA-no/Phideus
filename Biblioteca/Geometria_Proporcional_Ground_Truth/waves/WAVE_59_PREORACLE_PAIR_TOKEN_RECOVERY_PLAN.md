# Ola 59 — plan de recuperación pre-oracle del draw HGB

> **Estado:** `R430-SUBSTANTIVE-PASS / FOR-CANONICAL-ATTESTATION / PRE-IMPLEMENTATION / PRE-RECOVERY / SAME-ESCROW / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-05
> **Draw de origen:** `wave59_fresh_hgb_guard_bracket_v1.failed_20260905T071003529517Z`
> **Contrato científico:** `WAVE_59_FRESH_HGB_GUARD_BRACKET_PLAN.md`

## Situación

La preparación primaria de Ola 59 creó las tres claves, fijó el contrato
pre-generación y produjo el benchmark fresco completo. Se detuvo después de
sellar la realización, pero antes de inferencia, materialización del oracle
analítico, labels autorizados, bundles preparados o fases HGB. El registro
público del fallo declara `last_state=null`,
`maximum_truth_materialized=none` y `recovery_context=false`.

La causa no es una realización inválida. El config congeló
`expected_eligible_pair_tokens_per_split=768`, pero la ruta legacy de una
ejecución primaria compara ese campo contra `total_unique_pair_tokens`. La ley
generativa congelada determina una población esperada de `1152` tokens totales
y `768` elegibles; esos conteos son todavía un contrato a verificar sobre el
origen sellado, no una observación semántica anticipada. La ruta ejecutada
comparó el total contra el esperado elegible y produjo:

```text
RuntimeError: fresh benchmark pair-token count differs from prospective freeze
```

Cambiar el esperado a `1152` sería incorrecto: convertiría un contrato sobre
la población elegible en otro que incluye tokens fuera de catálogo. Redibujar
también sería incorrecto porque ya existe un escrow durable. La continuación
admisible es recuperar exactamente esa realización, con las mismas claves y el
mismo manifest, bajo una autoridad tipada y auditada antes de abrir el escrow.

## Evidencia durable del origen

El intento fallido permanece en:

```text
data/geometria_proporcional/
  wave59_fresh_hgb_guard_bracket_v1.failed_20260905T071003529517Z/
```

La identidad observada antes de este plan es:

| Artefacto | SHA-256 |
|---|---|
| contrato público embebido | `595ae8f819f2814aa51166f505a180ea4c5b984f452397f24f49887fb65a0e63` |
| `generation_escrow.json` | `94c311ef5630444d2d8f2b83c37d7a3278536a456428d40b8c45f69403bde89d` |
| `pre_generation_freeze.json` | `55d31e7c03f55020c94f50feb7c8a3f883eee4d8afbfb85029de852dec947aee` |
| `FAILURE.json` | `0bbab13a4a3930d99307a4437ccb13336048771392944f6772a47f71e7a286b6` |
| `failure_inventory.json` | `031f3a58fe761c9274397864ddb1b425442b3845e18ece2308669281bf90e3d3` |
| `failure_attestation.json` | `d121a8ffb5dc3c4b596d8dee4e746e466d1180ce5145ab0f11070fc0dbd25cf1` |
| `benchmark/manifest.json` | `88ed16832b8e11821b02af0822830807c04bc1cb6cd697cf631a2d092d30642d` |

El contrato nació en
`1ef98a109413b55d8dcf4a5df2171d7017c7cd57`. La atestación de fallo enlaza
los hashes de `FAILURE.json` y `failure_inventory.json` y está firmada con la
clave pública congelada. El árbol físico contiene exactamente `26` entradas:
`6` directorios y `20` archivos, todos `root:root`; los directorios son modo
`0700`, `pre_generation_freeze.json` es `0644` y los demás archivos son
`0600`. El amendment deberá incluir ese inventario completo: path, tipo, modo,
uid y gid para todas las entradas, y además tamaño y SHA-256 para cada archivo
regular. Los directorios no reciben un tamaño ni un hash de contenido
artificial.

El inventario de fallo clasifica como `extra` los diecisiete artefactos que la
preparación alcanzó a publicar antes del primer journal. Esa palabra describe
la cobertura del archivador, no material científico imprevisto: son escrow,
freeze y paquete generativo. El origen no contiene `inference/`,
`authorized_labels/`, `prepared/`, `bundles/`, `phases/`, `journals/`,
`preparation_freeze.json`, `generation_receipt.json` ni oracle analítico
materializado. Sí contiene, como debe hacerlo un draw ya generado, truth
sellada y secret files bajo `benchmark/sealed/`. `maximum_truth_materialized`
se interpreta en el sentido analítico del runner Wave 59; no niega la truth
sellada que preserva el generador.

## Invariantes científicos y operativos

La recuperación no cambia el plan, el config, los seeds, splits, features,
targets, modelos HGB, hiperparámetros, controles, brackets, bootstrap,
criterios, presupuestos ni fuentes upstream. En particular:

- `768` conserva el significado de pair tokens elegibles por split;
- la elegibilidad continúa siendo `is_out_of_catalog=false` y
  `calibration_population=canonical_preserving`;
- primero se filtran filas elegibles y luego se deduplican `pair_token`;
- las tres claves son exactamente las del escrow fallido;
- el benchmark regenerado reproduce el manifest
  `88ed1683...d30642d` byte por byte;
- la preparación y el replay conservan el mismo recovery amendment;
- ninguna fase analítica puede adelantarse durante la autenticación;
- no se modifica el config congelado ni se reescribe su mapa de 33 fuentes;
- `GO/NO-GO` y promoción arquitectónica continúan reservados a Mariano.

La rama de recovery ya calcula el conteo contractual con
`eligible_unique_pair_tokens`. No se agregará una excepción numérica ni se
alterará esa operación. El cambio se limita a reconocer una autoridad Wave 59
que haga alcanzable, con el mismo escrow, la semántica ya congelada.

## Autoridad en dos etapas

La recuperación hereda la separación validada en Ola 57, pero no reutiliza a
ciegas su schema ni sus supuestos físicos. Wave 59 posee dos registros públicos
adicionales —inventario y atestación de fallo—, otra topología y otro contrato
de fuentes. Debe tener dispatcher y validator propios.

### Etapa 1: autenticación content-blind

Antes de extraer claves, el validator puede interpretar solamente:

- `FAILURE.json`;
- `failure_inventory.json`;
- `failure_attestation.json`;
- `pre_generation_freeze.json`, declarado secret-free;
- `benchmark/manifest.json`;
- `benchmark/attestations/semantic_root.json`;
- los archivos visibles requeridos por `validate_visible_package()`.

Sobre `generation_escrow.json`, los tres secret files, los cuatro JSONL de
truth sellada y `benchmark/commitments/semantic.jsonl`, sólo puede hacer
`lstat` y hashing binario opaco mediante
`O_NOFOLLOW -> fstat -> streaming SHA-256`. En esta etapa se prohíbe llamar a
`read_escrow()`, `keys_from_escrow()`, `validate_semantic_attestation()`,
`sealed_population_counts()`, `read_jsonl`, parsers JSON, `bytes.fromhex` o
cualquier consumidor semántico sobre esos paths.

El preflight content-blind debe cerrar, en este orden lógico:

1. ruta canónica, schema, estado y forma exacta del amendment;
2. cadena Git lineal, paths exclusivos, hashes, HEAD exacto y worktree limpio;
3. inventario físico de `26` entradas, ownership, modos, tamaños y hashes;
4. identidad exacta del fallo y ausencia de estados analíticos posteriores;
5. consistencia de `failure_inventory.json` con el árbol físico y de su
   atestación detached con la clave pública congelada;
6. contrato público, commitments, manifest y atestación semántica detached;
7. delta de contrato y fuentes estrictamente limitado por el amendment.

El hashing de archivos sensibles verifica identidad sin materializar su
significado. Un solo byte, modo, owner, path o entrada adicional diferente
invalida el origen.

### Etapa 2: validación semántica autorizada

Sólo después del PASS completo de la etapa 1 puede comenzar la apertura
semántica. Inmediatamente antes del primer parseo, la etapa 2 debe volver a
ejecutar el inventario content-blind completo y exigir igualdad exacta con el
snapshot autorizado. Si un path, tipo, modo, owner, tamaño o hash cambió entre
etapas, se rechaza antes de llamar a cualquier parser. Una vez cerrada esa
continuidad de identidad, se comprueba que:

- escrow y freeze público son equivalentes;
- claves y commitments coinciden;
- manifest, firma semántica y truth sellada son internamente consistentes;
- cada split `train`, `val` y `lockbox` tiene `4992` filas;
- cada uno tiene `1152` tokens totales, `768` elegibles, `384` fuera de
  catálogo, `192` no canónicos y una intersección elegible/no canónica de
  `192` tokens.

Esta validación no autoriza fit, selección, consulta anticipada del monitor ni
materialización de labels fuera de la máquina de estados. Autoriza únicamente
la reutilización de una realización ya autenticada.

## Schema y rutas canónicas

Para `prospective_config.schema_version=wave59-fresh-hgb-guard-bracket-v1`, el
preparador aceptará exclusivamente:

```text
schema: wave59-hgb-guard-bracket-preoracle-recovery-amendment-v1
path:   experiments/geometria_proporcional/configs/
        wave59_preoracle_pair_token_recovery_amendment_v1.json
```

La cobertura específica se agregará en:

```text
tests/test_wave59_preoracle_recovery.py
```

Además se adaptará `tests/test_wave59_prospective.py`, porque su fixture físico
ejecuta hoy un paquete sintético no-recovery contra los hashes originales. Una
vez que preparador y runner cambien, conservar ese test byte-exacto haría
imposible ejecutar la regresión sin debilitar la producción. El test existente
pasará a declarar una provenance de recovery sintética autenticada y mantendrá
casos separados donde un paquete normal rechaza los hashes nuevos.

La prueba sintética debe ejercer la misma función pública de autoridad sobre un
repositorio Git temporal con cadena, amendment, freeze y provenance completos.
No se permite detección de pytest, bypass de producción ni una bandera para
omitir source binding. Si una parte del fixture analítico desacopla localmente
una revalidación ya ejercida, el reemplazo queda limitado al test y acompañado
por pruebas directas de aceptación/rechazo sobre la implementación real.

El test nuevo no se incorpora retroactivamente al mapa congelado de fuentes de
ejecución. El test prospectivo existente sí pertenece a ese mapa y su cambio
queda declarado como tercer delta old/new. Ambos tests autentican el mecanismo
de recovery mediante la cadena de commits y auditorías; ninguno se presenta
como fuente que hubiera regido el sorteo original.

## Delta de implementación permitido

La implementación puede modificar sólo:

- `experiments/geometria_proporcional/prepare_wave56_fresh.py`;
- `experiments/geometria_proporcional/run_wave59_hgb_guard_bracket.py`;
- `tests/test_wave59_prospective.py`;
- `tests/test_wave59_preoracle_recovery.py`.

Dentro del contrato de 33 fuentes, sólo preparador, runner y el test prospectivo
existente pueden cambiar de hash. El test nuevo no pertenece a ese conjunto.
Los hashes originales que el amendment debe declarar son:

| Fuente | SHA-256 original |
|---|---|
| preparador | `fb5345dd978a3bd2d658a8709f874e6ff3944f6f88e62f0330c9b7d9946c9778` |
| runner | `d78a414de79eea4ae51b98913de37b0c83a579b509574386f71d5060e5b9b816` |
| test prospectivo | `8121b00a8e2e87785ab9fb17859e902f6f336f32a7179bb3517509b00bff45e4` |

Las otras `30` fuentes deben coincidir exactamente con el config. El contrato
de ejecución debe conservar el mismo conjunto de campos y el mismo contenido
que el contrato público de origen salvo `git_commit` y esos tres hashes. No se
permiten deltas en config, source bindings, upstreams ni historical preflight.

### Preflight del preparador

`preparation_preflight()` actualmente compara las 33 fuentes contra el config
antes de construir el contrato de ejecución. La ruta Wave 59 recuperada podrá
aceptar provisionalmente los tres hashes nuevos sólo si el argumento apunta al
amendment canónico y sus deltas old/new coinciden con config, archivos físicos
y HEAD. Esa concesión no abre escrow ni habilita output: permite terminar el
preflight histórico y construir el contrato completo. Antes de extraer claves,
el validator integral vuelve a autenticar cadena, origen y contrato en la
etapa content-blind. Una autoridad parcial o falsa sólo puede causar rechazo.

Las rutas Wave 56 y Wave 57 conservan sus validators actuales. Una ejecución
Wave 59 primaria sin recovery amendment sigue exigiendo los 33 hashes
originales y continúa prohibiendo un segundo draw.

### Continuidad del runner analítico

El runner también autentica las fuentes antes de cada fase. En un paquete
recuperado puede aceptar los tres hashes nuevos solamente cuando:

- existe `recovery_amendment.json` dentro del paquete preparado;
- su SHA coincide con el amendment canónico versionado en el repositorio;
- `preparation_freeze.json` y `recovery_provenance` lo enlazan por hash;
- config, hashes old/new, implementación, auditorías, HEAD y limpieza pasan la
  misma autoridad pública;
- las 30 fuentes restantes conservan los hashes congelados.

El runner no abre escrow ni truth para autorizar fuentes. En paquetes no
recuperados, si falta cualquiera de esos enlaces o si aparece una fuente
distinta de las tres autorizadas, se mantiene el rechazo estricto
actual. Cada revalidación
entre fases vuelve a comprobar la misma autoridad.

## Cadena de commits cerrada

La autoridad se construye con seis commits lineales y sin paths mezclados:

1. este plan y ningún otro archivo;
2. auditoría independiente del plan y ningún otro archivo;
3. implementación en los cuatro paths predeclarados;
4. auditoría independiente de implementación y ningún otro archivo;
5. amendment canónico ya poblado con todos los hashes y commits observables;
6. auditoría final independiente del paquete y ningún otro archivo.

El commit 6 debe ser HEAD exacto y el worktree debe estar globalmente limpio
al iniciar tanto recovery como replay. El validator debe comprobar direct
parents, introduction commits, blobs, paths cambiados y un bloque de dictamen
parseable por informe. La auditoría de plan liga commit y SHA del plan; la de
implementación liga commit y hashes nuevos de preparador, runner, test
prospectivo y test específico; la final liga commit y SHA del amendment. Un
texto que contenga `PASS` fuera del bloque esperado, un segundo dictamen
contradictorio o una auditoría no ancestral no confiere autoridad.

## Pruebas exigidas

La implementación debe demostrar, al menos:

- dispatch separado Wave 56/Wave 57/Wave 59;
- rechazo de schema Wave 59 bajo otra ruta o de otro schema en la ruta Wave 59;
- rechazo de estado, assertions, población o forma alterados;
- rechazo de cualquier inventario, hash, tamaño, modo, owner o path distinto;
- verificación de `FAILURE`, inventario y atestación de fallo Wave 59;
- rechazo si aparece inferencia, journal, bundle, label u oracle analítico;
- spies que permiten sólo hashing binario opaco sobre material sensible y
  fallan si el preflight intenta parsearlo, extraer claves o contar truth;
- primer acceso semántico posterior al cierre total de autoridad content-blind;
- sustitución de un archivo sensible entre etapas rechazada por un segundo
  inventario opaco antes del primer parseo semántico;
- rechazo de un delta en config, bindings, upstream, historical preflight,
  conjunto de fuentes o cualquiera de las otras 30 fuentes;
- rechazo de hashes old/new falsos, commits no lineales, mixed commits,
  auditorías ausentes o contradictorias, HEAD posterior y worktree sucio;
- preflight provisional incapaz de habilitar claves u output por sí solo;
- ejecución primaria normal todavía ligada a los 33 hashes congelados;
- runner recuperado ligado simultáneamente a config, amendment copiado,
  provenance, preparation freeze y HEAD;
- runner no recuperado incapaz de aceptar los hashes nuevos;
- `secrets.token_bytes` imposible de invocar en recovery o replay;
- regeneración con mismas claves y manifest byte-exacto;
- aceptación por `eligible_unique_pair_tokens=768`, conservando visibles los
  otros conteos estructurales;
- copia exacta del amendment en primario y replay;
- replay exacto de escrow, freeze, benchmark, logits, bundles y provenance;
- regresión completa de las suites Wave 56, 57, 58 y 59 afectadas.

Los fixtures sintéticos pueden reducir datos, pero debe existir un probe final
contra el origen real que ejecute sólo la etapa content-blind y pruebe que el
escrow, secrets, commitments y truth sellada fueron tratados como bytes
opacos. Ese probe no inicia inferencia, no abre oracle y no produce labels.

## Ejecución autorizada después de la auditoría final

El primario canónico se reconstruirá con:

```text
--recovery-secrets-from \
  data/geometria_proporcional/
  wave59_fresh_hgb_guard_bracket_v1.failed_20260905T071003529517Z
--recovery-amendment \
  experiments/geometria_proporcional/configs/
  wave59_preoracle_pair_token_recovery_amendment_v1.json
```

Tras la autoridad content-blind, el preparador abre el escrow, valida
semánticamente el origen, vuelve a generar con las mismas claves, exige el
manifest `88ed1683...d30642d`, revalida el origen y recién entonces comprueba
los `768` tokens elegibles. Luego continúa con inferencia ciega y preparación
de bundles según la máquina Wave 59.

El replay usa el primario canónico recuperado como fuente de escrow y
referencia, el mismo amendment y un output separado. Debe ser exacto antes de
iniciar la secuencia analítica. Las fases HGB se ejecutan después sobre ambos
paquetes sin alterar código, config ni autoridad.

Si falla cualquier check, el nuevo intento se archiva con su estado físico y
no habilita otro draw. Un defecto de implementación se registra como tal; no se
convierte en observación científica.

## Presupuesto operativo

Diseño, auditorías, tests, regeneración, inferencia congelada y HGB son CPU.
El HGB de scikit-learn es CPU-native; trasladarlo a GPU cambiaría la
arquitectura en lugar de acelerar de manera equivalente este protocolo. Las
corridas observadas relevantes duran minutos, no horas, por lo que aquí no se
justifica solicitar GPU ni sustituir una etapa GPU breve por una carga CPU
desproporcionada.

Si una contingencia futura vuelve materialmente más eficiente una ejecución
CUDA, el ciclo se detendrá antes de usarla. Se preservará el estado, se
publicará evidencia durable y se avisará a Mariano por Telegram con objetivo,
duración y VRAM estimadas para esperar habilitación explícita. Colab no forma
parte de este flujo.

## Resolución de R429

R429 emitió `REVISE` con dos findings P1 y dos P2. Esta revisión corrige el
schema de dispatch a `wave59-fresh-hgb-guard-bracket-v1`; elimina la
contradicción de regresión autorizando y trazando el tercer delta del test
prospectivo existente, además del test nuevo; reduce a `30` el conjunto de
fuentes necesariamente invariantes; precisa que tamaño y hash corresponden a
archivos regulares; y exige un segundo inventario content-blind inmediatamente
antes de cualquier parseo semántico, con prueba adversarial TOCTOU. La ruta
productiva conserva rechazo estricto fuera de una recovery autenticada.

R430 confirmó sustantivamente el cierre de los cuatro findings, pero su salida
omitió el encabezado Markdown requerido por el parser de autoridad. El informe
se preserva sin retoques y no se usará como atestación ejecutable. Esta versión
se somete a una nueva auditoría cuyo reporte debe ser canónico además de
sustantivamente independiente.
