# Ola 59 — protocolo sucesor con normalización tipada del replay

> **Estado:** `PRE-IMPLEMENTATION / NEW-PROTOCOL / NEW-DRAW / CPU-ONLY / NO-GO-NOGO`
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

La nueva ejecución conserva el contrato científico y cambia solamente la
semántica previamente declarada del checker de replay. El resultado seguirá
separando observación, interpretación y decisión; `GO/NO-GO` y promoción de
arquitectura pertenecen a Mariano.

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
- integración `compare_runs()` con cadena primary/replay realista y
  `all_exact=true` sólo después de pasar todas las categorías;
- preservación del test que detecta amendment distinto, arrays alterados,
  modelo no portable y artefactos científicos divergentes;
- regresión completa de Wave 56, 57, 58 y 59 afectada, con CUDA invisible y
  cuatro threads;
- prueba negativa que demuestre que el draw anterior conserva
  `replay_exact=PENDING` y no es mutado por el protocolo sucesor.

Los tests sintéticos no pueden fabricar placeholders idénticos para ambos
receipts si con ello eliminan la diferencia de modo que causó el defecto real.
El test integrado debe materializar `recovery` o `primary` frente a `replay`,
calcular los dos hashes brutos distintos y conservar equivalencia semántica.

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
- `experiments/geometria_proporcional/run_wave59_hgb_guard_bracket.py`;
- `tests/test_wave59_prospective.py`.

El módulo de contrato debe reemplazar el sufijo único
`CONFIG_SOURCE_SUFFIX` por una allowlist de rutas relativas exactas que contenga
la config original y la sucesora. Tanto `validate_pre_draw_config()` como
`validate_execution_bindings()` deben aceptar sólo pertenencia exacta a esa
allowlist. Un path arbitrario que meramente termine en el nombre legacy sigue
rechazado. Este cambio no altera features, modelos ni criterios; hace alcanzable
la nueva identidad de config sin relajar su canonicalidad.

Su auditoría será una fuente requerida y contendrá el commit, hashes de los tres
archivos, evidencia de tests y un encabezado `## Dictamen: PASS` consumible por
el validator existente. La config nueva reemplaza en su source map la config
anterior y la auditoría R426 por sus equivalentes sucesores; agrega este plan y
su auditoría. El resto de fuentes debe coincidir con los blobs de HEAD. El
self-binding de config se calcula con su propio digest normalizado a ceros.

## Cadena de autoridad

La secuencia previa a cualquier escrow nuevo es lineal:

1. este plan y ningún otro path;
2. auditoría independiente del plan y ningún otro path;
3. implementación en módulo de contrato, runner y test prospectivo,
   exactamente esos tres paths;
4. auditoría independiente de implementación y ningún otro path;
5. config congelada nueva, con paths, source map, implementación y auditorías;
6. auditoría final independiente de config/cadena y ningún otro path;
7. preflight completo en HEAD limpio;
8. draw primario fresco;
9. replay desde el escrow primario y comparación exacta;
10. fases analíticas primary/replay y auditoría de artefactos.

Ningún commit posterior a la auditoría final puede modificar una execution
source antes de las corridas. El preflight debe comprobar que cada archivo es
el blob de HEAD, que el implementation commit es ancestral, que su auditoría
contiene `## Dictamen: PASS` y que el config self-binding cierra.

## Preservación del antecedente

Se conservan sin edición:

- `wave59_fresh_hgb_guard_bracket_v1/` con primario `COMPLETE` y condiciones
  de replay pendientes;
- `wave59_fresh_hgb_guard_bracket_v1_replay.failed_20260905T102929791142Z/`
  con `last_state=COMPLETE` y failure records firmados;
- R445 como auditoría causal y delimitación de claims.

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
