# R400 — Reauditoría focal del plan de recuperación pre-oracle de Ola 57

**Plan commit:** `834639ef71a6d9fd543c4fedeff8f773b2713868`
**Plan SHA-256:** `5d9a4e1e305cef4b1184bcef19411f054264f72ef760dcea02cd419ba64a2967`
**Result:** `REVISE`

## Dictamen ejecutivo

La revisión cierra correctamente el segundo finding de R399: reconoce que el
origen contiene truth sellada y reemplaza las aserciones ambiguas por
`no_materialized_oracle_in_origin` y `no_authorized_labels_in_origin`. También
introduce la separación arquitectónica correcta entre un preflight de autoridad
content-blind y una validación semántica posterior ya autorizada.

El primer finding, sin embargo, no está cerrado de forma implementable. El mismo
apartado autoriza calcular el SHA-256 de escrow, secret files y truth sellada
mediante flujo binario opaco, pero prohíbe literalmente abrir esos archivos y
exige spies que fallen ante cualquier intento de apertura. Calcular esos hashes
requiere abrir y leer sus bytes. La intención correcta puede inferirse, pero el
contrato de tests admite dos implementaciones incompatibles: una que omite
hashes obligatorios para satisfacer la prohibición literal y otra que abre los
archivos para hashearlos pero viola los negativos tal como están redactados.

Por afectar exactamente la frontera pre-key que autentica el recovery, esta
contradicción es material y bloquea `PASS`. La corrección es textual y acotada;
no exige ampliar archivos, cambiar config ni alterar el diseño en dos etapas.

## Finding priorizado

### ALTO — “Hash opaco permitido” y “ninguna apertura” son requisitos incompatibles

**Observación.** El plan permite al preflight “calcular hashes de bytes sin
interpretarlos” y aclara que el SHA-256 de archivos sensibles puede comprobarse
como flujo opaco
(`WAVE_57_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:94-108`). En el mismo bloque
prohíbe “abrir o parsear” cualquier `benchmark/sealed/*_secret.json` y ordena que
los tests fallen si se invoca una función de apertura
(`WAVE_57_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:99-110`). La contradicción se
repite en tres lugares:

- el check 7 exige validar hashes e inventario “sin abrir escrow, secret files
  ni truth sellada”
  (`WAVE_57_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:134-160`);
- la matriz de tests falla si se “intenta abrir” un `*_secret.json`
  (`WAVE_57_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:191-208`);
- el probe real debe demostrar “ausencia de apertura” del escrow, secret files y
  truth sellada
  (`WAVE_57_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:215-219`).

Un SHA-256 íntegro no puede calcularse sin abrir el archivo y consumir sus
bytes. El helper físico vigente hace precisamente esa lectura content-blind:
abre con `O_RDONLY | O_NOFOLLOW`, verifica identidad con `fstat` y alimenta un
hasher por bloques, sin parsear el contenido
(`experiments/geometria_proporcional/prepare_wave56_fresh.py:278-321`). Por lo
tanto, “apertura” no puede ser simultáneamente el evento prohibido y el mecanismo
obligatorio de hash.

**Impacto.** No existe un oracle de test inequívoco para el límite de acceso. Un
spy sobre `open`, `Path.open` u `os.open` rechazará también el camino seguro de
hash; un spy limitado a los parsers sí permitirá ese camino, pero ya no prueba
la prohibición literal escrita en el plan. La implementación podría quedar
formalmente no conforme aun respetando la seguridad, o debilitar inventario y
hashes para satisfacer el test.

**Corrección obligatoria.** El plan debe definir la distinción operacional, no
sólo la intención:

1. permitir exclusivamente `lstat` más apertura `O_NOFOLLOW` y streaming
   binario hacia SHA-256 para escrow, secret files y JSONL sellados;
2. prohibir antes del gate todo `read_text`, `json.load`/`json.loads` sobre esos
   contenidos, `read_jsonl`, conversión `bytes.fromhex`, `read_escrow`,
   `keys_from_escrow`, `validate_semantic_attestation`,
   `sealed_population_counts` o equivalente semántico;
3. exigir que los spies distingan y permitan el lector opaco de hash, mientras
   fallan ante las rutas semánticas anteriores;
4. reemplazar en los checks 7, tests, probe y resolución de R399 “sin abrir” o
   “ausencia de apertura” por “sin parsear, interpretar ni extraer”, dejando
   explícita la única lectura binaria autorizada.

Costo: bajo. Efecto: convierte la frontera content-blind en una propiedad
testeable sin sacrificar la autenticación criptográfica del inventario.

## Cierre focal de R399

### Finding alto R399 — PARCIALMENTE CERRADO

La revisión identifica correctamente el problema del helper histórico y
prohíbe durante el preflight `read_escrow`, `keys_from_escrow`,
`validate_semantic_attestation` y `sealed_population_counts`
(`WAVE_57_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:86-120,203-207`). Sitúa el primer
acceso semántico sólo después de autenticar DAG, HEAD, worktree, contrato,
inventario, hashes, manifest y firma detached
(`WAVE_57_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:122-160`). Esa arquitectura
cierra el defecto conceptual de R399.

No queda cerrado el contrato operativo de apertura versus hash descrito arriba.
Hasta resolverlo, la suite no puede demostrar a la vez integridad del árbol y
ausencia de acceso semántico con una expectativa única.

### Finding medio R399 — CERRADO

El plan declara ahora la presencia esperada de truth sellada y sus secret files,
y limita las ausencias al oracle analítico materializado y a labels autorizados
(`WAVE_57_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:52-61`). El schema del amendment
usa exactamente `no_materialized_oracle_in_origin` y
`no_authorized_labels_in_origin`
(`WAVE_57_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:134-142`). La matriz de tests y
el probe repiten esas semánticas (`:191-219`). No queda ambigüedad material sobre
la existencia de truth sellada frente a la ausencia de sus derivados
analíticos.

## Regresiones buscadas y no encontradas

- **Identidad Git.** El commit auditado es HEAD, modifica únicamente el plan y
  desciende directamente del commit que agregó R399. El SHA-256 coincide con el
  bloque inicial de este informe.
- **No-redraw.** Se mantienen el mismo escrow, las mismas tres claves, el mismo
  manifest y la imposibilidad de invocar `secrets.token_bytes` en recovery
  (`WAVE_57_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:63-84,208-213,221-245`).
- **Poblaciones.** Permanecen `4992` filas, `1152` tokens totales y `768`
  elegibles por split, con filtrado por fila previo a deduplicación; la revisión
  no toca el guard ni relaja mínimos (`:8-31,63-84,209-211`).
- **Source contract.** La implementación sigue limitada al preparador compartido
  y `tests/test_wave57_prospective.py`; el delta conserva config, bindings,
  inventario de fuentes y todos los campos salvo `git_commit` y esos dos hashes
  (`:150-153,184-213`). Los blobs de preparador, test y config no cambiaron en
  esta revisión.
- **Cadena de autoridad.** Se conserva la DAG lineal de seis commits, commits de
  un solo path/propósito, HEAD final exacto y worktree limpio (`:162-182`).
- **Separación Wave 56/Wave 57.** El dispatch continúa ligado al
  `prospective_config.schema_version` y exige negativos de regresión para la
  autoridad histórica de Ola 56 (`:122-132,191-205`).
- **Replay.** Continúan siendo obligatorios el primario canónico, la misma copia
  de amendment, escrow/freeze, manifest, visibles, logits y provenance exactos
  (`:154-160,209-213,231-245`).

La reauditoría fue read-only respecto del origen: no abrió escrow, secret files
ni truth sellada; no ejecutó inferencia, oracle, labels, training o GPU/CUDA.

## Condición concreta para PASS

Reemplazar todas las prohibiciones de “apertura” por una prohibición precisa de
lectura semántica, parsing y extracción, declarar el streaming binario de hash
como única apertura pre-key permitida y exigir spies que permitan ese camino
opaco mientras rechazan todos los consumidores semánticos enumerados. No hace
falta modificar ninguna otra garantía del plan.

## Veredicto

**REVISE.** La revisión resolvió las aserciones de ausencia y acertó con la
arquitectura content-blind seguida de validación semántica autorizada, pero su
contrato literal de I/O sigue siendo internamente contradictorio. El finding no
afecta la validez del same-escrow recovery, no justifica redraw, no exige cambio
de config científico y no constituye GO/NO-GO.

## Machine-verifiable decision

**Final decision:** `REVISE`

