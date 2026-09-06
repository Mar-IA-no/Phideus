# Ola 60 — resolución de R473 antes de la autoridad source law v2

> **Estado:** `PRE-AUDIT / PRE-IMPLEMENTATION-REVISION / CPU-ONLY / NO-DRAW / NO-GO-NOGO`
> **Fecha:** 2026-09-06
> **Contrato científico base:** `WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md`
> **Plan de recuperación:** `WAVE_60_SOURCE_LAW_RECOVERY_PLAN.md`
> **Reauditoría del plan:** `R472 / PASS / 0 HIGH + 0 MEDIUM + 0 LOW`
> **Implementación rechazada:** `53c383284e222cd60890a87857d66a8245f1a148`
> **Auditoría de implementación:** `R473 / REVISE / 1 HIGH + 2 MEDIUM + 0 LOW`
> **Commit R473:** `50b07dcfb8509ec5b235da44a6fa4328fd4db737`
> **SHA-256 R473:** `07f0c750ec9c13a364b51aa5729eb10723831dd190b8d6d10054bffb22fdb387`

## 1. Motivo y alcance

R473 confirmó que la implementación `53c3832` preserva el terminal v1, el
sandbox del worker, el presupuesto durable y la ausencia de draw, pero encontró
tres defectos previos a una ejecución canónica:

1. el validador de auditoría exige el scope histórico `IMPLEMENTATION` y por
   eso rechazaría la autoridad nueva
   `SOURCE_LAW_RECOVERY_IMPLEMENTATION`;
2. un request recovery dirigido a un output alternativo puede reservar y
   publicar un terminal legacy fuera de la root v2, con
   `recovery_allowed=true`;
3. la suite no atraviesa positivamente el publisher recovery completo ni prueba
   API, CLI, journal acumulativo, attestation, manifest y rename como una sola
   operación.

El plan de recuperación original sólo fijó la cadena nominal para un PASS en la
primera auditoría. No especificó cómo preservar una auditoría `REVISE` y una
implementación sucesora sin reescribir commits. Este documento resuelve esa
omisión y los tres findings R473. No modifica la ley científica, fuentes,
estimandos, thresholds, políticas, controles, seeds, bootstrap, presupuesto,
terminal v1 ni namespace del futuro draw.

## 2. Preservación de historia y nueva cadena

Quedan inmutables y recuperables:

- plan de recuperación `a8e8932`;
- PASS R472 `346e9fc`;
- implementación rechazada `53c3832`;
- informe R473 `50b07dc`, child directo de `53c3832`.

La resolución continúa linealmente, sin amend, squash, reset ni cherry-pick:

```text
R472 PASS
  -> implementación 53c3832
  -> R473 REVISE 50b07dc
  -> este plan de resolución
  -> auditoría PASS de este plan
  -> implementación corregida
  -> reauditoría PASS de implementación
  -> auditoría PASS de source law v2
  -> config final
  -> auditoría PASS de config
```

El runtime autentica ambas capas:

1. base: plan `a8e8932` y R472 `346e9fc`;
2. revisión: `53c3832`, R473 `50b07dc`, este plan y su auditoría PASS.

La implementación corregida debe ser child directo de la auditoría PASS de este
plan. Su reauditoría debe ser un commit exclusivo y child directo de la
implementación corregida. La auditoría source law sigue siendo child directo de
esa reauditoría. El request v2 liga directamente implementación corregida y
reauditoría; el código ejecutado liga transitivamente la historia R473 completa.

Esta extensión reemplaza únicamente la arista nominal
`parent(implementación)=R472` del §10 del plan de recuperación cuando existe un
`REVISE` autenticado. Todas las aristas posteriores permanecen iguales.

## 3. Autoridad exacta de R473 y de esta resolución

La implementación corregida agrega constantes para:

- commit, path y SHA-256 de R473;
- commit y SHA-256 de la implementación rechazada;
- commit, path y SHA-256 de este plan;
- commit, path y SHA-256 de su auditoría.

Antes de aceptar el request v2, verifica:

- cada commit y blob físico por Git y SHA-256;
- exclusividad de cada commit documental;
- parents directos de la cadena §2;
- exactamente un bloque JSON canónico por auditoría;
- R473 con `technical_verdict=REVISE`, findings `1/2/0`, scope
  `SOURCE_LAW_RECOVERY_IMPLEMENTATION`, target `53c3832`, sin archivos
  modificados y sin GPU consultada;
- auditoría de este plan con `technical_verdict=PASS`, findings `0/0/0`, scope
  `SOURCE_LAW_RECOVERY_R473_RESOLUTION_PLAN` y target por commit+SHA del plan;
- implementación corregida limitada a los cinco paths ya autorizados por el
  plan de recuperación;
- reauditoría final con scope
  `SOURCE_LAW_RECOVERY_IMPLEMENTATION`, target exacto de la implementación
  corregida y PASS `0/0/0`.

Un mismatch ocurre después de reservar exclusivamente el namespace v2 y produce
`SOURCE_LAW_INVALID`, `recovery_allowed=false`. Las excepciones de invocación y
namespace definidas en §5 ocurren antes de toda escritura.

## 4. Resolución del HIGH: scopes de auditoría

`validate_implementation_audit_authority` recibe un scope esperado explícito:

- `IMPLEMENTATION` sólo para autenticar la implementación histórica ligada al
  terminal v1;
- `SOURCE_LAW_RECOVERY_IMPLEMENTATION` para el request v2, la futura config y
  toda validación de la implementación corregida.

El runner, el worker cuando corresponda y `prepare_wave56_fresh.py` deben usar
el scope recovery en la rama Wave 60 v2. Ningún caller puede depender de un
default ambiguo.

Las pruebas construyen o simulan una autoridad Git con forma exacta R473/R474 y
demuestran:

- aceptación del scope recovery correcto por publisher y config;
- rechazo de `IMPLEMENTATION` para la implementación corregida;
- preservación de `IMPLEMENTATION` para el request histórico v1;
- rechazo de scope, target, verdict, conteos o parent alterados.

## 5. Resolución del namespace: entrypoints separados

La API expone dos entrypoints explícitos:

- `publish_source_law_authority`: compatibilidad legacy, nunca acepta el schema
  recovery;
- `publish_source_law_recovery`: única ruta autorizada para
  `wave60-source-law-recovery-request-v1`.

El CLI `verify-source-law` de Wave 60 usa exclusivamente el entrypoint recovery.

Antes de crear staging, `publish_source_law_recovery`:

1. valida que el request sea un archivo regular, no symlink y legible;
2. canonicaliza el output respecto de `REPO_ROOT`;
3. exige igualdad exacta con
   `data/geometria_proporcional/wave60_frozen_policy_transport_source_law_v2`;
4. rechaza output fuera del repo, traversal, symlink, alias, target existente o
   staging preexistente;
5. identifica el schema recovery sin convertir errores semánticos posteriores
   en publicaciones legacy.

Un output recovery distinto levanta excepción sin crear ni modificar target o
staging. Nunca publica `recovery_allowed=true`. Una vez reservado el staging v2,
cualquier fallo semántico, criptográfico, de lineage, presupuesto o worker
produce el único terminal v2 no recuperable definido por el plan base.

El entrypoint legacy rechaza el schema recovery antes de reservar su namespace.
No puede utilizarse para rodear la igualdad v2.

## 6. Prueba positiva integrada

La suite nueva agrega una prueba positiva que atraviesa el mismo entrypoint
recovery que usa el CLI. Debe verificar como una unidad:

- request recovery completo y auditoría recovery autenticable;
- path API repo-relative;
- path API absoluto;
- dispatch CLI con path repo-relative;
- publicación atómica sólo en la root solicitada;
- request preservado byte a byte;
- v1 inmutable antes y después;
- worker sin acceso a v1, attempt ni truth;
- reproducción exacta de los outputs fuente;
- journal con duración actual, duración durable v1 y suma exacta `<900`;
- RSS `<1.5 GiB`, `truth_accessed=false` y CUDA invisible;
- attestation válida y ligada a request, freeze, receipt y journal;
- manifest closed-world congruente con todos los bytes;
- target final `SOURCE_LAW_VERIFIED` y staging ausente;
- attempt y staging de attempt ausentes.

La prueba puede usar un repositorio Git temporal para la cadena de auditoría o
inyectar de manera explícita sólo la frontera de autoridad ya probada por el
test Git. No puede monkeypatchear el worker, el presupuesto, el journal, la
attestation, el manifest ni el rename en el caso positivo integrado.

También agrega casos negativos para:

- request recovery pasado al publisher legacy;
- output recovery seguro pero distinto, relativo y absoluto;
- staging alternativo y target alternativo ausentes después del rechazo;
- scope histórico usado en la implementación nueva;
- scope recovery usado para autenticar el request v1 histórico.

La ejecución productiva posterior, si la implementación es aceptada, repite el
caso CLI repo-relative sobre la root v2 canónica. Esa ejecución no sustituye la
cobertura previa: la prueba integrada debe cerrar antes de consumir el namespace
canónico.

## 7. Presupuesto, recursos y artefactos

Continúan vigentes:

- CPU-only y CUDA invisible;
- `OMP_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4`, `MKL_NUM_THREADS=4` para la
  regresión amplia;
- source law acumulativa `<900 s`;
- RSS por proceso `<1610612736` bytes;
- monitoreo periódico de RAM y registro de swaps;
- prohibición de Colab, Mendieta y GPU;
- conservación de outputs reutilizables, manifests, receipts y estados crudos.

Los basetemps se crean en `/mnt/m2-1TB`, se atribuyen al turno y se eliminan por
path exacto una vez terminados los procesos. Los errores `EXDEV` o `ENOSPC`
provocados por ubicar fixtures fuera de sus precondiciones no se cuentan como
fallos del código, pero el comando canónico de regresión debe ejecutarse con un
basetemp compatible.

## 8. Matriz de verificación antes del nuevo freeze

La implementación corregida no se congela hasta demostrar:

1. tests focales de los tres findings R473;
2. suite Wave 60 completa;
3. regresión Wave 56–60 completa sobre el commit candidato;
4. `git diff --check` y compilación Python;
5. hashes físicos v1 exactos antes y después;
6. ausencia de source law v2, su staging, attempt y staging de attempt;
7. commit limitado a los cinco paths autorizados;
8. parent directo igual a la auditoría PASS de este plan;
9. reauditoría independiente posterior, sin modificar archivos y con
   `SOURCE_LAW_RECOVERY_IMPLEMENTATION / PASS / 0/0/0`.

Si la reauditoría abre un finding sustantivo, se preserva su informe y se repite
un ciclo explícito de resolución. No se ejecuta source law mientras exista un
finding abierto.

## 9. Lectura inferencial

Resolver R473 sólo habilita una publicación pre-draw correcta. No demuestra
transporte entre draws, no mejora los resultados de Ola 59, no valida HGB, no
promueve proposer/guard y no constituye GO/NO-GO.

El valor metodológico de esta resolución es más estrecho: evita que una
incompatibilidad de scopes o un alias de output consuma la única recuperación
permitida, y obliga a probar el publisher completo antes de usar el namespace
canónico.
