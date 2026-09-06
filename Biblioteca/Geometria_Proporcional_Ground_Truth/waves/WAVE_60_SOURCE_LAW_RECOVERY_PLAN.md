# Ola 60 — recuperación pre-draw de la autoridad de ley fuente

> **Estado:** `DRAFT / PRE-AUDIT / PRE-IMPLEMENTATION / CPU-ONLY / NO-DRAW / NO-GO-NOGO`
> **Fecha:** 2026-09-06
> **Contrato científico base:** `WAVE_60_FROZEN_POLICY_TRANSPORT_PLAN.md`
> **Implementación previa aceptada:** `d694d803bb72bfddd5877139bb2c9be40cc74579`
> **Auditoría previa:** `R470 / PASS / 0 HIGH + 0 MEDIUM + 0 LOW`
> **Pregunta:** ¿puede recuperarse la publicación de la ley fuente después de
> un fallo de canonicalización de path ocurrido antes del worker y antes del
> draw, sin sustituir el terminal v1 ni cambiar el estimando de Ola 60?

## 1. Incidente y alcance

La primera invocación de `verify-source-law` recibió como `--output` el path
relativo:

```text
data/geometria_proporcional/
  wave60_frozen_policy_transport_source_law_v1
```

El publicador comparó ese `Path` relativo mediante
`output.relative_to(REPO_ROOT)`, donde `REPO_ROOT` es absoluto. Python rechazó
la operación antes de validar la implementación y antes de construir o lanzar
el worker. El coordinador aplicó correctamente el contrato terminal existente:
publicó atómicamente una root `SOURCE_LAW_INVALID`, firmada, sin outputs
científicos y con `truth_accessed=false`.

La causa queda identificada de manera reproducible por:

```text
error_type                 = ValueError
error_message_sha256       = 78950540d7b16ef8e8d21acd5149ed3c186d0655f29471f88a0a579e881f6eb7
failed_git_commit          = ab60d325075996fbfde14d685c7fdf5c1d0909af
source_law_request_sha256  = 0d53edf28658bb9437e74c1e5ab53ac69bfd2299670d12a10a00134f61904ba0
journal_sha256             = ad96491b595deecfa249ee5c5eafee418487bc2b99962a96f82117eea863ce19
failure_sha256             = 54ba12ba6a60cd4eeb802a9587a9fed263c5423562453d63610cba4a57813e3d
failure_inventory_sha256   = 983f8af0badc6426ab401ca6ceb5b8b4dcc6199f39fdec018243c4c576024d07
failure_attestation_sha256 = 39f88dab7d1cf6b50e6398820403c5c75fbef1e4b18931c7a544d8bb3475175a
```

La duración durable declarada fue `0.00401783362030983 s` y el RSS máximo
registrado fue cero. No existen
`wave60_frozen_policy_transport_attempt_v1/` ni su staging, y tampoco existe un
staging residual de la autoridad fuente. Por construcción y por el orden del
código, no se abrió ningún artefacto Wave 59 posterior al request, no se lanzó
el worker, no se creó escrow Wave 60 y no se dibujó ninguna realización.

Este plan no recupera un resultado observado. Recupera una operación de
publicación que falló antes de producir evidencia científica.

## 2. Invariantes científicos

La recuperación no puede cambiar:

- el contrato científico, estimandos, patrones ni límites de Ola 60;
- la ejecución Wave 59 que constituye la fuente;
- los nueve hashes científicos `SOURCE_HASHES`;
- los trece estados HGB usados y los tres estados excluidos;
- las `1.300` tree keys, las diecisiete features y su orden;
- los thresholds, operadores estrictos, `score_mask` y `decision_mask`;
- las dos políticas principales, los diez controles y las dos referencias;
- el benchmark, encoder, seeds `{17,29,43}`, utility matrix y penalty `1.25`;
- el bootstrap `PCG64`, seed `6007`, `5.000` réplicas y unidad `pair_token`;
- el presupuesto de las roots, los estados terminales y la autoridad de decisión
  del usuario.

La recuperación tampoco habilita refit, recalibración, selección de políticas,
consulta de train/validation nuevos, acceso a truth ni generación del draw. Si
cualquiera de esos invariantes no puede demostrarse, no se publica una autoridad
v2.

## 3. Preservación del terminal v1

La root siguiente permanece inmutable y nunca se renombra, borra, completa ni
reutiliza como staging:

```text
data/geometria_proporcional/
  wave60_frozen_policy_transport_source_law_v1/
```

Antes de una ejecución v2, el coordinador valida físicamente:

1. root regular, no symlink, `root:root/0700`;
2. subdirectorio `journals/` regular, no symlink, `root:root/0700`;
3. conjunto exacto de cinco archivos;
4. modos `0444`, owner y group de cada archivo;
5. keysets y schemas exactos de request, journal, failure, inventory y
   attestation;
6. hashes físicos iguales a los fijados en §1;
7. inventario closed-world congruente con esos bytes;
8. firma Ed25519 válida de `failure_attestation.json` contra la clave pública
   congelada;
9. enlaces internos request → journal → failure → inventory → attestation;
10. `terminal=SOURCE_LAW_INVALID`, `phase=verify_source_law`,
    `truth_accessed=false`, `recovery_allowed=true`;
11. commit fallido y hash del error exactos;
12. ausencia de todo output científico source law dentro de v1;
13. ausencia de cualquier contenedor o staging de intento Wave 60.

Un mismatch produce un nuevo terminal v2 inválido; nunca habilita reparar v1.

## 4. Autoridad sucesora y request de recuperación

La única candidata sucesora vive en:

```text
data/geometria_proporcional/
  wave60_frozen_policy_transport_source_law_v2/
```

Su request usa el schema cerrado
`wave60-source-law-recovery-request-v1`. Conserva todos los campos del request
base y agrega una única clave `recovery`, con esta forma exacta:

```text
recovery = {
  schema_version,
  recovery_plan_commit, recovery_plan_path, recovery_plan_sha256,
  recovery_plan_audit_commit, recovery_plan_audit_path,
  recovery_plan_audit_sha256,
  prior_authority_path,
  prior_source_law_request_sha256,
  prior_journal_sha256,
  prior_failure_sha256,
  prior_failure_inventory_sha256,
  prior_failure_attestation_sha256,
  prior_terminal, prior_git_commit,
  prior_error_type, prior_error_message_sha256,
  prior_truth_accessed, prior_recovery_allowed
}
```

El mapa liga este plan y su auditoría independiente, además del terminal físico
v1. `output_path` debe ser exactamente el path v2 anterior. Los paths son
repo-relative, normalizados, sin `..`; los commits y hashes se verifican contra
Git y contra los archivos físicos. La auditoría del plan debe ser un commit
exclusivo, parent directo del commit exclusivo de este plan, y debe otorgar
`PASS` sin findings abiertos en un bloque machine-readable.

El request v2 no necesita introducirse como commit entre la auditoría de
implementación y la auditoría source law. Queda preservado byte a byte dentro
de la autoridad v2, ligado por `source_law_freeze.json`, por la attestation y
por `source_authority_manifest.json`. Esto evita romper la cadena directa:

```text
implementation commit
  -> implementation audit commit
  -> source-law audit commit
  -> final config commit
  -> final config audit commit
```

## 5. Corrección de canonicalización

Todo entrypoint convierte el argumento de output a una ruta absoluta bajo
`REPO_ROOT` antes de cualquier comprobación de existencia, creación de staging
o `relative_to`. Un argumento relativo se interpreta siempre respecto de
`REPO_ROOT`, no respecto del directorio de trabajo accidental del proceso.

La función pública aplica la misma normalización aunque sea invocada sin CLI.
Después exige:

- pertenencia física al repositorio;
- igualdad exacta entre path normalizado y `REPO_ROOT / request.output_path`;
- para el request de recuperación, igualdad exacta con la root v2;
- ausencia de symlinks en el target y en componentes existentes del path;
- target y staging inexistentes.

La normalización ocurre antes de reservar el namespace. Un path fuera del repo,
un traversal o un alias por symlink se rechaza sin redirigir la publicación a
otra ubicación.

## 6. Ejecución y presupuesto durable

Sólo después de validar §3–§5 se crea el staging v2 y se lanza
`verify_source_law` bajo UID/GID `65534`, capabilities vacías y
`NoNewPrivs=true`. El worker recibe exactamente el request v2 y los mismos once
aliases físicos del contrato anterior; no recibe el directorio v1, ningún draw
Wave 60 ni truth.

El worker admite el schema de recuperación como una variante tipada del request
de source law. Valida su keyset y el mapa `recovery` completo contra las
constantes auditadas, pero no gana acceso al terminal v1. La validación física y
criptográfica de ese terminal pertenece al coordinador; el freeze científico
liga el SHA-256 del request completo.

El presupuesto source-law continúa siendo `900 s` y `1.5 GiB` por proceso. Los
`0.00401783362030983 s` durables de v1 se descuentan antes de lanzar el worker.
El journal exitoso v2 registra por separado:

```text
duration_seconds
prior_durable_elapsed_seconds
cumulative_duration_seconds
max_rss_bytes
truth_accessed=false
```

y exige `cumulative_duration_seconds < 900`. El RSS de v1 fue cero; el límite
por proceso no se acumula. CUDA permanece invisible y no se consulta GPU,
Colab ni Mendieta.

## 7. Outputs, binding y terminales

Los cinco outputs científicos permanecen bajo `wave60-source-law-v1`; no se
modifica su semántica por tratarse de una recuperación operacional:

```text
source_law_freeze.json
transport_law_manifest.json
transport_law_arrays.npz
frozen_policy_spec.json
feature_schema.json
```

La autoridad v2 agrega los mismos receipt, attestation, journal, request y
manifest del contrato base. El request completo hace transitive el binding al
terminal v1 y al plan de recuperación. No se copia ningún archivo v1 por
hardlink.

El terminal v2 tiene sólo dos posibilidades:

| Terminal | Obligatorio | Prohibido | recovery_allowed |
|---|---|---|---:|
| `SOURCE_LAW_VERIFIED` | paquete source law completo y manifest closed-world | triple de failure | n/a |
| `SOURCE_LAW_INVALID` | request v2, journal y triple de failure | outputs científicos, config y draw | `false` |

Un fallo v2 no autoriza una tercera ejecución por este plan. Cualquier sucesor
exigiría otro diseño y otra auditoría.

La config final sólo acepta el path v2. Su binding al manifest y a la auditoría
source law hace transitivo el request de recuperación. La root v1 no se
presenta como autoridad científica y no se copia a primary/replay.

## 8. Pruebas obligatorias

### 8.1 Terminal v1

- aceptación del paquete físico exacto actualmente preservado;
- rechazo por bit alterado en cada uno de sus cinco archivos;
- rechazo de firma inválida, hash desligado o keyset extra/faltante;
- rechazo de owner, group, modo, symlink, hardlink interno inesperado o extra;
- rechazo si aparece un output científico en v1;
- rechazo si existe cualquier attempt/staging Wave 60 antes del source law.

### 8.2 Paths y namespaces

- CLI con el path repo-relative correcto publica en la root absoluta v2;
- invocación API con path relativo produce la misma identidad;
- path absoluto correcto continúa funcionando;
- rechazo de path fuera del repo, `..`, symlink de componente, output drift y
  staging preexistente;
- v1 existente nunca se modifica al intentar v2;
- un output v2 preexistente se rechaza y no se reescribe.

### 8.3 Request y cadena de autoridad

- aceptación del request recovery exacto;
- rechazo de cada hash/path/commit/boolean del mapa recovery alterado;
- rechazo de plan o auditoría no exclusivos, no contiguos o sin `PASS` exacto;
- rechazo de implementación no auditada o auditoría no direct-child;
- request, freeze, receipt, journal, attestation y manifest ligados sin ciclo;
- source audit exclusiva, parent directo de la auditoría de implementación;
- config exclusiva, parent directo de la auditoría source law;
- auditoría de config exclusiva como HEAD antes del draw.

### 8.4 Equivalencia y acceso

- reproducción exacta de los 13 scores, 26 arrays seleccionadas y hard de Ola
  59;
- igualdad del roster, 1.300 keys, thresholds, features y operadores con el
  contrato base;
- monkeypatch que falla si se invoca fit, quantile, selección o generación;
- worker sin acceso físico a v1, truth o cualquier path Wave 60 futuro;
- ausencia de attempt antes y después de publicar la autoridad v2;
- presupuesto acumulado menor que `900 s`, RSS menor que `1.5 GiB` y swaps del
  proceso iguales a cero.

### 8.5 Regresión

- tests focales nuevos para la recuperación;
- suite Wave 60 completa;
- regresión Wave 56–60 con CUDA invisible y cuatro threads;
- preservación exacta de los artefactos Wave 59 y del terminal v1.

## 9. Paths autorizados para la implementación

La corrección puede modificar exclusivamente:

```text
src/geometria_proporcional/wave60_frozen_policy_transport.py
experiments/geometria_proporcional/run_wave60_frozen_policy_transport.py
experiments/geometria_proporcional/_wave60_phase_worker.py
experiments/geometria_proporcional/prepare_wave56_fresh.py
tests/test_wave60_frozen_policy_transport.py
```

El preparador sólo cambia para aceptar y autenticar el path v2 y la nueva
cadena source-law. No cambia generación, inferencia, bundles, splits ni
preparación científica.

## 10. Cadena de autoridad

1. terminal v1 y request original preservados bajo commit `ab60d32`;
2. commit exclusivo de este plan;
3. auditoría independiente del plan de recuperación;
4. implementación exclusiva dentro de los cinco paths de §9;
5. auditoría independiente de implementación y regresión;
6. request v2 materializado y ejecutado sin introducir un commit intermedio;
7. auditoría independiente de la autoridad v2, como child directo de la
   auditoría de implementación;
8. config final exclusiva que liga autoridad v2 y sus hashes;
9. auditoría independiente de config como HEAD exacto;
10. recién entonces inicialización del attempt v1 y preparación del draw.

Cada auditoría conserva un bloque machine-readable con scope y target exactos,
`technical_verdict=PASS`, conteos `0/0/0`, `files_modified=false` y
`gpu_used_or_queried=false`. Un dictamen `REVISE`, una decisión contradictoria
o un `PASS` incidental en prosa no constituye autoridad.

## 11. Lectura inferencial

Una autoridad v2 válida sólo demostraría que la ley congelada de Ola 59 fue
proyectada y autenticada de acuerdo con el contrato. No constituye un resultado
de transporte, no aporta una nueva seed o réplica y no valida la arquitectura
proposer/guard.

La recuperación es admisible porque el fallo v1 precedió al worker, al draw y a
todo acceso científico. Esa admisibilidad no se generaliza a fallos posteriores:
una vez creado el draw, rigen íntegramente los terminales y las reglas de
recuperación del plan base. `scientific_decision` permanece `null` y la decisión
`GO/NO-GO` continúa perteneciendo al usuario.
