# Ola 57 — plan de recuperación pre-oracle del contrato de pair tokens

> **Estado:** `REVISED-AFTER-R399 / FOR-FOCAL-REAUDIT / PRE-IMPLEMENTATION / PRE-RECOVERY / SAME-ESCROW / CPU-ONLY / NO-GO-NOGO`
> **Fecha:** 2026-09-04
> **Draw de origen:** `wave57_contextual_tail_guard_fresh_v1.failed_20260904T215941566314Z`
> **Contrato científico:** `WAVE_57_CONTEXTUAL_TAIL_GUARD_PLAN.md`

## Situación que exige el addendum

La preparación primaria de Ola 57 extrajo y publicó durablemente tres claves,
generó el benchmark fresco y se detuvo antes de inferencia, materialización de
labels y apertura de cualquier fase analítica. El fallo no provino de una
realización anómala. Los tres splits tienen exactamente `4992` fixtures,
`1152` pair tokens totales y `768` pair tokens elegibles. Es la población que
el plan y el campo
`expected_eligible_pair_tokens_per_split=768` habían congelado.

La ruta inicial del preparador compartido conservaba, sin embargo, una
semántica legacy: fuera de recovery comparaba ese campo contra
`total_unique_pair_tokens`. Por eso contrastó `768` con `1152` y produjo:

```text
RuntimeError: fresh benchmark pair-token count differs from prospective freeze
```

No corresponde cambiar el tamaño esperado a `1152`, porque eso transformaría
un mínimo de población elegible en un conteo que incorpora OOD. Tampoco
corresponde redibujar: el escrow ya existe y la realización es válida. La única
operación autorizable es recuperar el mismo draw, verificarlo byte por byte y
aplicar la semántica que el nombre, el plan y la selección primaria ya fijaron:
filtrar elegibilidad por fila antes de deduplicar `pair_token`.

## Evidencia inmutable del origen

El intento fallido queda preservado en:

```text
data/geometria_proporcional/
  wave57_contextual_tail_guard_fresh_v1.failed_20260904T215941566314Z/
```

Su identidad observada antes de diseñar este addendum es:

| Artefacto | SHA-256 |
|---|---|
| contrato embebido en escrow | `57e727120346363a940d3645f20c064e49aa080a806e65d3c9893ae37e7d7be7` |
| `generation_escrow.json` | `d489f443a5c837269e68547d4c4514348ca4115e2d676221c4b4617861be0643` |
| `pre_generation_freeze.json` | `20c94171a053ba2a4d17b13c9dcc865a995bb328c990a53d0bf5f0a7daecea91` |
| `FAILURE.json` | `710b7d29de8c0436304ffb7abdfb2adcd958ed443ab72a9190ce74495e8602af` |
| `benchmark/manifest.json` | `3d444db8a1e761c0c67234471a827e43d52afaf29ca60b8a52a760167f713b2b` |

El contrato del escrow nació en el commit
`379229f1cae0f1b713fe5c293f303ed60ed7f187`. El árbol físico contiene `24`
entradas. No contiene `inference/`, `authorized_labels/`, `bundles/`,
`phases/`, `preparation_freeze.json` ni `generation_receipt.json`. El amendment
canónico deberá incluir el inventario físico completo —tipo, modo, uid, gid,
tamaño y hash— y los conteos por split. Sí contiene, como corresponde al output
del generador, truth sellada y sus tres secret files bajo `benchmark/sealed/`.
Las aserciones de ausencia se refieren exclusivamente al oracle analítico
materializado y a labels autorizados, no a esa truth sellada esperada. Una
diferencia posterior invalida la recuperación.

## Invariantes que no cambian

La recuperación no modifica el config prospectivo, el plan científico, las
features, modelos, targets, grids, mínimos, estimandos, bootstrap, criterios,
seeds ni políticas. En particular:

- `768` continúa significando pair tokens elegibles por split;
- la elegibilidad continúa siendo
  `is_out_of_catalog=false` y
  `calibration_population=canonical_preserving`;
- el filtrado ocurre por fila antes de deduplicar tokens;
- las tres claves deben ser exactamente las del escrow fallido;
- el benchmark regenerado debe tener el mismo manifest SHA-256 que el origen;
- no se habilita acceso analítico a truth durante la recuperación;
- el worktree y los artefactos de autoridad deben estar cerrados antes de
  ejecutar;
- `GO/NO-GO` y promoción arquitectónica permanecen fuera de este addendum.

El cambio de código queda limitado al mecanismo que autentica la autoridad de
recovery. La rama de ejecución ya usa
`eligible_unique_pair_tokens` cuando existe un `recovery_context`; no se cambia
esa operación ni se agrega una excepción de conteo ad hoc.

## Dos etapas de acceso que no pueden confundirse

R399 mostró que el helper histórico `validate_failed_recovery_origin()` no es
una validación pre-key: llama a `read_escrow()`, reconstruye las tres claves,
abre los tres `*_secret.json` y carga truth sellada mediante
`validate_semantic_attestation()`. La implementación Wave 57 no puede invocar
ese helper durante la autenticación inicial.

La primera etapa es un **preflight de autoridad content-blind**. Puede hacer
`lstat`, inventariar modos/ownership/tamaños, calcular hashes de bytes sin
interpretarlos, leer `FAILURE.json`, manifest, visibles, atestación detached y
el `pre_generation_freeze.json` secret-free. Debe validar toda la cadena Git,
el HEAD, el worktree, el contrato público, el delta de fuentes, el inventario y
la firma detached usando sólo la clave pública. En esta etapa está prohibido:

- llamar a `read_escrow()` o `keys_from_escrow()`;
- abrir o parsear cualquier `benchmark/sealed/*_secret.json`;
- llamar a `validate_semantic_attestation()` o a otro helper que cargue truth
  sellada;
- parsear los JSONL bajo `benchmark/sealed/` para contar poblaciones.

El SHA-256 de esos archivos sí puede comprobarse como flujo opaco porque el
inventario y el manifest lo exigen; ese hash no materializa claves ni targets.
Los tests instrumentan las funciones de apertura y deben fallar si alguna se
invoca durante el preflight.

Sólo después de que toda la autoridad anterior pasa comienza la segunda etapa,
**validación semántica autorizada del origen**. Allí se abre por primera vez el
escrow de origen, se comprueba que su contrato y commitments coinciden con el
freeze público autenticado, se validan la atestación semántica y los secret
files, y se calculan los conteos estructurales sobre los JSONL sellados. Esta
apertura no habilita inferencia, oracle analítico ni labels autorizados: sólo
demuestra que el draw opaco ya autorizado es internamente consistente. Las
revalidaciones posteriores del origen y de la copia republished ocurren dentro
de esa autoridad ya concedida y deben conservar los mismos hashes.

## Autoridad específica para Ola 57

El preparador compartido conserva intacta la autoridad histórica de Ola 56 y
agrega un despacho tipado por `prospective_config.schema_version`. Para
`wave57-contextual-harm-guard-v1`, sólo acepta:

```text
schema: wave57-contextual-tail-guard-preoracle-recovery-amendment-v1
path:   experiments/geometria_proporcional/configs/
        wave57_preoracle_pair_token_recovery_amendment_v1.json
```

El nuevo validator debe comprobar, antes de extraer claves:

1. que la invocación sea `recovery` o `replay` y apunte a la ruta canónica;
2. que el amendment sea JSON canónico, versionado y con estado
   `APPROVED_PREORACLE_RECOVERY`;
3. que sus cuatro aserciones sean verdaderas: `no_redraw`,
   `no_inference_in_origin`, `no_materialized_oracle_in_origin` y
   `no_authorized_labels_in_origin`;
4. que plan, auditoría de plan, implementación, auditoría de implementación,
   amendment y auditoría final formen la cadena lineal predeclarada;
5. que cada commit de esa cadena cambie sólo los paths autorizados;
6. que el HEAD de ejecución sea exactamente el commit de auditoría final y el
   worktree esté globalmente limpio;
7. que el preflight content-blind haga coincidir el origen con basename,
   contrato público, hashes, inventario físico, manifest y firma detached sin
   abrir escrow, secret files ni truth sellada;
8. que el delta entre contrato de escrow y contrato de ejecución conserve
   todos los campos salvo `git_commit` y hashes de las dos fuentes aprobadas;
9. que esas dos fuentes sean únicamente el preparador compartido y el test de
   recovery Wave 57, con hashes old/new declarados;
10. que replay parta del primario canónico recuperado y de la misma copia del
    amendment.

Tras esos diez checks, y nunca antes, la etapa semántica valida el escrow, los
commitments, la atestación completa y los conteos declarados. Un fallo allí
detiene recovery, pero no retroactivamente convierte el preflight de autoridad
en una lectura de claves.

Los reports de auditoría usan un bloque único y parseable, fuera de fences o
comentarios. La auditoría de plan declara commit y SHA del plan; la de
implementación declara commit y hashes del preparador y test; la final declara
commit del amendment y SHA del amendment. Un resultado distinto de `PASS`, una
segunda atestación contradictoria o un path ejecutable no otorga autoridad.

## Secuencia de commits cerrada

La cadena se construye sin commits mezclados:

1. la revisión vigente de este plan y ningún otro path;
2. reauditoría independiente de la revisión y ningún otro path;
3. implementación del dispatcher/validator y sus tests;
4. auditoría independiente de implementación y ningún otro path;
5. amendment canónico poblado con hashes y commits ya observables;
6. auditoría final del paquete de autoridad y ningún otro path.

El commit 6 debe ser HEAD exacto al iniciar recovery. No se permiten cambios
posteriores hasta que la preparación primaria recuperada quede completa. Esta
linealidad evita que un informe favorable autentique código o un amendment
distintos de los que leyó.

## Implementación y pruebas exigidas

La implementación modifica solamente:

- `experiments/geometria_proporcional/prepare_wave56_fresh.py`;
- `tests/test_wave57_prospective.py`.

La cobertura debe demostrar al menos:

- dispatch Wave 56/Wave 57 sin debilitar la ruta histórica;
- rechazo de amendment Wave 57 en path alternativo;
- rechazo de schema, estado, assertions o población alterados;
- rechazo de inventario, hash, ownership o modo alterados;
- rechazo si aparece inferencia, label materializado, oracle analítico o un
  archivo extra en el origen;
- rechazo si el contrato cambia config, bindings, fuente no autorizada o
  inventario de fuentes;
- rechazo de commits no lineales, mixed commits, hashes old/new falsos,
  auditorías ausentes/contradictorias y HEAD posterior;
- instrumentación que haga fallar el preflight si llama a `read_escrow()`,
  `keys_from_escrow()`, `validate_semantic_attestation()`,
  `sealed_population_counts()` o intenta abrir un `*_secret.json`;
- comprobación de que el preflight usa el contrato secret-free del freeze
  público y termina toda la cadena antes del primer acceso semántico al origen;
- `secrets.token_bytes` imposible de invocar en recovery;
- regeneración con las mismas claves y manifest byte-exacto;
- conteo aceptado sólo por `eligible_unique_pair_tokens=768`, manteniendo
  visibles `total_unique_pair_tokens=1152` y los demás conteos;
- copia canónica del amendment en primario y replay;
- replay exacto de escrow, freeze, benchmark visible, logits y provenance.

Los tests sintéticos pueden construir repositorios temporales y orígenes
reducidos, pero al menos un probe final debe validar el amendment canónico
contra el origen real en modo content-blind, con spies que demuestren ausencia
de apertura del escrow, secret files y truth sellada. El probe no inicia
inferencia ni materializa oracle o labels autorizados.

## Ejecución recuperada

Con la cadena aprobada, el primario se reconstruye en
`wave57_contextual_tail_guard_fresh_v1/` usando simultáneamente:

```text
--recovery-secrets-from <failed-origin>
--recovery-amendment <canonical-amendment>
```

Después del preflight completo, el preparador abre el escrow autorizado y
valida semánticamente el origen. Luego vuelve a generar el benchmark desde las
mismas claves, compara
su manifest con `3d444d...`, revalida el origen antes y después de generación,
y sólo entonces cruza la comprobación de población elegible. Después ejecuta
la inferencia ciega y publica `PREPARED`. Las fases FIT, SELECT y ADJUDICATE se
ejecutan sin cambiar código ni contrato.

El replay usa el primario canónico como fuente de escrow y referencia. Debe
reproducir preparación y fases analíticas de forma exacta bajo la matriz ya
auditada de Ola 57.

Si cualquier validación falla, se preserva el nuevo intento con su
`FAILURE.json`. No se autoriza otro draw, no se relaja un mínimo y no se
recodifica un defecto de implementación como evidencia científica.

## Presupuesto operativo

Diseño, validación, regeneración e inferencia permanecen en CPU. La regeneración
repite trabajo determinista breve ya realizado y la inferencia usa el encoder
pequeño congelado; una GPU no cambiaría la autoridad del recovery ni ofrece una
ventaja material que justifique reservarla. Si apareciera una etapa distinta
que sí requiriera CUDA o una carga CPU desproporcionada, el ciclo se detendría
antes de ejecutarla y se escalaría a Mariano por el mecanismo vigente.

## Resolución de R399

R399 emitió `REVISE` con un finding alto y uno medio. Esta revisión separa el
preflight content-blind de la validación semántica autorizada, prohíbe y prueba
explícitamente toda apertura de escrow, secret files o truth sellada durante la
primera etapa, y sitúa el primer acceso semántico sólo después de autenticar la
cadena completa. También reemplaza las aserciones ambiguas por
`no_materialized_oracle_in_origin` y
`no_authorized_labels_in_origin`, distinguiéndolas de la truth sellada que el
generador debe preservar.
