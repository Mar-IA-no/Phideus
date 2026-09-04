# Ola 57 — plan de recuperación pre-oracle del contrato de pair tokens

> **Estado:** `PRE-IMPLEMENTATION / PRE-RECOVERY / SAME-ESCROW / CPU-ONLY / NO-GO-NOGO`
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
tamaño y hash— y los conteos por split. Una diferencia posterior invalida la
recuperación.

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
   `no_inference_in_origin`, `no_oracle_in_origin` y `no_labels_in_origin`;
4. que plan, auditoría de plan, implementación, auditoría de implementación,
   amendment y auditoría final formen la cadena lineal predeclarada;
5. que cada commit de esa cadena cambie sólo los paths autorizados;
6. que el HEAD de ejecución sea exactamente el commit de auditoría final y el
   worktree esté globalmente limpio;
7. que el origen coincida con basename, contrato, hashes, inventario físico,
   atestación semántica y conteos declarados;
8. que el delta entre contrato de escrow y contrato de ejecución conserve
   todos los campos salvo `git_commit` y hashes de las dos fuentes aprobadas;
9. que esas dos fuentes sean únicamente el preparador compartido y el test de
   recovery Wave 57, con hashes old/new declarados;
10. que replay parta del primario canónico recuperado y de la misma copia del
    amendment.

Los reports de auditoría usan un bloque único y parseable, fuera de fences o
comentarios. La auditoría de plan declara commit y SHA del plan; la de
implementación declara commit y hashes del preparador y test; la final declara
commit del amendment y SHA del amendment. Un resultado distinto de `PASS`, una
segunda atestación contradictoria o un path ejecutable no otorga autoridad.

## Secuencia de commits cerrada

La cadena se construye sin commits mezclados:

1. este plan y ningún otro path;
2. auditoría independiente del plan y ningún otro path;
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
- `secrets.token_bytes` imposible de invocar en recovery;
- regeneración con las mismas claves y manifest byte-exacto;
- conteo aceptado sólo por `eligible_unique_pair_tokens=768`, manteniendo
  visibles `total_unique_pair_tokens=1152` y los demás conteos;
- copia canónica del amendment en primario y replay;
- replay exacto de escrow, freeze, benchmark visible, logits y provenance.

Los tests sintéticos pueden construir repositorios temporales y orígenes
reducidos, pero al menos un probe final debe validar el amendment canónico
contra el origen real sin abrir labels ni iniciar inferencia.

## Ejecución recuperada

Con la cadena aprobada, el primario se reconstruye en
`wave57_contextual_tail_guard_fresh_v1/` usando simultáneamente:

```text
--recovery-secrets-from <failed-origin>
--recovery-amendment <canonical-amendment>
```

El preparador vuelve a generar el benchmark desde las mismas claves, compara
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
