# R571 — Reauditoría final independiente del plan del paquete físico set-valued

**Fecha:** 2026-09-07  
**Commit auditado:** `cc19b655844e8de720540a9d6939753fdc272be2`  
**Parent:** `86f3bdbdfb75ef4bd095b9e0a39a98e3627594a5`  
**Objeto:** `experiments/geometria_proporcional/PLAN_PROPORTIONAL_SET_VALUED_PHYSICAL_PACKAGE_CPU.md`  
**SHA-256 del objeto:** `0256305424b1a80bf6dada54aad5d3e7c392150141ba093e391ce7ab7b12dcdb`  
**Régimen:** lectura completa, contraste con R569/R570 y probes CPU read-only; sin consultar, inicializar ni usar GPU/CUDA  
**Antecedente ejecutable:** runner y checker set-valued cerrados en R564

## Veredicto

**PASS TÉCNICO DEL PLAN — 0 HIGH / 0 MEDIUM / 0 LOW.**

La revisión `cc19b65` cierra los tres findings de R570 sin reabrir ninguno de
los ocho findings de R569. La allowlist staged forma ahora un cierre de imports
válido bajo la identidad, capabilities y environment prescriptos; el handoff
`evaluation_metadata.npz` transporta sólo la metadata pública necesaria y
queda ligado al input público y al action freeze; y el límite RSS cubre
nominalmente cada proceso y harness de cualquier campaña. La lectura integral
no encontró otro defecto de validez, seguridad o reproducibilidad que requiera
corregir el plan antes de implementar.

Este PASS evalúa la suficiencia técnica del **plan**. No acredita una
implementación todavía inexistente, no convierte el preflight abierto en
evidencia prospectiva, no promueve una arquitectura y no constituye una
decisión científica `GO/NO-GO`.

## Cierre de los ocho findings de R569

| Finding R569 | Estado en `cc19b65` | Evidencia |
|---|---|---|
| H1 — autoridad fresh incompleta | **Cerrado** | `FRESH_PROSPECTIVE` es reservado y terminalmente rechazado con `FRESH_PROSPECTIVE_NOT_AUTHORIZED_V1` antes de crear input package, output, stage o worker; no hay flag ni fallback (`PLAN...:145-164`). La suite exige rechazo antes de crear paths (`:759-760`). |
| H2 — invariantes cross-array incompletos | **Cerrado** | Endian, dtypes, shapes y anchos están congelados (`:166-188`); cardinalidad/target, identidad, rango y vocabulario están definidos (`:190-225`); el eje `[17,29,43]`, hashes productores y ensemble bit-exacto se validan dos veces (`:227-242`). |
| M1 — `authorized_rows` y thresholds por canal outcome-aware | **Cerrado** | Proposer divide `selection_key_metadata`/`apply_metadata`; evaluator recibe masks pero no thresholds, logits, scores ni state, computa `authorized_rows` desde `override.sum()` y devuelve keys mínimas; freeze recupera thresholds del objeto target-blind y reaplica bit a bit (`:302-350`). |
| M2 — identidad y comparadores R564 | **Cerrado** | Source freeze nomina commit/informe y manifests R564 con SHA-256 (`:483-512`); comparadores por clase son exactos y prohíben un `allclose` global (`:694-710`). Las cuatro identidades declaradas volvieron a coincidir con Git y los archivos reales. |
| M3 — commit protocol, recovery y layout | **Cerrado** | Input inmutable está separado de ambos outputs (`:525-558`); estados, enum, preparación durable, sibling pending, `fsync`/`os.replace`, tabla state×artifact y recovery están fijados (`:579-670`). |
| M4 — receipts de campañas | **Cerrado** | Unit, primary/replay check, mutaciones, recovery y evidence manifest tienen paths y contenido obligatorio; `--evidence` valida cobertura final (`:809-837`). |
| M5 — runtime/env/capabilities | **Cerrado** | El stage tiene allowlist exacta, modos, `python -s -P`, env desde cero y receipts de módulos; UID/GID, grupos, NNP y las cinco capability sets están prescritos (`:412-481`). El probe descrito abajo confirmó que la lista corregida importa. |
| L1 — budgets auxiliares | **Cerrado** | Hay wall budgets separados para primary/replay, checker+mutations, unit/permissions y recovery; límites de scratch/evidencia/archivo y medición previa a cleanup (`:841-861`). La cota RSS ahora alcanza cada proceso o harness. |

## Cierre de los tres findings de R570

### R570-H1 — cierre importable del runtime staged

**Cerrado.** La allowlist agrega
`geometria_proporcional/wave49_schema.py`, conserva el initializer que lo
importa, eleva correctamente el inventario a seis blobs y liga initializer más
dependencias W49/W53/W54 en el source freeze (`PLAN...:457-500`). La inspección
estática de imports relativos encontró sólo:

- `__init__.py → wave49_schema`;
- `proportional_set_valued_native.py → wave53_uncertainty, wave54_joint_set`;
- `wave54_joint_set.py → wave53_uncertainty`.

Todos esos módulos están en la lista exacta.

El probe CPU construyó un runtime temporal con los cinco módulos de package ya
existentes, `root:root/0444` y directorio `0555`; ejecutó desde un `cwd` de
stage mediante el Python del venv con `-s -P`, environment creado con `env -i`,
`PYTHONNOUSERSITE=1`, `PYTHONPATH` único al runtime, los cuatro límites de
threads en 1 y `CUDA_VISIBLE_DEVICES=''`. Se aplicó además el comando prescripto:

```text
setpriv --reuid=65534 --regid=65534 --clear-groups --no-new-privs \
  --bounding-set=-all --inh-caps=-all --ambient-caps=-all
```

Resultado observado:

```text
import_ok = true
Uid/Gid = 65534
Groups = empty
NoNewPrivs = 1
CapInh/CapPrm/CapEff/CapBnd/CapAmb = 0000000000000000
torch_loaded = false
NumPy/SciPy/sklearn = 2.3.5 / 1.17.0 / 1.8.0
```

Los cinco módulos del package cargados resolvieron exclusivamente desde el
runtime temporal. `_proportional_set_valued_phase_worker.py` aún no existe,
como corresponde a la superficie futura de implementación (`PLAN...:868-881`);
por eso el probe acredita el cierre real de dependencias disponible, no la
ejecución anticipada del worker futuro. La aceptación exige repetir el probe
con los seis blobs una vez implementado.

### R570-M1 — metadata pública estrecha hacia `EVALUATION_TRUTH`

**Cerrado.** `EVALUATION_APPLY` crea `evaluation_metadata.npz` con keys exactas
`pair_token <U64`, `design_stratum <U16` y `cardinality <i8`, sin logits ni
targets, y el action freeze liga el bundle y su correspondencia exacta con
`evaluate_public` (`PLAN...:353-369`). `EVALUATION_TRUTH` recibe explícitamente
ese bundle, no recibe modelos/logits/scores/thresholds, verifica orden e
identidad de tokens y contrasta cardinalidad con `target.sum` (`:372-380`). La
matriz de stages conserva la misma frontera (`:397-404`) y las mutaciones
cubren alteración de token/stratum/cardinality y adición de keys de logits o
target (`:782-790`).

El conjunto es suficiente para la recomposición: `pair_token` liga la unidad
de bootstrap; `cardinality` reproduce los slices de sensibilidad R564;
`design_stratum` preserva el estrato público para patterns y verificaciones;
las masas, acciones, masks y soportes —incluidas las tres sensibilidades por
checkpoint— ya fueron materializadas target-blind por `EVALUATION_APPLY`. El
evaluator con truth necesita entonces sólo targets, utilidades y esos arrays;
no necesita volver a recibir `ensemble_logits`, `per_seed_logits`, estados ni
thresholds.

### R570-L1 — cobertura RSS de harnesses

**Cerrado.** La cota es ahora `RSS máximo por cada proceso o harness de
cualquier campaña = 1.5 GiB` (`PLAN...:841-853`). Los harnesses miden wall/RSS
y disco, incluso `failures/`, antes del cleanup; los receipts preservan las
mediciones y cualquier excedencia invalida el preflight (`:854-861`). P15
incluye explícitamente wall, RSS y disco (`:722-738`).

## Controles transversales revalidados

- `cc19b65` es un commit plan-only: modifica sólo el objeto auditado respecto
  de `86f3bdb`.
- El plan tiene exactamente **934 líneas** y se leyó linealmente hasta la línea
  final; no se auditó sólo el diff.
- El SHA-256 del plan vigente es
  `0256305424b1a80bf6dada54aad5d3e7c392150141ba093e391ce7ab7b12dcdb`.
- Las referencias R564 siguen resolviendo exactamente: commit
  `f7ad9227868f83f381ebbc0a8995fefa5a1a272f`, informe
  `2221b3938b03728e28133ac0ac5b05918c56b5a62845966293ac113aea4479cb`,
  manifest primario
  `a0401834c3958680ef687ad264b8b56a017a8996eaade904873b342319528a39` y
  replay `2752c103f80ae8655747cf92709fe9462ef753dd282c20f449694a90b6e44039`.
- Fresh permanece rechazado antes de cualquier superficie; el único status
  ejercitable es abierto/no prospectivo (`PLAN...:121-164`, `:921-934`).
- Primary/replay consumen el mismo input inmutable y comparan exactamente los
  artefactos científicos, con exclusiones por campo y no por directorio
  (`:525-558`, `:675-710`).
- El checker no importa coordinador, worker ni core; recompone matemática y
  estado desde una copia propia del contrato (`:712-742`).
- Mutaciones cubren autoridad, schemas, permisos, channels, metadata,
  ciencia, recovery, replay e inventario (`:744-808`).
- No apareció una vía nueva para transportar truth hacia una fase pública ni
  logits/state/thresholds hacia el evaluator con truth.

## Comprobaciones realizadas

1. Lectura completa de las 934 líneas del plan vigente en tres tramos
   consecutivos.
2. Verificación de commit, parent, diff plan-only, conteo y SHA-256.
3. Contraste individual con los ocho findings de R569 y los tres de R570.
4. Revalidación de commit/informe/manifests R564 contra Git y disco.
5. Inspección estática del grafo de imports del runtime staged.
6. Probe CPU aislado del runtime bajo `setpriv`, `-s -P`, env cerrado, threads
   acotados y `CUDA_VISIBLE_DEVICES=''`.
7. Contraste de inputs usados por la recomposición/sensitivity R564 con el
   nuevo handoff de metadata y arrays target-blind.
8. Revisión macro de autoridad, schemas, stages, permisos, commit protocol,
   recovery, replay, checker, mutaciones, receipts, budgets y claims.

No se ejecutó el runner físico futuro, no se editó plan ni implementación y no
se consultó ni utilizó GPU/CUDA.

## Condición para la auditoría de implementación

La implementación deberá materializar literalmente estas allowlists y schemas
y repetir el probe con el worker real. El PASS del plan no sustituye la
auditoría posterior de código, artefactos, receipts, primary/replay,
mutaciones, recovery ni documentación que el propio orden de construcción
exige.
