# R401 — Auditoría focal final del plan de recuperación pre-oracle de Ola 57

**Plan commit:** `d21f3c96e0313077f8e121dab731c9f8a7a93320`
**Plan SHA:** `8485b07e8632f8a247f68db5eadabdc90dfe9944479a274d25fdb01a36165d52`
**Result:** `PASS`

## Dictamen ejecutivo

La revisión final cierra la contradicción material identificada por R400. El
preflight content-blind ya no prohíbe toda apertura de los archivos sensibles:
autoriza una única ruta física, `O_NOFOLLOW → fstat → streaming binario
exclusivo a SHA-256`, y prohíbe separadamente todo parsing, interpretación,
conversión o helper semántico antes de completar la autoridad. La matriz de
tests y el probe real deben distinguir explícitamente esas dos capacidades.

No queda en el plan una vía semántica pre-authority. Escrow, secret files y
truth sellada sólo pueden contribuir bytes opacos al hash durante el preflight;
`read_escrow`, `keys_from_escrow`, `validate_semantic_attestation`,
`sealed_population_counts`, `read_text`, `read_jsonl`, parsers JSON y
`bytes.fromhex` quedan bloqueados sobre esos paths. El primer acceso semántico
ocurre después de autenticar amendment, DAG Git, HEAD, worktree, contrato
público, source delta, inventario, manifest y firma detached.

Los dos findings de R399 y el finding residual de R400 están cerrados sin
regresión de no-redraw, poblaciones, source contract, cadena de autoridad,
separación Wave 56/Wave 57 ni replay. El plan puede pasar a implementación; este
dictamen no autoriza por sí solo recovery físico, inferencia, oracle, labels ni
una decisión científica GO/NO-GO.

## Findings priorizados

### ALTA

Ninguno.

### MEDIA

Ninguno.

### BAJA

Ninguno que requiera revisar el plan. La auditoría de implementación deberá
comprobar literalmente que los spies son path-aware y distinguen el lector
binario opaco de cualquier consumidor semántico; ésta es una obligación ya
predeclarada, no una deuda del diseño.

## Cierre exacto de R400

### Capacidad física content-blind — CERRADO

El plan define para los archivos sensibles una sola lectura permitida:

- `lstat` para identidad física;
- apertura binaria con `O_NOFOLLOW`;
- comprobación estable con `fstat`;
- streaming por bloques dirigido exclusivamente a SHA-256.

La regla aparece en la definición normativa del preflight
(`WAVE_57_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:94-102`), se vuelve obligatoria
para los hashes de inventario y manifest (`:111-115`), forma parte del check 7
de autoridad (`:139-161`) y se exige tanto en tests como en el probe canónico
(`:197-229`). Ya no hay una frase posterior que prohíba esa lectura física.

El mecanismo es implementable con el helper físico existente, que abre con
`O_RDONLY | O_NOFOLLOW`, contrasta `lstat`/`fstat`, hashea por bloques y vuelve a
contrastar identidad después del cierre
(`experiments/geometria_proporcional/prepare_wave56_fresh.py:278-321`). El plan
no obliga a reutilizar un helper semántico para producir el hash.

### Consumidores semánticos pre-authority — CERRADO

La prohibición ya se formula sobre operaciones semánticas, no sobre el syscall
de apertura. Antes de conceder autoridad no se puede usar `read_text`,
`json.load`/`json.loads`, `read_jsonl`, `bytes.fromhex` ni otro parser o
conversor sobre escrow, secret files o truth sellada
(`WAVE_57_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:102-109`). Tampoco se puede
invocar `read_escrow`, `keys_from_escrow`, `validate_semantic_attestation`,
`sealed_population_counts` ni un equivalente que cargue truth (`:88-115`).

Los negativos exigen que los spies permitan el lector opaco y fallen ante cada
consumidor nombrado o ante un parser sobre cualquier path sensible
(`WAVE_57_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:197-216`). El probe contra el
origen real debe demostrar ausencia de parsing, interpretación o extracción,
pero aceptar el streaming exclusivo al hasher (`:224-229`). Esto proporciona un
oracle de test único y elimina la incompatibilidad señalada en R400.

### Orden de autoridad — CERRADO

El preflight puede leer semánticamente sólo artefactos públicos: `FAILURE.json`,
manifest, visibles, atestación detached y freeze pre-generación secret-free. Con
ellos valida la cadena Git, HEAD, worktree, contrato público, delta de fuentes,
inventario y firma mediante la clave pública (`:94-115`). Sólo después abre el
escrow, contrasta contrato y commitments, valida la atestación completa y
calcula poblaciones sobre los JSONL sellados (`:117-125,163-166`). El plan no
concede inferencia, oracle analítico ni labels autorizados en esa segunda etapa.

## Cierre de los findings R399

### Separación pre-key/semántica — CERRADO

Además del orden anterior, el plan identifica expresamente que el helper
histórico no es pre-key porque extrae claves y carga truth, y prohíbe usarlo en
la autenticación inicial
(`WAVE_57_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:86-92`). La implementación Wave
57 debe despacharse por `prospective_config.schema_version`, conservando la
autoridad histórica Wave 56 (`:127-137,197-216`).

### Oracle materializado y labels autorizados — CERRADO

El plan distingue la truth sellada esperada de sus derivados analíticos y
declara que las ausencias se refieren únicamente al oracle analítico
materializado y a labels autorizados
(`WAVE_57_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:52-61`). Las aserciones del
amendment son ahora `no_materialized_oracle_in_origin` y
`no_authorized_labels_in_origin` (`:139-146`), coherentes con los negativos y el
probe (`:197-229`).

## Regresiones revalidadas

- **Identidad del plan.** `HEAD` es exactamente
  `d21f3c96e0313077f8e121dab731c9f8a7a93320`; el commit modifica sólo el plan y
  su SHA-256 coincide con el bloque inicial. Su padre directo es el commit que
  agregó R400, por lo que la revisión final comienza una cadena nueva y limpia
  `plan → reauditoría`.
- **Estado de implementación.** Los blobs de preparador, test Wave 57 y config
  conservan respectivamente `796b5e8f580c...`, `5238778afb00...` y
  `fb21a43cb4e3...`; esta revisión no adelantó implementación ni cambió el
  contrato científico.
- **No-redraw.** Se conservan el mismo escrow, las mismas claves y el manifest
  exacto; recovery debe hacer imposible `secrets.token_bytes` y cualquier fallo
  preserva el intento sin habilitar un draw nuevo
  (`WAVE_57_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md:63-84,217-222,231-255`).
- **Poblaciones.** Continúan congelados `4992` fixtures, `1152` tokens totales y
  `768` elegibles por split, con filtrado por fila antes de deduplicar y sin
  relajar mínimos (`:8-31,63-84,217-220`). Los conteos se calculan sólo en la
  etapa semántica ya autorizada (`:117-125`).
- **Source delta.** Sólo pueden cambiar el preparador compartido y
  `tests/test_wave57_prospective.py`; config, bindings, inventario de fuentes y
  todos los demás campos permanecen idénticos salvo `git_commit` y esos dos
  hashes (`:152-159,190-222`).
- **Cadena Git.** Se mantiene la DAG lineal de seis commits, cada uno limitado a
  sus paths, con final-audit como HEAD exacto y worktree limpio antes de recovery
  (`:139-172,174-188`).
- **Replay.** Debe partir del primario canónico y de la misma copia del amendment
  y conservar exactos escrow, freeze, benchmark visible, logits y provenance
  (`:160-166,217-222,241-255`).

La reauditoría no abrió escrow, secret files ni truth sellada; no ejecutó
inferencia, oracle, materialización de labels, training ni GPU/CUDA.

## Veredicto

**PASS.** La frontera pre-key es ahora inequívoca, implementable y testeable: el
hash opaco conserva integridad física sin conceder interpretación semántica, y
el primer acceso a claves o truth ocurre sólo después de autenticar toda la
autoridad. No quedan findings materiales en el plan vigente.

## Machine-verifiable decision

**Final decision:** `PASS`

