# Wave 59 signed preparation authority plan audit R433

**Plan commit:** `b2e17a0e38d3ad64844d49208205eee7ee0aec0f`  
**Plan SHA:** `240bc92325e8ee15136b03203c65a2f79c0306719eab3f26f98f57884ff22c5c`  
**R432 commit:** `56e90a4e3262bb4efb4efa6a19902d3a8ee14b24`  
**R432 report SHA-256:** `ad98740419a175fd419aec1de3b13bdd604b4adb2331fc1be21e3651b7322461`  
**Rejected implementation commit:** `d02422f132ebfc7e07fa0e508be80a46cd76ff93`  
**Result:** `PASS`

## Scope

Se auditó la revisión completa del plan contra el P1 reproducido por R432 y contra el código vigente necesario para determinar realizabilidad. No se abrió semánticamente escrow, secrets, commitments ni truth sellada; no se ejecutó recovery, inferencia, oracle, GPU ni Mendieta, y no se modificó el worktree.

## Findings graduados

### P0 — ninguno

No se identificaron requisitos capaces de destruir la realización congelada, autorizar un redraw o anticipar acceso semántico.

### P1 — ninguno

La revisión cierra sustantivamente el P1 de R432. La autoridad del paquete deja de descansar en declaraciones internas copiables y pasa a depender de una atestación Ed25519 detached, emitida con una clave privada externa y verificada exclusivamente contra la clave pública ya congelada.

### P2 — ninguno

No se encontraron contradicciones entre el nuevo contrato firmado, las rutas físicas, la comparación primary/replay, el archivado de fallos, la cadena Git y las pruebas exigidas.

## Cierre del P1 R432

### Autoridad no autofirmada

Las líneas 270–302 introducen `preparation_attestation.json` con firma Ed25519 externa. El payload tiene schema y forma cerrados; el runner debe reconstruirlo desde los bytes físicos y verificarlo contra la clave pública del repositorio. Copiar amendment, provenance, receipts o freeze ya no basta para fabricar autoridad.

La primitiva necesaria existe y es compatible con el diseño: `sign_attestation()` comprueba que la privada derive exactamente la pública confiada, firma JSON canónico y `verify_attestation()` valida fingerprint y firma (`src/geometria_proporcional/wave49_attestation.py:52–108`). El preparador ya recibe la privada externa y valida su correspondencia durante la generación.

### Path, ownership, modos, schema y forma

Las líneas 242–268 y 309–316 exigen:

- ruta primaria o replay congelada conforme al rol;
- raíz `root:root/0700`;
- directorios `benchmark`, `inference` y `prepared` en `root:root/0700`;
- `journals` en `root:root/0755`;
- modos diferenciados para manifest, logits, bundles truth/safe y JSON públicos;
- `lstat`, rechazo de symlinks, claves exactas, schema y phase exactos.

Estas condiciones son realizables con la producción vigente: la raíz se crea en `0700`, inference y prepared en `0700`, los bundles ya distinguen `0600/0644`, y el journal se crea en `0755` (`prepare_wave56_fresh.py:2994–2998`, `3099–3115`, `3597–3633`; `run_wave59_hgb_guard_bracket.py:233–324`).

### Cobertura del payload firmado

Las líneas 280–302 ligan directamente por `path + bytes + sha256`:

- amendment;
- pre-generation freeze;
- manifest;
- generation receipt;
- preparation freeze;
- preparation receipt;
- config snapshot;
- source bindings;
- journal de preparación.

El mismo payload fija commit, rol/modo, provenance y los mapas completos de inference y bundles. El runner debe contrastar esos mapas con los archivos físicos. Así, el enlace alcanza freeze, receipts, journal, manifest, amendment, inference y bundles sin interpretar escrow ni truth.

No aparece una circularidad criptográfica: la atestación no se incluye en su propio payload, y el audit final precede a la preparación física, por lo que su hash puede entrar en provenance cuando se firma el paquete.

### Primary/replay

La revisión distingue correctamente:

- igualdad exacta para freeze, benchmark, logits, bundles, amendment y provenance;
- comparación semántica para rol, timestamps y receipts derivados;
- verificación criptográfica independiente de cada atestación.

Esto encaja con las dos capas vigentes: `compare_preparation()` ya compara escrow, freeze, manifest, logits y bundles, mientras `compare_runs()` separa `scientific_exact`, arrays, secretos por hash y artefactos operativos normalizados (`prepare_wave56_fresh.py:3281–3359`; `run_wave59_hgb_guard_bracket.py:1595–1717`). La atestación puede agregarse a `operational_semantic` sin forzar igualdad de firmas correspondientes a payloads legítimamente diferentes.

### Archivo de fallos

Las líneas 304–307 ordenan incorporar la nueva atestación a la matriz cerrada y clasificarla cuando exista. Es compatible con `_artifact_classes()`, `_failure_coverage()` y `archive_failed_attempt()`, que ya operan mediante inventarios explícitos y clases cerradas (`run_wave59_hgb_guard_bracket.py:1745–1801`, `1989–2107`, `2110–2190`).

### Cadena de commits

La cadena revisada es lineal y no reutiliza el dictamen rechazado:

1. R432 quedó en `56e90a4`;
2. el plan revisado `b2e17a0` es su parent directo y modifica sólo el plan;
3. seguirá auditoría de plan;
4. implementación corregida en cuatro paths;
5. auditoría de implementación;
6. amendment;
7. auditoría final como HEAD exacto.

El texto exige parents directos, commits de introducción pertinentes, blobs, paths exclusivos, worktree limpio y dictámenes parseables. También rechaza `PASS` incidental, decisiones contradictorias y auditorías no ancestrales.

No hay conflicto entre el amendment y el audit final: el amendment necesita predeclarar el path del audit; el audit posterior liga commit y SHA del amendment.

## Pruebas exigidas

La matriz de líneas 345–389 cubre los vectores ausentes en R432:

- aceptación directa por la rama pública de un paquete auténticamente firmado;
- rechazo de raíz alternativa, `0777`, uid/gid o modos incorrectos y symlinks;
- rechazo de schemas, phases, formas incompletas o extendidas;
- rechazo de amendment copiado con provenance falso;
- rechazo de freeze, receipt o journal desligados;
- rechazo de firma ausente, payload divergente o clave distinta;
- rechazo de mapas de inference o bundles incompatibles con los bytes;
- mantenimiento del rechazo no-recovery;
- cadena Git, HEAD, limpieza y deltas de fuentes;
- regresión Wave 56–59;
- probe content-blind sobre el origen real sin recovery ni acceso semántico.

La cobertura responde directamente a los casos forjados por R432 y agrega el caso positivo que faltaba.

## Condiciones de implementación a verificar en R434

No son defectos del plan, pero la siguiente auditoría debe comprobar especialmente:

1. `preparation_receipt.json` se modifica actualmente después de `execute_preparation()`, primero para agregar `superseded_output` y luego `coordinator_budget` (`prepare_wave56_fresh.py:3700–3703`, `3817–3820`). La firma debe emitirse después de esas mutaciones o el flujo debe reordenarse; firmar antes produciría una atestación inmediatamente obsoleta.

2. El test específico ya fue introducido en `d02422f`. La condición vigente que exige que su introduction commit sea el nuevo implementation commit (`prepare_wave56_fresh.py:1353–1358`) debe adaptarse al segundo ciclo: conservar trazabilidad de la introducción histórica, pero ligar el blob corregido y su modificación al nuevo commit.

3. La comparación semántica de attestations no debe ignorar simplemente `signature_base64`: debe verificar criptográficamente cada paquete y comparar luego los campos invariantes de sus payloads, admitiendo sólo las diferencias operativas declaradas.

4. Si la firma final ocurre después de la consolidación del receipt, cualquier fallo de firma o publicación debe entrar en el mismo archivado físico de fallos; no puede dejar una raíz canónica parcialmente preparada y sin inventario.

Estas condiciones son resolubles dentro de los cuatro paths autorizados y ya están implicadas por los invariantes y pruebas del plan.

## Verificaciones realizadas

- Plan leído completo: `462` líneas.
- R432 leído completo: `51` líneas.
- SHA del plan igual al esperado: `240bc92325e8ee15136b03203c65a2f79c0306719eab3f26f98f57884ff22c5c`.
- `HEAD` exacto: `b2e17a0e38d3ad64844d49208205eee7ee0aec0f`.
- Parent directo: `56e90a4e3262bb4efb4efa6a19902d3a8ee14b24`.
- El commit del plan cambia exclusivamente `WAVE_59_PREORACLE_PAIR_TOKEN_RECOVERY_PLAN.md`.
- `git diff --check` limpio.
- Worktree global limpio.
- Los tres hashes originales declarados por el plan coinciden con los blobs del contrato `1ef98a1`.
- El config mantiene `33` fuentes y `33` hashes; el delta autorizado continúa limitado a tres fuentes, con `30` invariantes.
- Los cuatro SHA-256 vigentes coinciden con R432:
  - preparador: `ef210e3fd02965dc3e0d270d19623ca0fac35cd56a57d61c51d15e36230b0881`;
  - runner: `58fcd7c3d23facbf68a26a073f0fd7dfe51b6364b2fe539e4a9047ea6512ae73`;
  - test prospectivo: `d728e535cd867c6ebc86c7fcad9d0b6470166e7445b28e6bc8ea0acea701abdb`;
  - test recovery: `40f85b07d6c4e36539b4fb745a6265f8c84c424218c22e0a31ee02265d867c04`.
- No se ejecutaron tests que pudieran iniciar recovery; esta auditoría fue estática y CPU-only.
- No se abrió semánticamente material sellado ni se usó web, GPU o Mendieta.

**Final decision:** `PASS`
