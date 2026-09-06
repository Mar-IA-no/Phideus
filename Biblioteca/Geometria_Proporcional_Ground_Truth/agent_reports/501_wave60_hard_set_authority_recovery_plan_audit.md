# R501 — Auditoría del plan de recovery de autoridad `hard_set_tau` de Ola 60

**Dictamen técnico: `PASS` — 0 HIGH / 0 MEDIUM / 0 LOW.**

El plan corregido resuelve el único finding preliminar y define un recovery v4
autenticable, acumulativo y sin redraw ni overwrite. La autoridad del draw
permanece en el escrow originado en v1; la autoridad temporal del delta de
código queda separada y anclada a la config v3 aceptada por R500. El recovery
no altera la pregunta científica ni autoriza scoring anticipado, adaptación o
una realización independiente adicional.

## Target

El target auditado es exactamente:

- commit: `a9bb0cfd1062da5076f8cbdd8c2d64978270dce7`;
- parent directo: `b53349a764840bb49a8a9c1de5b445dade87f44f`;
- único path modificado:
  `Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_HARD_SET_AUTHORITY_RECOVERY_PLAN.md`;
- SHA-256 del blob Git y del archivo físico:
  `6a4e95e4373f5fc5fa1c40e31e9efb090483c986b69af4d48c9738918fc5472c`.

El commit es exclusivo y `git diff --check` queda limpio. Leí completos el plan
corregido, su versión previa, la config v3, R500, el amendment v3, los
validadores de recovery y presupuesto, el materializador real y los artefactos
físicos v2/v3 pertinentes.

## Resolución de R501-01

La nueva sección de doble procedencia elimina la ambigüedad entre escrow y
baseline de código. Exige que `prior_source_sha256` sea idéntico, clave por
clave, al mapa `source_sha256` de la config v3/R500. Los `old_sha256` de
preparer y test provendrán exclusivamente de ese mapa; los `new_sha256` deberán
coincidir con implementación, config v4 y filesystem.

La partición queda cerrada sobre los ocho sources vigentes:

- cambian respecto de v3 sólo preparer y test;
- permanecen invariantes runner, módulo científico, worker y los informes R475
  y R476;
- la config cambia únicamente mediante su self-binding y autoridad v4.

El contraste con el escrow físico confirma la necesidad y exactitud de esta
regla: fuera del source de config, `origin_contract.sources` de v1 difiere de
la config v3 exactamente en runner, preparer y test. El runner v3 está fijado a
`b35cd563f715bdff9b6e7489ac04712c728673563898d4a6aebf0144d4a50261` y
su diferencia respecto de v1 no se confundirá con un cambio v4.

Los nuevos requisitos de prueba materializan expresamente ese baseline cruzado
y atacan el mapa previo, los `old_sha256`, el runner invariante y el roster.
Esto cubre tanto el camino positivo como el fallo que motivó la revisión.

## Terminal v3 y semántica de truth

La validación read-only del intento físico v3 autentica las firmas, terminales,
inventarios y manifest del par:

```text
pair terminal       PAIR_ABORTED_PRE_TRUTH
primary terminal    INVALID_PREPARATION
replay terminal     INVALID_PREPARATION
any_truth_accessed  false
recovery_allowed    true
```

El inventario físico contiene 44 registros en primaria y 8 en replay, sin
faltantes ni prohibidos. No persisten `prepared`, labels temporales,
`preparation_receipt.json` ni `preparation_attestation.json`. El error conserva
el SHA-256 exacto de `str(KeyError("hard_set_tau"))`, y el access receipt declara
inferencia sin fit, sin labels disponibles y un probe de truth rechazado con
`PermissionError`.

La descripción del plan no sobreextiende el flag firmado. El materializador
real crea temporalmente labels root-only de `train` mediante
`compute_oracle_splits()` antes de consultar `config["hard_set_tau"]`; al fallar,
su bloque de error retira por completo `prepared`. Por tanto hubo cómputo
transitorio de oracle durante preparación, pero ninguna fase de scoring o
evaluación abrió bundles de verdad, no se congelaron acciones de lockbox y no
se observaron resultados científicos. La distinción del plan es fiel al orden
real del código y a los artefactos preservados.

## Autoridad hard-set, namespace y ledger

La cadena física existente manifest → request → alias único → snapshot de Ola
59 acepta `hard_set_tau=0.5`; el snapshot tiene SHA-256
`f6edfd2106fe87c8150562d096469e29b64a108a73de2dae0d371bd689a4a9b6`.
La config canónica v3 no contiene esa clave. El plan exige resolver y validar
esta autoridad antes de materializar draw, inferencia o modificar la raíz
inicializada, y sólo entonces pasar una copia efímera de la config al
materializador. El mismo seam real deberá quedar cubierto para recovery v2 y
v4; un fake no podrá acreditar esa interfaz.

El namespace v4 es nuevo y v3 permanece sellado. El draw se recopia desde el
origen nested v3 con igualdad de bytes, mapa closed-world e inodos nuevos. La
amendment futura ligará terminal v3, config/R500, amendment/R499, terminal y
ledger v2, error y access receipt, hard-set contract, plan/auditorías y
partición de sources. Los ataques obligatorios cubren redraw, `--force`, fuente
incorrecta, inventario abierto y alteraciones de lineage.

El presupuesto tampoco duplica autoridad. Los `215.36700256168842 s` son el
acumulado firmado de v2 e incluyen el débito anterior. Como v3 no produjo
ledger firmado, v4 añade una sola vez un débito conservador de `60.0 s` ligado
al terminal v3. El inicio primario será `275.3670025616884 s`; replay heredará
después el acumulado firmado de primaria v4. El plan exige rechazo de doble
débito y respeto del límite total de 900 s. Los tiempos externos de v3 y sus
swaps quedan correctamente descritos como contexto no durable.

## Alcance y pruebas futuras

La implementación autorizada queda limitada a preparer y test. Runner, módulo
científico y worker permanecen byte-exactos. El plan exige E2E v3→v4, seam real
del materializador, resolución previa fail-closed, ataques de terminal,
inventario, firmas, source delta y presupuesto, más suite Wave 60 y regresión
Waves 56–59 con RSS y swaps.

La inspección y las validaciones read-only fueron CPU-only con
`CUDA_VISIBLE_DEVICES=''` y `PYTHONDONTWRITEBYTECODE=1`; no usé ni consulté GPU.
No modifiqué plan, código, config ni datos. Este informe es el único archivo
creado. R501 no ejecuta v4, no promueve una arquitectura y no decide
`GO/NO-GO`.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R501",
  "scope": "HARD_SET_AUTHORITY_RECOVERY_PLAN",
  "target": {
    "plan_commit": "a9bb0cfd1062da5076f8cbdd8c2d64978270dce7",
    "plan_sha256": "6a4e95e4373f5fc5fa1c40e31e9efb090483c986b69af4d48c9738918fc5472c"
  },
  "technical_verdict": "PASS",
  "findings": {
    "high": 0,
    "medium": 0,
    "low": 0
  },
  "files_modified": false,
  "gpu_used_or_queried": false
}
```
