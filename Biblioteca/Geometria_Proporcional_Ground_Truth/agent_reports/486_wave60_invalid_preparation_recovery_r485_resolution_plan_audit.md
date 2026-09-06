# R486 — Auditoría independiente del plan de resolución de R485

**Dictamen técnico: `REVISE` — 0 HIGH / 2 MEDIUM / 0 LOW.**

El plan resuelve correctamente el diseño de mutaciones aisladas, la divergencia blob Git/archivo físico, la partición de los sources científicos, el keyset de 31 claves y la lineage futura. Sin embargo, todavía no puede autorizarse: omite R486 de la matriz exhaustiva de auditorías que ella misma incorpora al sufijo y formula una aserción de provenance incompatible con la estructura actual de la attestation y con el alcance productivo permitido.

## Identidad y contraste del target

El commit abreviado solicitado resuelve a `0c07b10abf7831dc6577c39637f3b68c2a3a02b2`. Es hijo directo de R485 `b038cf91fffe66689b103e1b3a28503b6975d645` y modifica exclusivamente `Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_60_INVALID_PREPARATION_RECOVERY_R485_RESOLUTION_PLAN.md`. El SHA-256 físico y del blob Git es `cbbdbf361deaa54d9bf2461aceda7bef54412bb1fa40d821a974f2fb02520e4a`.

Leí completos el plan auditado, R485, la resolución R483 y R484, y contrasté las funciones y pruebas vigentes del sufijo. La historia inmediata también es exacta: `f32ba2b` desciende de R484 y cambia sólo preparador/test; R485 desciende de `f32ba2b`, es un commit exclusivo de informe y tiene SHA-256 físico/Git `d58148ce731c3d06e61101f44dd7db5165b9932ce47c8acc85b45383ffe15650`.

## Aspectos técnicamente correctos

### Aislamiento de mutaciones y divergencia blob/físico

El método de auditorías alternativas es realizable. Para mutar un campo interno de `scope`, `target`, `technical_verdict` o `findings`, puede crearse un reporte hermano desde el parent requerido, commitear sólo su path y actualizar `commit/path/sha256` del binding externo, manteniendo intacto el objeto externo semántico. Las precondiciones exigidas por el plan hacen que `validate_wave60_bound_document()` o `validate_wave60_audit_commit()` superen primero existencia física, hash, introducción, exclusividad y parent, y que el rechazo llegue después al parser semántico (`WAVE_60_INVALID_PREPARATION_RECOVERY_R485_RESOLUTION_PLAN.md:125-160`; `prepare_wave56_fresh.py:420-507`).

La divergencia A/B también alcanza el guard correcto: con el path introducido exclusivamente por el commit ligado, se puede dejar blob Git A, escribir físicamente B sin commitearlo y declarar SHA(B). Así `require_repo_artifact()` acepta el archivo físico y la comprobación posterior `git_blob_sha256(...) != binding.sha256` produce el rechazo, sin depender de path ausente, hash físico erróneo ni parent inválido (`WAVE_60_INVALID_PREPARATION_RECOVERY_R485_RESOLUTION_PLAN.md:162-180`; `prepare_wave56_fresh.py:460-507`). Esta construcción debe restaurar el archivo físico después de cada caso, pero no requiere cambiar producción.

### Exclusividad y sources científicos

La corrección de commits con path adicional evita el falso positivo observado por R485: cada commit malo nace de su parent correcto y agrega exactamente un path al delta propio de su clase, sin reutilizar el tree acumulado de una revisión posterior (`WAVE_60_INVALID_PREPARATION_RECOVERY_R485_RESOLUTION_PLAN.md:182-194`).

La cadena científica secundaria también es ejecutable. Un commit sintético entre R475 y el anchor R480 puede alterar exactamente uno de los tres sources; desde R480 en adelante se reconstruye el sufijo con parents directos y deltas exclusivos de cada clase. La mutación se hereda hasta la implementación final, cuyo commit continúa cambiando sólo preparador/test, de modo que la comparación nueva entre `git_blob(final_implementation, source)` y `git_blob(R475, source)` es el primer hecho inválido relevante (`WAVE_60_INVALID_PREPARATION_RECOVERY_R485_RESOLUTION_PLAN.md:196-225`). Esto prueba contenido real y no sólo el keyset de `unchanged_source_law_sources`.

### Keyset, nombres, numeración y ciencia

Las 27 claves aprobadas por R483 más `r483_resolution_implementation`, `r483_resolution_implementation_audit`, `r485_resolution_plan` y `r485_resolution_plan_audit` dan exactamente 31. Los nombres conservan sin colisión las tres implementaciones incompletas y sus auditorías `REVISE`, mientras `recovery_implementation` queda reservado al sucesor aceptado por R487 (`WAVE_60_INVALID_PREPARATION_RECOVERY_R485_RESOLUTION_PLAN.md:46-108`).

La secuencia R485 → plan → R486 → implementación → R487 → amendment → R488 → config → R489 es acíclica y los IDs R486–R490 no se reutilizan (`WAVE_60_INVALID_PREPARATION_RECOVERY_R485_RESOLUTION_PLAN.md:51-71,264-300`). El alcance productivo se limita a guards de lineage en el preparador y a la suite; módulo, runner y worker permanecen byte-exactos bajo R475, con hashes `46e31fa…`, `1c778c3e…` y `c6c5c832…` (`WAVE_60_INVALID_PREPARATION_RECOVERY_R485_RESOLUTION_PLAN.md:227-262`). No hay ampliación científica.

## Finding MEDIUM 1 — la matriz exhaustiva omite la nueva auditoría R486

El plan añade `r485_resolution_plan_audit` como uno de los cuatro objetos nuevos y lo define como R486 PASS (`WAVE_60_INVALID_PREPARATION_RECOVERY_R485_RESOLUTION_PLAN.md:73-92`). Sin embargo, la matriz que dice cubrir “cada auditoría” enumera exactamente R481, R482, R483, R484, R485 y R487: R486 no aparece (`WAVE_60_INVALID_PREPARATION_RECOVERY_R485_RESOLUTION_PLAN.md:125-145`). La omisión alcanza tanto los campos internos `audit_id/scope/target/technical_verdict/findings` como sus bindings `commit/path/sha256`.

Esto deja sin prueba específica precisamente una autoridad nueva que el sucesor deberá autenticar entre este plan y la implementación final. La cadena puede validar R486 positivamente y aun así repetir para ella la cobertura parcial que R485 observó en auditorías anteriores.

**Corrección concreta:** añadir R486 a la enumeración de §4 y exigir los mismos negativos aislados de campos semánticos y bindings externos. Cada reporte alternativo R486 debe ser hijo directo del commit de este plan, cambiar exclusivamente el path canónico de R486, actualizar coherentemente el binding físico y comprobar como precondición parent, delta, path y SHA antes de invocar el parser/validador.

## Finding MEDIUM 2 — la provenance no aparece como objeto dentro de la attestation vigente

El plan exige que la provenance completa aparezca “idénticamente” en generation receipt, preparation freeze, preparation receipt **y attestation** (`WAVE_60_INVALID_PREPARATION_RECOVERY_R485_RESOLUTION_PLAN.md:110-123`). Esa exigencia literal no es realizable con el contrato actual: `publish_wave60_preparation_attestation()` incluye un mapa `records` con path, bytes y SHA-256 de `preparation_receipt.json`, pero el payload no contiene una clave `recovery_provenance` (`prepare_wave56_fresh.py:6125-6178`). La prueba vigente compara el objeto en los tres artefactos y luego demuestra que la attestation válida firma el hash del receipt físico (`tests/test_wave60_frozen_policy_transport.py:2136-2165`); no compara una provenance inexistente dentro de la attestation.

El propio alcance del sucesor sólo permite cambios productivos de historia, keyset, blobs científicos y numeración (`WAVE_60_INVALID_PREPARATION_RECOVERY_R485_RESOLUTION_PLAN.md:227-245`). Cumplir la frase literalmente exigiría ampliar silenciosamente el schema firmado, mientras ignorarla incumpliría el plan.

**Corrección concreta:** reemplazar la obligación por: “la provenance completa debe ser idéntica en generation receipt, preparation freeze y preparation receipt; la attestation debe verificar criptográficamente y ligar por path/bytes/SHA-256 ese `preparation_receipt.json` físico”. Mantener además `implementation_audit.audit_id=R487`. Esto conserva la firma transitiva ya aprobada y no requiere modificar producción.

No se ejecutaron suites largas porque los findings son contradicciones del contrato escrito frente a las APIs inspeccionadas. Todo el trabajo fue CPU-only; no se usó ni consultó GPU. No se modificó código ni otro documento.

```json
{
  "schema_version": "wave60-audit-authority-v1",
  "audit_id": "R486",
  "scope": "INVALID_PREPARATION_RECOVERY_R485_RESOLUTION_PLAN",
  "target": {
    "plan_commit": "0c07b10abf7831dc6577c39637f3b68c2a3a02b2",
    "plan_sha256": "cbbdbf361deaa54d9bf2461aceda7bef54412bb1fa40d821a974f2fb02520e4a"
  },
  "technical_verdict": "REVISE",
  "findings": {
    "high": 0,
    "medium": 2,
    "low": 0
  },
  "files_modified": false,
  "gpu_used_or_queried": false
}
```
