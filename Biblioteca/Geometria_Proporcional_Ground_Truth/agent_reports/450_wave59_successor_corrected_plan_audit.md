# Wave 59 replay-normalized successor corrected plan audit R450

**Plan commit:** `c30fdd4e1758638107f6bc788b0a24385ae468bd`  
**Plan SHA-256:** `ad8ad47f3cbb7107e583fcd82e2cdafcde3620d1d1e898452d3075b2f861f222`  
**R449 commit:** `5f0a6af86d5f3e4bb4e4bc21115117d5b22054d8`  
**R449 report SHA-256:** `f5a87a27d0828443dc6083419cfd01abc9f4bfcdc0c134ecee70b061b3f7f95d`  
**Rejected implementation commit:** `f43507a172b88f1dfd9b4406cdc038257da14b00`  
**Result:** `REVISE`

## Dictamen: REVISE

El plan corrige de forma realizable los dos findings de R449 y construye una genealogía nueva sin reescribir historia. Sin embargo, deja fuera del cierre detallado contradiction-safe a un eslabón que ahora vuelve a ser futuro y material: la nueva auditoría de este mismo plan.

### BLOCKER — La nueva auditoría del plan permanece expuesta al parser débil

La cadena corregida exige:

1. plan revisado;
2. nueva auditoría independiente;
3. implementation commit hijo directo de esa auditoría.

Así consta en `WAVE_59_REPLAY_NORMALIZATION_SUCCESSOR_PLAN.md:297-319`. Esa auditoría será luego consumida por `successor_authority.plan_audit`.

El código vigente valida ese artefacto mediante `_require_unique_report_lines()` (`prepare_wave56_fresh.py:1169-1201`). Dicho helper sólo busca líneas individuales y no verifica el bloque terminal ni rechaza un dictamen contradictorio (`prepare_wave56_fresh.py:1113-1123`). Por tanto, una nueva auditoría con los campos superiores y `Result: PASS`, pero con `Final decision: REVISE` o un encabezado adicional `## Dictamen: REVISE`, podría autorizar el implementation commit correctivo.

La revisión declara en términos generales un “parser canónico único para las auditorías futuras” (`WAVE_59_REPLAY_NORMALIZATION_SUCCESSOR_PLAN.md:419-426`), pero sus requisitos normativos y su matriz de pruebas enumeran solamente las auditorías de implementación y config (`:176-179,233-241,278-290`). Un implementador podría cumplir literalmente esos apartados dejando intacto el parser débil de `plan_audit`.

Corrección requerida:

- Incluir explícitamente la nueva auditoría del plan en el mismo régimen canónico fail-closed.
- Definir su bloque inicial esperado —incluidos los anchors de plan, R449 y el implementation commit rechazado— y exigir coherencia única entre `Result`, `Dictamen` y `Final decision`.
- Agregar negativos para decisión terminal contradictoria, `PASS` sólo en prosa, campos ausentes/duplicados y coexistencia de dictámenes.
- Sustituir también para `plan_audit` el uso de `_require_unique_report_lines()` por el parser estricto común.

### Superficies que sí cierran

- La auditoría futura de implementación queda definida con orden canónico, los cinco hashes, `Result: PASS`, un único `Dictamen: PASS` y decisión terminal coherente (`plan:233-241`).
- La auditoría final de config recibe el mismo cierre contradiction-safe (`plan:278-295`).
- El replay fresco fallido queda separado de recovery mediante autoridad real y una prueba tardía que exige `recovery_context=false`, ausencia de amendment y un inventario correcto (`plan:180-183,419-423`).
- La genealogía es viable sin reescritura: `c30fdd4` modifica sólo el plan y es hijo de `5f0a6af`; este último agregó sólo R449 y es hijo de `f43507a`. Una auditoría nueva puede ser hija de `c30fdd4` y el implementation commit correctivo, hijo directo de aquélla.
- `f43507a` y R449 permanecen como antecedentes rechazados y quedan excluidos expresamente de `successor_authority`, `implementation_binding` y el futuro `source_sha256` (`plan:321-325`).
- El delta de fuentes continúa siendo compatible con el validator vigente: cuatro reemplazos de blobs preexistentes, alta separada del test recovery, nuevas autoridades y conservación del resto (`plan:250-276`).
- Los cinco blobs ejecutables actuales todavía coinciden exactamente con `f43507a`; volver a modificar los cinco paths en el commit correctivo permite que la config futura ligue únicamente los nuevos blobs.
- La matriz de tests es suficiente para los findings de R449 una vez incorporada la cobertura de la nueva auditoría del plan.

### Evidencia operativa

- `HEAD=c30fdd4e1758638107f6bc788b0a24385ae468bd`.
- Worktree limpio.
- `git diff --check c30fdd4^ c30fdd4` termina con exit `0`.
- Se leyeron completos el plan vigente —426 líneas— y R449 —62 líneas—.
- Se contrastaron los cinco paths declarados y su identidad Git respecto de `f43507a`.
- No se editaron archivos ni se ejecutaron tests, draw, recovery o preparación.
- No se abrieron datos sellados y no se usaron GPU, web ni Mendieta.

Este dictamen no constituye `GO/NO-GO` científico.

## Machine-verifiable decision

**Final decision:** `REVISE`
