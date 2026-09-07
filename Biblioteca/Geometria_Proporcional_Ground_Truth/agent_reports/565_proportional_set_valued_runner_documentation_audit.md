# R565 — Auditoría independiente de cierre documental del runner set-valued

**Fecha:** 2026-09-07  
**HEAD auditado:** `22aff27d6ab60f8ecec5a83cf8507e7967b10d76`  
**Commit de evidencia del cierre:** `b27c044bd490495a335b3b6c3469fc662d0f6abd`  
**Régimen:** CPU exclusivamente; `CUDA_VISIBLE_DEVICES=''` y un thread para BLAS/OpenMP; no se consultó ni utilizó GPU/CUDA.  
**Escrituras:** únicamente este informe; no se modificó documentación pública ni implementación.

## Dictamen

**REVISE — 0 HIGH, 1 MEDIUM, 1 LOW.**

Los siete artefactos modificados por `22aff27` son internamente coherentes con el cierre técnico R564: números, regímenes de evidencia, limitaciones de soporte, próximo relevo y frontera de claims coinciden con las fuentes. No hay promoción de arquitectura, decisión `GO/NO-GO`, reapertura del draw ni confusión entre el diagnóstico sobre histórico abierto y evidencia prospectiva.

El cierre documental no está, sin embargo, sincronizado en toda la superficie pública vigente. Tres páginas `current` de la wiki todavía presentan como próximo paso implementar el runner ya cerrado. Además, la lectura completa de la bitácora volvió visibles dos rutas `/tmp` concretas y una referencia al runtime privado de una skill; son residuos históricos, no defectos introducidos por este commit, pero incumplen la política pública de paths y capas internas.

## Alcance leído completo

Se leyeron completos, no por snippets, los seis documentos públicos modificados:

| Documento | SHA-256 |
|---|---|
| `Documents/00_TRONCAL/Proyecto_Estado_Actual.md` | `b7f93133a236d9caa03efcd974a65dff5b54a0cb388735942c69431abc859cd7` |
| `Documents/00_TRONCAL/bitacora_desarrollo.md` | `15ba31ff63e338e5e37ee2826e9de5a1c9912d154adec85bf2fbac03d4147ba5` |
| `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/CATALOGO_NARRATIVO_DESCRIPTORES_RATIOS_PHIDEUS.md` | `7cb0a55b6fc2da720e4a7c757d115f67fe6f15156b83b57eb0dae996d5c08fe2` |
| `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md` | `21ecc28f545474e0653abb3bd6a11cbae91aa000cadd720e9d33d6b3876baa95` |
| `Documents/05_WIKI/LLM_CONTEXT.md` | `e090cddd439e01454cee81f86e01db25a7fa422ec07c5177ae14a4c76ecfcafc` |
| `Documents/05_WIKI/roadmaps/proportional-architecture-experiments.md` | `9084fac8e4094806d0b3e55d006db08c66c6a2eb1580a6877bee2b1758d0faaf` |

También se leyó completo el catálogo generado `Documents/05_WIKI/catalog.json`, SHA-256 `83ab92cd24caad61c095061f4c7c666b9981c65e82ff3e8269568de6c3fa7e50`, y las cuatro fuentes de cierre:

| Fuente | SHA-256 |
|---|---|
| `PROGRAM_CLOSURE_SET_VALUED_NATIVE_RUNNER_PREFLIGHT.md` | `b63c7ba1165f83bdc07e1a944efa24b4afab9f0a178ef4352331f51f0f261dad` |
| R562 | `e7163a4c48aba9fad2403a0b9bf072fd5942c020f8a1141dd3f80d2903a62842` |
| R563 | `4023cd306126765f2d1c9a7dba3df4afea1d7ab3a89e4c43c899d08a7eac2733` |
| R564 | `2221b3938b03728e28133ac0ac5b05918c56b5a62845966293ac113aea4479cb` |

Para comprobar sincronización transversal se revisaron además las secciones pertinentes y los metadatos completos de las tres páginas `current` citadas en F-01.

## Findings

### F-01 — MEDIUM — Tres superficies `current` todavía ordenan implementar el runner ya cerrado

**Evidencia.** Las páginas conservan `updated`, `verified_at` y `valid_at` en `2026-09-07`, pero su cuerpo y su `evidence_commit` todavía corresponden al freeze previo (`e76f7d9cdb49e262575ef234502de5e4e83ac61d`):

- `Documents/05_WIKI/roadmaps/current-portfolio.md:106` pide “Implementar y auditar por CPU el runner”; `:217` afirma que set-valued quedó “listo únicamente para implementar su runner”. SHA-256 del archivo: `b7781e024397b594240c577c41ce8a4159967f0563db9aa312670577da6bf5bd`.
- `Documents/05_WIKI/MAPA_VISUAL_DEL_PROGRAMA.md:83` dice “set-valued listo para runner”; `:107` conserva la implementación como trabajo vivo; `:149` rotula “SIGUIENTE: implementar runner CPU”. SHA-256: `5515d0f47cb2d7558a7dfb3e78ee05005312d50aa89c25f16101022aaf0664c1`.
- `Documents/05_WIKI/concepts/ground-truth-geometria-proporcional.md:831-835` termina el relato en R557 y declara al contrato listo únicamente para implementar el runner. SHA-256: `640a3fee6f3e38df1465d38022f6b23e89112d684e4c56fc278ffe1ba31b3e19`.

Esto contradice el estado correcto ya asentado en `Proyecto_Estado_Actual.md:13,24,220,282`, `LLM_CONTEXT.md:526-534,895` y `proportional-architecture-experiments.md:158-166,204-219,1313-1319`: el runner cerró primario/replay `14/14`, unitarios `9/9`, mutaciones `54/54` y R564 `PASS 0/0/0`; el siguiente paso es diseñar y auditar el paquete prospectivo físicamente separado, todavía sin crear el draw.

**Impacto.** Una persona o agente que entre por el mapa visual, el roadmap de cartera o la página conceptual recibe una orden operacional obsoleta y puede duplicar una implementación ya auditada. El defecto no altera los números del cierre ni promueve una arquitectura, pero rompe la consistencia del estado y del próximo responsable entre páginas que se declaran vigentes.

**Corrección requerida.** Actualizar juntas las tres páginas al relevo prospectivo, incorporar las fuentes de cierre/R564 y el commit de evidencia correspondiente, y regenerar `catalog.json`. Mantener explícitos histórico abierto, ausencia de draw fresco, no promoción y autoridad del usuario sobre `GO/NO-GO`.

### F-02 — LOW — La bitácora pública conserva paths efímeros concretos y una referencia a runtime interno

**Evidencia.** `Documents/00_TRONCAL/bitacora_desarrollo.md:4090` publica `/tmp/phideus-vibetensor-spike`; `:4641`, `/tmp/test_constellation.npz`; y `:4180`, el runtime `$CODEX_HOME/skills/phideus-doc-maintainer/`. SHA-256 del archivo: `15ba31ff63e338e5e37ee2826e9de5a1c9912d154adec85bf2fbac03d4147ba5`.

**Impacto.** Son entradas históricas anteriores al cierre y no exponen un home real, credencial o dato científico. Aun así, dos rutas dependen de una máquina concreta y la tercera describe una capa local de agente que la política pública pide mantener fuera de la navegación/documentación canónica. El riesgo es de higiene y portabilidad, no de validez metodológica.

**Corrección requerida.** En una pasada documental separada, sustituir sólo esos locadores por placeholders públicos o rutas relativas y reencuadrar la referencia al runtime interno sin reescribir el hecho histórico. No afecta el cierre set-valued.

## Trazabilidad numérica y metodológica

Las afirmaciones nuevas son trazables y consistentes:

- Inventario: `35` archivos, `17` NPZ, `17` JSON, `585` arrays y `35.701.634` valores por artefacto; `32` archivos comparables byte-exactos (`PROGRAM_CLOSURE...:29-30`; R564 `:26-29,64`).
- Validación: primario y replay `14/14`, unitarios `9/9`, mutaciones `54/54` (`PROGRAM_CLOSURE...:42-45`; R564 `:61-69,81-89`).
- Coste canónico: `28.27180427312851 s` y `986238976 B` para primario; `28.148873522877693 s` y `985055232 B` para replay; suma `56.4206777960062 s`. La bitácora redondea correctamente a `56,421 s` y menos de `1 GiB` por proceso (`bitacora_desarrollo.md:14-19`; R564 `:68-73`).
- Diagnóstico: joint favorece exact-set NLL pero no resuelve Brier; contextual favorece regret medio e incompatibilidad pero empeora worst regret; soporte común `67/215` marginal y `74/235` joint deja controles `NOT_EVALUABLE` (`PROGRAM_CLOSURE...:63-77`; R562 `:104`; `bitacora_desarrollo.md:21-26`).
- Régimen: todos los documentos nuevos distinguen `OPENED_DATA_IMPLEMENTATION_DIAGNOSTIC` de una realización prospectiva y mantienen la envolvente física como próximo trabajo (`Proyecto_Estado_Actual.md:24,220`; `CATALOGO_NARRATIVO...:63-79`; `INFORME_HISTORICO...:63-71`; `LLM_CONTEXT.md:526-534`; roadmap `:204-219`).
- Claims: no se encontró promoción de `JOINT`, `CONTEXTUAL`, A19 o PPU, autorización de draw, techo ni decisión `GO/NO-GO` en el material de cierre.

## Verificaciones mecánicas

- `venv/bin/python scripts/lint_phideus_wiki.py`: `PASS: 18 páginas, 57 fuentes, IDs y enlaces válidos`.
- Reconstrucción en memoria del catálogo mediante la misma lógica del linter: `catalog.json` es un match exacto, con `18` páginas.
- `git diff --check 22aff27^ 22aff27`: exit `0`.
- Los commits completos `b27c044bd490495a335b3b6c3469fc662d0f6abd` y `e76f7d9cdb49e262575ef234502de5e4e83ac61d` existen y son ancestros de `22aff27d6ab60f8ecec5a83cf8507e7967b10d76`.
- `source_paths`: `85/85` resolvieron en `LLM_CONTEXT.md`; `128/128` en `proportional-architecture-experiments.md`.
- En los seis documentos modificados se resolvieron `11/11` enlaces Markdown locales.
- Las adiciones del cierre no introducen `/mnt`, `/root`, `/home`, `file://`, `.codex`, `.claude` ni referencias a `PENDIENTES.md`. F-02 corresponde exclusivamente al cuerpo histórico preexistente de la bitácora.

## Conclusión

El contenido modificado por `22aff27` representa correctamente el cierre técnico y conserva sus límites científicos. El dictamen `REVISE` responde a una sincronización transversal incompleta, no a un defecto numérico o metodológico de R564. Resolver F-01 basta para alinear el próximo paso en las superficies `current`; F-02 puede tratarse como higiene pública independiente y no bloquea el relevo prospectivo.
