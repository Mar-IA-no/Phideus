# R567 — Reauditoría final de la documentación del runner set-valued

**Fecha:** 2026-09-07
**HEAD auditado:** `57da97eae1976b7b3392a86246af0e2c706932d8`
**Commit de evidencia del cierre:** `b27c044bd490495a335b3b6c3469fc662d0f6abd`
**Régimen:** CPU exclusivamente; no se consultó ni utilizó GPU/CUDA.
**Escrituras:** únicamente este informe; no se modificó documentación pública, registros, implementación ni artefactos experimentales.

## Dictamen

**REVISE — 0 HIGH, 1 MEDIUM, 0 LOW.**

Las correcciones explícitas de R565 F-01 y F-02 están resueltas. Las cinco
superficies señaladas por R566 F-01R también incorporaron correctamente el
cierre del runner: README, informe transversal, registro set-valued, registro
de fuentes y log coinciden ahora en histórico abierto, validación exacta, falta
de evidencia prospectiva y diseño del paquete físico como próximo paso. El
catálogo generado es exacto y hereda el commit de evidencia actualizado.

La lectura completa del registro arquitectónico encontró, sin embargo, un
campo operativo adicional no corregido: la rama relacional cuyo freeze fue
rechazado conserva `priority: immediate`. Ese valor contradice tanto su propio
`next_action` —preservar el freeze, no retunear y exigir un plan nuevo— como el
estado transversal que identifica al paquete prospectivo set-valued como único
relevo habilitado. Por ello R566 F-01R queda sustancialmente corregido, pero no
cerrado en toda la representación machine-readable.

No se detectaron errores nuevos en cifras, regímenes de evidencia, fuentes,
YAML, enlaces, promoción, techo o autoridad de GO/NO-GO.

## Superficies corregidas por R566

### README público

`README.md:551-567` conserva la secuencia histórica correcta: el dual native
freeze habilitó el runner; el runner ya cerró sobre histórico abierto con
primario/replay `14/14`, unitarios `9/9`, mutaciones `54/54` y R564
`PASS 0/0/0`; joint, contextual y controles matched retienen sus límites; y el
próximo paso diseña la envolvente física antes de crear un draw. No promueve
arquitectura ni declara GO/NO-GO.

### Informe transversal

`Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/GEOMETRIA_PROPORCIONAL_BASES_DE_VERDAD.md:679`
ya no termina en el permiso de implementar el runner. Registra que la mecánica,
el replay y la auditoría cerraron sobre datos históricos abiertos, separa esos
resultados de transporte prospectivo y ubica la continuación en la envolvente
física pre-draw.

### Registro set-valued

`Documents/05_WIKI/architecture-registry.yaml:381-430` actualiza la prioridad
set-valued a `prospective_physical_package_design_pending`, agrega
`runner_preflight` con validación `14/14`, `14/14`, `9/9`, `54/54`, replay
byte-exacto y R564 `PASS 0/0/0`, preserva el diagnóstico mixto y desplaza
`next_action` al paquete prospectivo físicamente separado. Sus diez rutas de
evidencia resuelven `10/10` e incluyen cierre, reporte local y R564.

Los campos `dual_native_freeze` que todavía hablan de permiso para implementar
son históricos y correctos: preceden inmediatamente al nuevo bloque
`runner_preflight`. No constituyen una instrucción vigente.

### Registro de fuentes y log

`Documents/05_WIKI/sources.yaml:1-84,585-592` fija
`evidence_commit: b27c044...`, integra el cierre R564 en el snapshot y registra
exactamente una vez `SRC-PROP-SET-VALUED-RUNNER`. El registry contiene `58`
fuentes, `58` IDs únicos y `58/58` paths existentes. El registro anterior
`SRC-PROP-DUAL-NATIVE-FREEZE` conserva correctamente el estado histórico de su
propia fuente.

`Documents/05_WIKI/log.md:3-18` añade una entrada posterior, sin reescribir la
historia: registra `14/14`, `9/9`, `54/54`, `32` archivos comparables
byte-exactos, R564, soporte `67/215` y `74/235`, y el relevo al paquete físico.
Declara expresamente que no hubo nuevo draw, monitor, lockbox, GPU, promoción
ni GO/NO-GO.

### Catálogo

`Documents/05_WIKI/catalog.json:1-5` ya hereda
`b27c044bd490495a335b3b6c3469fc662d0f6abd`. Una reconstrucción independiente
con la misma lógica de `lint_phideus_wiki.py` produjo igualdad exacta del objeto
completo: `18` páginas y cero diferencias.

## Correcciones anteriores

### R565 F-01 — resuelto

Las tres páginas originalmente obsoletas permanecen alineadas y sin cambios
desde R566:

- `roadmaps/current-portfolio.md:109,215-223`;
- `MAPA_VISUAL_DEL_PROGRAMA.md:86,110,150-153`;
- `concepts/ground-truth-geometria-proporcional.md:827-841`.

Todas usan `evidence_commit: b27c044...`. Sus `source_paths` resuelven
`63/63`, `43/43` y `31/31`; cada página contiene el trío cierre, reporte local
y R564. Las formulaciones que todavía mencionan implementar el runner son
secuencias históricas inmediatamente cerradas por el resultado R564 o pasos
enumerados como completados, no próximas acciones vigentes.

### R565 F-02 — resuelto

La bitácora conserva los reemplazos públicos y portables:

- `bitacora_desarrollo.md:4090`: `$TMPDIR/phideus-vibetensor-spike`;
- `bitacora_desarrollo.md:4180`: instalación privada fuera del repositorio;
- `bitacora_desarrollo.md:4641`: `$TMPDIR/test_constellation.npz`.

No reaparecieron los tres locators originales ni otros paths concretos privados
en la bitácora o en las tres páginas corregidas.

## Finding vigente

### R567-F01 — MEDIUM — La rama relacional rechazada conserva prioridad inmediata

**Evidencia.** `Documents/05_WIKI/architecture-registry.yaml:10-13` declara:

```yaml
id: proportional-coherence-graph-core
status: candidate
priority: immediate
```

En el mismo record, el resultado posterior establece
`relational_freeze_rejected_by_predeclared_K192_confirmation`
(`:292-297`) y su `next_action` ordena preservar el freeze rechazado, no
retunear K y exigir un plan nuevo antes de reformular el surrogate (`:298`). El
estado canónico transversal identifica en cambio al set-valued como la única
rama habilitada y al diseño del paquete prospectivo como próximo trabajo
(`Proyecto_Estado_Actual.md:220`; `current-portfolio.md:109`; registro
set-valued `:384,413-418`).

**Impacto.** Un consumidor humano que lea todo el record puede resolver la
contradicción por cronología. Un consumidor machine-readable que ordene por
`priority`, en cambio, puede seleccionar como inmediata una rama que requiere
un plan nuevo y desplazar el relevo vigente. Es la misma clase de riesgo
operativo que R566 detectó en el antiguo `next_action`, aunque queda limitada a
un único campo y no afecta la evidencia experimental.

**Corrección requerida.** Sustituir `priority: immediate` por un estado legible
coherente con freeze rechazado/preservado y reformulación bajo plan nuevo. No
cambiar `status: candidate`: R557 rechazó ese freeze concreto y no declaró un
techo para toda arquitectura relacional.

## Coherencia científica

Las superficies vigentes forman una misma lectura, salvo el campo de prioridad
anterior:

- el runner cerró únicamente sobre poblaciones históricas abiertas;
- primario y replay pasaron `14/14`, las pruebas `9/9`, las mutaciones `54/54`
  y los `32` archivos comparables fueron byte-exactos;
- joint favorece exact-set NLL pero no resuelve Brier;
- contextual favorece regret medio e incompatibilidad, pero empeora worst
  regret;
- los controles matched no son evaluables por soporte común `67/215` marginal
  y `74/235` joint;
- el siguiente trabajo es diseñar y auditar el paquete prospectivo físicamente
  separado antes de crear el draw;
- no hubo draw nuevo, uso de GPU, promoción arquitectónica ni decisión
  GO/NO-GO.

Las cifras y límites coinciden con
`PROGRAM_CLOSURE_SET_VALUED_NATIVE_RUNNER_PREFLIGHT.md`, el `REPORT.md` local y
R564. No se mezcló el preflight de histórico abierto con evidencia prospectiva.

## Verificaciones mecánicas

- YAML: `sources.yaml` y `architecture-registry.yaml` parsean como mappings.
- Wiki lint: `PASS: 18 páginas, 58 fuentes, IDs y enlaces válidos`.
- Source registry: `58` entradas, `58` IDs únicos, `58/58` paths existentes;
  `SRC-PROP-SET-VALUED-RUNNER` aparece una sola vez.
- Evidencia de páginas: `18/18` commits existen y `18/18` son ancestros de
  HEAD; hay cuatro commits únicos.
- `b27c044bd490495a335b3b6c3469fc662d0f6abd` existe y es ancestro de
  `57da97eae1976b7b3392a86246af0e2c706932d8`.
- Las tres páginas de R565 conservan `137/137` `source_paths` resolubles y el
  trío de cierre `3/3` en cada una.
- El record set-valued conserva `10/10` rutas de evidencia resolubles.
- Catálogo: JSON válido y match exacto con la reconstrucción del generador,
  `18` páginas y commit global `b27c044...`.
- `git diff --check HEAD^ HEAD`: exit `0`.
- `git diff --check 28a44ea..HEAD`: exit `0`, incluyendo R566 y su corrección.
- La búsqueda semántica global no encontró otra próxima acción que ordene
  implementar el runner; los hits restantes pertenecen a secuencias históricas
  explícitamente cerradas. El único residuo operativo es `priority: immediate`
  de R567-F01.

## SHA-256 de artefactos auditados

| Artefacto | SHA-256 |
|---|---|
| `README.md` | `eeaa74acf228dae49c243795a2d30edccbe4d86695455cf668c4f428e298f9aa` |
| `GEOMETRIA_PROPORCIONAL_BASES_DE_VERDAD.md` | `585daedf286ff74564ce70140e02444d3ba11a956619529e47e8c3529110ab3e` |
| `architecture-registry.yaml` | `244ddd73c32c4fa595b8588c1ec5aad46cdbc5b367e22e54d1d1c7ace5b946bb` |
| `sources.yaml` | `6f66e38c7735f11d37f41cd3b7ca3634f4ec616cb91f50d784f099ada265f377` |
| `log.md` | `166bc1f01e3c37e99315c0e60c8e38e50538f41b9b83e0b5d51c752fa071a147` |
| `catalog.json` | `28e64b951316dc006a2b2547784f556741bfa0bebf6975a38962e92df56dbef1` |
| `current-portfolio.md` | `02accf6792619ba385e09b5f4d97097e7340b4755990adf84ce416cba1d033a6` |
| `MAPA_VISUAL_DEL_PROGRAMA.md` | `a87e3124ba28713d08db514188c86e98d8d2fb8814a74810a37ecb773eefc976` |
| `ground-truth-geometria-proporcional.md` | `9ac9cbcd354dec2a53ba627e986b4feeb728fb4853807e985c0eb29c349e05f9` |
| `bitacora_desarrollo.md` | `8928013823c85e22af893b57c7149968ce5244b479965adae985be43cf512e9a` |

## Conclusión

R565 F-01 y F-02 están resueltos. R566 F-01R corrigió todas las superficies que
enumeró y dejó sincronizados contenido, fuentes y catálogo, pero la lectura
integral del registro reveló una prioridad relacional anterior incompatible
con el relevo vigente. Corregir ese único campo permitiría cerrar la
sincronización documental sin tocar historia, implementación, artefactos ni
decisiones científicas.
