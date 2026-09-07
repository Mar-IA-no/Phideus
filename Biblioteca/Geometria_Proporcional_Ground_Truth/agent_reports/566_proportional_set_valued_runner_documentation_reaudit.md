# R566 — Reauditoría documental del cierre del runner set-valued

**Fecha:** 2026-09-07
**HEAD auditado:** `28a44eaa4c9cc4f114246b86af2c77f823c3f313`
**Commit de evidencia del cierre:** `b27c044bd490495a335b3b6c3469fc662d0f6abd`
**Régimen:** CPU exclusivamente; no se consultó ni utilizó GPU/CUDA.
**Escrituras:** únicamente este informe; no se modificó documentación pública, implementación ni artefactos experimentales.

## Dictamen

**REVISE — 0 HIGH, 1 MEDIUM, 0 LOW.**

F-02 de R565 está completamente resuelto. Las tres páginas concretas de F-01
también fueron corregidas correctamente: su estado, próximo paso, commit de
evidencia y fuentes de cierre coinciden con R564. Sin embargo, F-01 no está
resuelto en toda la superficie canónica exigida. El README público, el informe
transversal principal y tres registros operativos de la wiki todavía terminan
en el permiso de implementar el runner ya cerrado. El catálogo es una
reconstrucción byte-exacta de su generador, pero hereda de `sources.yaml` un
commit global anterior al cierre.

No apareció ningún defecto nuevo en números, régimen de evidencia, límites de
soporte, separación histórico/prospectivo, promoción arquitectónica o
GO/NO-GO. El finding es exclusivamente de sincronización documental y
trazabilidad del próximo paso.

## Verificación de las correcciones de R565

### F-01 — parcialmente resuelto

Las tres páginas originalmente citadas quedaron corregidas de forma completa:

- `Documents/05_WIKI/roadmaps/current-portfolio.md:109,215-223` registra el
  runner cerrado sobre histórico abierto —primario/replay `14/14`, unitarios
  `9/9`, mutaciones `54/54`, replay exacto y R564 `PASS 0/0/0`— y desplaza el
  próximo paso a diseñar la envolvente prospectiva físicamente separada, sin
  crear el draw.
- `Documents/05_WIKI/MAPA_VISUAL_DEL_PROGRAMA.md:86,110,150-153` representa el
  runner como ejecutado y conecta el paso siguiente con el paquete prospectivo
  físico.
- `Documents/05_WIKI/concepts/ground-truth-geometria-proporcional.md:827-841`
  conserva la secuencia freeze → runner cerrado → envolvente física y declara
  explícitamente que no hay evidencia prospectiva.

En las tres páginas, `evidence_commit` es
`b27c044bd490495a335b3b6c3469fc662d0f6abd`. Ese commit existe y es ancestro de
HEAD. Sus `source_paths` resuelven `63/63`, `43/43` y `31/31`, respectivamente;
cada página incorpora los tres anclajes requeridos:

1. `PROGRAM_CLOSURE_SET_VALUED_NATIVE_RUNNER_PREFLIGHT.md`;
2. `data/geometria_proporcional/proportional_set_valued_native_preflight_v1/REPORT.md`;
3. R564.

El cierre programático y R564 ya existen en el árbol de `b27c044` y permanecen
sin cambios en HEAD. El `REPORT.md` resuelve en el worktree y pertenece al árbol
local de datos ignorado por la política `/data/`; los dos artefactos versionados
adjudican su contenido y sostienen las cifras citadas.

### F-02 — resuelto

Los tres locators históricos de la bitácora fueron sustituidos sin alterar el
hecho narrado:

- `bitacora_desarrollo.md:4090`: `$TMPDIR/phideus-vibetensor-spike`;
- `bitacora_desarrollo.md:4180`: instalación privada fuera del repositorio;
- `bitacora_desarrollo.md:4641`: `$TMPDIR/test_constellation.npz`.

El barrido focalizado de la bitácora y de las tres páginas corregidas no
encontró paths concretos bajo `/tmp`, `/mnt`, `/root` o `/home`, URLs `file://`
ni referencias a runtimes privados de Codex o Claude. No queda finding vigente
de F-02.

## Finding vigente

### F-01R — MEDIUM — El relevo del runner sigue obsoleto fuera de las tres páginas corregidas

**Evidencia.** Cinco superficies con función vigente no incorporaron el cierre
R564:

- `README.md:551-563`, ancla pública canónica, dice que el contrato set-valued
  está listo únicamente para implementar y auditar el runner y termina en
  R557.
- `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/GEOMETRIA_PROPORCIONAL_BASES_DE_VERDAD.md:679`,
  informe transversal canónico del frente, afirma que el contrato quedó listo
  sólo para implementar el runner CPU.
- `Documents/05_WIKI/architecture-registry.yaml:408-413` conserva el estado
  `set_valued_freeze_valid_for_runner_implementation_only` y ordena implementar
  y auditar el runner como `next_action`; su evidencia termina en R557
  (`:415-422`).
- `Documents/05_WIKI/sources.yaml:3,72-80` mantiene como snapshot global el
  freeze anterior, con `evidence_commit` `e76f7d9...`, y no incorpora el runner,
  R564 ni el próximo paquete prospectivo.
- `Documents/05_WIKI/log.md:3-19` tiene como entrada más reciente el cierre del
  dual native freeze. Esa entrada es históricamente correcta y no debe
  reescribirse, pero falta una entrada posterior para el cierre del runner, pese
  a que `SCHEMA.md:94-99` exige añadirla durante la ingesta.

El registro puntual `SRC-PROP-DUAL-NATIVE-FREEZE` de
`sources.yaml:572-579` y los pasos históricos ya marcados como completados en
el roadmap no son stale: describen correctamente el estado que tuvo esa fuente
o la secuencia de ejecución. Del mismo modo, la bitácora conserva el antiguo
permiso dentro de su secuencia histórica, precedido por una entrada nueva que
expone el cierre actual. El problema está en las superficies que todavía actúan
como estado o siguiente acción vigente.

**Efecto en el catálogo.** La reconstrucción independiente produjo un match
exacto de `catalog.json` con la lógica del linter: `18` páginas y cero
diferencias. No obstante, `catalog.json:4` hereda de `sources.yaml` el commit
global `e76f7d9...`. Por tanto, el archivo generado es exacto respecto de una
fuente registral semánticamente desactualizada; regenerarlo sin corregir antes
`sources.yaml` reproduce la inconsistencia.

**Impacto.** El README y el informe transversal pueden dirigir a una persona
hacia trabajo duplicado; el registro arquitectónico puede hacerlo de forma
machine-readable; y la procedencia global del catálogo no representa el cierre
que tres de sus páginas ya declaran. La inconsistencia no invalida R564 ni
autoriza un draw, pero rompe la regla de estado canónico único y el relevo
operativo recuperable.

**Corrección requerida.** Sin reescribir historia, propagar el cierre del
runner y el próximo paquete prospectivo al README, al informe transversal, al
registro arquitectónico y al snapshot/commit de `sources.yaml`; añadir una
entrada nueva al log; incorporar cierre/R564 a los registros de evidencia
vigentes; y después regenerar `catalog.json`.

## Coherencia científica y trazabilidad

Las correcciones de las tres páginas reproducen las fuentes sin extrapolar:

- primario y replay `14/14`, unitarios `9/9` y mutaciones `54/54`;
- joint mejora exact-set NLL pero no resuelve Brier;
- contextual mejora regret medio e incompatibilidad, pero empeora worst regret;
- el soporte común `67/215` marginal y `74/235` joint impide evaluar los
  controles matched;
- el runner es un diagnóstico sobre datos históricos abiertos, no una
  realización prospectiva;
- no hay arquitectura promovida, nuevo draw, techo ni decisión GO/NO-GO.

Las formulaciones históricas de `LLM_CONTEXT.md:516-534` y
`roadmaps/proportional-architecture-experiments.md:158-166,210-219,1310-1319`
son correctas porque enlazan explícitamente el permiso anterior con la
ejecución ya cerrada y marcan el paso 32 como completado.

## Verificaciones mecánicas

- `venv/bin/python scripts/lint_phideus_wiki.py`: `PASS: 18 páginas, 57 fuentes,
  IDs y enlaces válidos`.
- Reconstrucción independiente del catálogo con la misma lógica del linter:
  match exacto, `18` páginas. La salvedad semántica de su commit global queda
  documentada en F-01R.
- Los `137/137` registros `source_paths` de las tres páginas corregidas
  resuelven localmente; el trío de cierre está presente `3/3` en cada página.
- `b27c044bd490495a335b3b6c3469fc662d0f6abd` existe y es ancestro de HEAD.
- `git diff --check HEAD^ HEAD`: exit `0` para el commit correctivo `28a44ea`.
- El rango acumulado `22aff27..HEAD` devuelve exit `2` únicamente por cuatro
  hard line breaks Markdown —dos espacios finales— en el informe R565 ya
  integrado por `b8b1780`; no fueron introducidos por `28a44ea`, no afectan una
  superficie pública y no constituyen finding sustantivo de esta reauditoría.

## SHA-256 de artefactos auditados

| Artefacto | SHA-256 |
|---|---|
| `current-portfolio.md` | `02accf6792619ba385e09b5f4d97097e7340b4755990adf84ce416cba1d033a6` |
| `MAPA_VISUAL_DEL_PROGRAMA.md` | `a87e3124ba28713d08db514188c86e98d8d2fb8814a74810a37ecb773eefc976` |
| `ground-truth-geometria-proporcional.md` | `9ac9cbcd354dec2a53ba627e986b4feeb728fb4853807e985c0eb29c349e05f9` |
| `bitacora_desarrollo.md` | `8928013823c85e22af893b57c7149968ce5244b479965adae985be43cf512e9a` |
| `catalog.json` | `587480033d548ba0bf3467abdd197ca11f960db7f5aff03cda44f7082e1dde48` |
| `sources.yaml` | `f67619c40e63915f80b9b2ad0759f8a4f96b585187542307b9f96c7a977c5871` |
| `architecture-registry.yaml` | `3c081d4f7608abd7db0f1915fdaf4302f18d92e5bd14e38b84a16a703fc3c96e` |
| `log.md` | `0fb1968cd7bda1207811aa6445d703f4a4ec28bef576b3fd6fef341f94372b6c` |
| `README.md` | `308cb4a1281424fe7c3905275a5e63c9694a8ea6b4e6569d228e997499e30364` |
| `GEOMETRIA_PROPORCIONAL_BASES_DE_VERDAD.md` | `24edc3f89e8cd44af8f0c60aabd4393fbac6f3b502b873c66be4ea24561bae3e` |

## Conclusión

El parche `28a44ea` resuelve fielmente los dos defectos locales señalados en
R565: las tres páginas concretas ya no duplican el runner y la bitácora ya no
publica los tres locators problemáticos. El dictamen global permanece `REVISE`
porque la propagación se detuvo antes de las anclas canónicas y de los registros
machine-readable de la wiki. Una segunda corrección focalizada puede cerrar
F-01R sin tocar implementación, evidencia experimental ni decisiones
científicas.
