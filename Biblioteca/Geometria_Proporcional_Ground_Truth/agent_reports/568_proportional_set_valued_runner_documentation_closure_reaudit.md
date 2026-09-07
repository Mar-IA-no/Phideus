# R568 — Reauditoría de cierre documental del runner set-valued

**Fecha:** 2026-09-07
**HEAD auditado:** `95d487f0d80e74c88443ca920b87b8523906814a`
**Commit de evidencia del cierre:** `b27c044bd490495a335b3b6c3469fc662d0f6abd`
**Régimen:** CPU exclusivamente; no se consultó ni utilizó GPU/CUDA.
**Escrituras:** únicamente este informe; no se modificó documentación pública, registros, implementación ni artefactos experimentales.

## Dictamen

**PASS — 0 HIGH, 0 MEDIUM, 0 LOW.**

R567-F01 quedó corregido. La prioridad de la rama relacional expresa ahora que
el freeze rechazado se preserva y que toda reformulación requiere un plan
nuevo. La revisión completa de las cuatro parejas `priority`/`next_action` del
registro no encontró otra contradicción machine-readable. Los hallazgos R565
F-01/F-02, R566 F-01R y R567-F01 están todos resueltos.

El estado canónico permanece acotado: el runner set-valued cerró mecánica y
replay sobre datos históricos abiertos; el próximo trabajo es diseñar y auditar
el paquete prospectivo físicamente separado antes de crear un draw; la GPU
permanece suspendida; no hubo promoción arquitectónica ni decisión GO/NO-GO.

## Corrección R567-F01

`Documents/05_WIKI/architecture-registry.yaml:10-13` declara ahora:

```yaml
id: proportional-coherence-graph-core
status: candidate
priority: preserved_requires_new_plan_after_rejected_freeze
```

El valor coincide con el resultado posterior del mismo record:

- `:292-297`: el freeze relacional fue rechazado por la confirmación K192
  predeclarada, sin convertir ese rechazo en techo de la arquitectura;
- `:298`: debe preservarse el freeze, no retunear K sobre la confirmación
  abierta y exigir un plan nuevo antes de reformular el surrogate.

Mantener `status: candidate` es correcto. R557 rechazó esta materialización
fixed-K y no descartó toda arquitectura relacional. El cambio afecta sólo la
prioridad operativa que antes decía `immediate`.

## Coherencia de prioridades y próximas acciones

La lectura completa de `architecture-registry.yaml` dejó cuatro records
coherentes:

| Arquitectura | Estado | Prioridad | Próxima acción | Lectura |
|---|---|---|---|---|
| `proportional-coherence-graph-core` | `candidate` | `preserved_requires_new_plan_after_rejected_freeze` | preservar, no retunear, exigir plan nuevo | coherente con rechazo específico, sin declarar techo |
| `joint-set-posterior-policy` | `experimental` | `prospective_physical_package_design_pending` | diseñar y auditar paquete físico antes del draw | único relevo experimental habilitado |
| `relative-spectrum-reader` | `candidate` | `blocked_by_query_authority` | definir query externa no agotada por executor | coherente con bloqueo de autoridad |
| `typed-router-executor-system` | `incubated` | `later_integration` | esperar una primitive que sobreviva controles | coherente con router diferido |

El bloque histórico `dual_native_freeze` del record set-valued conserva
correctamente el permiso anterior de implementar el runner. Está seguido por
`runner_preflight`, que registra su cierre, y por un `next_action` prospectivo;
no funciona como prioridad o instrucción vigente.

## Cierre de findings anteriores

### R565 F-01

Resuelto. `current-portfolio.md`, `MAPA_VISUAL_DEL_PROGRAMA.md` y
`ground-truth-geometria-proporcional.md` registran runner cerrado, validación
`14/14`, `9/9`, `54/54`, replay exacto, R564 `PASS 0/0/0`, histórico abierto y
paquete prospectivo físico como próximo paso. Sus `137/137` `source_paths`
resuelven y cada página conserva el trío cierre, reporte local y R564.

### R565 F-02

Resuelto. La bitácora conserva `$TMPDIR/phideus-vibetensor-spike`, la
descripción no locativa de la instalación privada y
`$TMPDIR/test_constellation.npz`. No reaparecieron los tres locators originales
ni otros paths privados concretos en el alcance revalidado.

### R566 F-01R

Resuelto. README, informe transversal, registro set-valued, `sources.yaml`, log
y catálogo coinciden en:

- runner cerrado sobre poblaciones históricas abiertas;
- primario/replay `14/14`, unitarios `9/9`, mutaciones `54/54` y R564
  `PASS 0/0/0`;
- joint favorable en exact-set NLL sin resolver Brier;
- contextual favorable en regret medio e incompatibilidad, adverso en worst
  regret;
- controles matched no evaluables por soporte común `67/215` marginal y
  `74/235` joint;
- diseño del paquete prospectivo físico antes de crear un draw;
- sin GPU, promoción ni GO/NO-GO.

El registro de fuentes tiene `58` entradas, `58` IDs únicos y `58/58` paths
existentes. `SRC-PROP-SET-VALUED-RUNNER` aparece exactamente una vez y el commit
global es `b27c044...`. El log añadió el cierre sin reescribir su entrada
histórica anterior.

### R567-F01

Resuelto por `priority: preserved_requires_new_plan_after_rejected_freeze`.
No queda prioridad inmediata para la rama relacional.

## Lenguaje vigente

La búsqueda global en README, estado troncal, teoría transversal y toda la wiki
no encontró una próxima acción vigente que ordene implementar el runner. La
única coincidencia textual con “lista sólo para implementar runner” pertenece
al registro inmutable de la fuente dual-native-freeze y describe correctamente
el estado histórico de ese cierre. Las demás menciones históricas enlazan de
inmediato con el runner ya ejecutado o marcan el paso como completado.

No se encontró lenguaje que presente el diagnóstico abierto como prospectivo,
autorice un draw o GPU, promueva `JOINT`/`CONTEXTUAL`, declare un techo o tome
la decisión GO/NO-GO.

## Verificaciones mecánicas

- YAML: `architecture-registry.yaml` y `sources.yaml` parsean como mappings.
- Wiki lint: `PASS: 18 páginas, 58 fuentes, IDs y enlaces válidos`.
- Registro de fuentes: `58` entradas, `58` IDs únicos, `58/58` paths
  existentes y un solo `SRC-PROP-SET-VALUED-RUNNER`.
- Catálogo: JSON válido y match exacto con la reconstrucción del generador;
  `18` páginas y `evidence_commit: b27c044...`.
- Evidencia de páginas: `18/18` commits existen y `18/18` son ancestros de
  HEAD.
- `b27c044bd490495a335b3b6c3469fc662d0f6abd` existe y es ancestro de
  `95d487f0d80e74c88443ca920b87b8523906814a`.
- Páginas corregidas por R565: `63/63`, `43/43` y `31/31` source paths
  resolubles; trío de cierre `3/3` en cada una.
- Record set-valued: `10/10` rutas de evidencia resolubles.
- `git diff --check HEAD^ HEAD`: exit `0`.
- `git diff --check 57da97e..HEAD`: exit `0`, incluyendo R567 y su corrección.
- Worktree limpio antes de crear este informe.

## SHA-256 de artefactos auditados

| Artefacto | SHA-256 |
|---|---|
| `architecture-registry.yaml` | `64b37aff2d8296a402f80cc9a9487da656ea6e20f58048d5cb760061e45ed46e` |
| `sources.yaml` | `6f66e38c7735f11d37f41cd3b7ca3634f4ec616cb91f50d784f099ada265f377` |
| `log.md` | `166bc1f01e3c37e99315c0e60c8e38e50538f41b9b83e0b5d51c752fa071a147` |
| `catalog.json` | `28e64b951316dc006a2b2547784f556741bfa0bebf6975a38962e92df56dbef1` |
| `README.md` | `eeaa74acf228dae49c243795a2d30edccbe4d86695455cf668c4f428e298f9aa` |
| `GEOMETRIA_PROPORCIONAL_BASES_DE_VERDAD.md` | `585daedf286ff74564ce70140e02444d3ba11a956619529e47e8c3529110ab3e` |
| `current-portfolio.md` | `02accf6792619ba385e09b5f4d97097e7340b4755990adf84ce416cba1d033a6` |
| `MAPA_VISUAL_DEL_PROGRAMA.md` | `a87e3124ba28713d08db514188c86e98d8d2fb8814a74810a37ecb773eefc976` |
| `ground-truth-geometria-proporcional.md` | `9ac9cbcd354dec2a53ba627e986b4feeb728fb4853807e985c0eb29c349e05f9` |
| `bitacora_desarrollo.md` | `8928013823c85e22af893b57c7149968ce5244b479965adae985be43cf512e9a` |

## Conclusión

El cierre documental del runner set-valued queda consistente en superficies
humanas y machine-readable. La rama relacional se preserva sin prioridad
inmediata; la set-valued conserva el único relevo activo hacia diseño
prospectivo pre-draw; las fuentes, catálogo y estados operativos coinciden. No
quedan findings abiertos en el alcance auditado.
