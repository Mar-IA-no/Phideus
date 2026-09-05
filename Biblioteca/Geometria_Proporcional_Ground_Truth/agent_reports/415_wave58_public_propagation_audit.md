# R415 — Auditoría independiente de propagación pública de la Ola 58

## Veredicto

**REVISE.** La lectura científica y los números propagados son fieles a R414 y
a los artefactos primario/replay. No encontré promoción arquitectónica,
`GO/NO-GO` ni una generalización indebida de la Ola 57. Persisten dos defectos
de actualización en superficies canónicas: el registro machine-readable quedó
anclado a un corte anterior a la Ola 58 y la cabecera ejecutiva todavía declara
Olas 26–57.

## Findings priorizados

### Media 1 — El registro de fuentes declara un snapshot y un commit anteriores a la Ola 58

`Documents/05_WIKI/sources.yaml:3-5` conserva
`evidence_commit: 37b1d081d7f5a6ff06a76bbda182457b66067517` y abre su
`snapshot_note` con «Olas 26-49 integradas». El texto del snapshot llega sólo
hasta la Ola 50 (`:53-59`), aunque el mismo archivo ya registra `SRC-PROP-W58`
en `:506-513`. El catálogo generado hereda el commit antiguo en
`Documents/05_WIKI/catalog.json:4`.

El hash `37b1d081...` existe, pero es el commit de propagación de la Ola 57 y no
contiene
`Biblioteca/Geometria_Proporcional_Ground_Truth/waves/WAVE_58_OPEN_MODEL_CLASS_DIAGNOSTIC_CLOSED.md`.
En cambio, `e1202c9c3f45e3f12cda38ea052411b44789d332` sí contiene el cierre y
es el corte que declaran `Documents/05_WIKI/index.md:4` y las páginas
actualizadas por `cccee80`. Por eso el linter puede aceptar las 52 fuentes y,
sin embargo, la procedencia global del registro sigue desfasada. Corregir el
snapshot y regenerar `catalog.json` desde el corte Wave 58.

### Media 2 — La cabecera del estado ejecutivo contradice su propio cuerpo

`Documents/00_TRONCAL/Proyecto_Estado_Actual.md:13` todavía anuncia «Olas
26–57 integradas» y resume el cierre de la Ola 57. El mismo documento describe
correctamente una campaña de 58 olas en `:24` y vuelve a fijar ese total en
`:281`. Al ser la primera línea temporal de la superficie ejecutiva, puede
inducir recuperación o lectura humana del estado anterior aunque el desarrollo
posterior sea correcto. Debe actualizarse a la Ola 58 sin borrar el matiz de la
Ola 57.

## Contraste independiente confirmado

- R414 y el cierre Wave 58 clasifican el resultado como
  `COMPLETE / OPEN-DATA / ADAPTIVE / SELECTED-AFTER-MONITOR-INSPECTION`, con
  `scientific_decision:null` y `architecture_promoted:false`.
- `analysis.json` contiene 36 candidatos canónicos y 24 probes históricos;
  todos los 60 tienen estado `PASS`. Hay 21 elegibles y esos mismos 21 forman
  el frente Pareto.
- La recomputación directa de `scores_and_masks.npz` produce 19 firmas
  conductuales para los 36 IDs. Nueve candidatos terminan en `HARD_ONLY`; las
  otras clases redundantes tienen tamaños `3, 3, 2, 2, 2, 2, 2`.
- El candidato nominado es
  `C-HGB-HGB-INCOMPATIBILITY-JOINT`. Su variante `JOINT` es idéntica a
  `SEQUENTIAL` en propuestas, autorizaciones y acciones de validation y
  monitor. La igualdad está correctamente acotada a la política nominada en
  las superficies densas; no se interpretó como igualdad de todo el retículo.
- Contra hard, el nominado conserva en validation: accuracy `-0.003587`, IC95
  `[-0.011727,+0.004001]`; compatibilidad `+0.004967`,
  `[+0.000138,+0.010072]`; regret `-0.005427`,
  `[-0.012314,+0.001288]`; worst regret `+0.008002`,
  `[-0.007450,+0.023455]`. En monitor: accuracy `+0.006400`,
  `[-0.002315,+0.015251]`; compatibilidad `+0.009123`,
  `[+0.000950,+0.016612]`; regret `-0.014490`,
  `[-0.023818,-0.005106]`; worst regret `-0.002451`,
  `[-0.019063,+0.013617]`.
- La alternativa HGB/HGB-harm conserva el contraste de cola: delta de worst
  regret `-0.025327`, IC95 `[-0.049292,-0.002996]`, frente al nominado. La
  documentación lo presenta como tradeoff y no como ganador global.
- Primario y replay coinciden en los diez artefactos científicos. El chequeo
  `LEGACY-W57` reproduce `34/34`. `analysis.json` y `REPORT.md` tienen hashes
  idénticos entre ambas corridas (`266049f5...` y `6e8d829c...`).
- `runtime.json` y los manifests registran ejecución CPU, dispositivo CUDA
  invisibilizado, UID/GID `65534` y cuatro hilos. No hubo uso de GPU ni de
  Mendieta en esta auditoría.
- La Ola 57 conserva su lectura: mejora incremental frente al proposer, no
  superioridad contra hard; `2/5` shams, patrón terminal `null`, replay
  `23/23 + 13/13`. Wave 58 se añade como diagnóstico abierto y no la
  reinterpreta.
- Los commits `e1202c9c3f45e3f12cda38ea052411b44789d332` y
  `cccee80358fe68cfb5984ca85b2229b0d661b971` existen. Los paths de evidencia
  incorporados por esos commits existen y el linter resuelve sus enlaces.
- El cuerpo narrativo mantiene el estilo explicativo: observación, limitación
  post-selección y próximo discriminante aparecen articulados sin convertir la
  lista de métricas en decisión científica.

## Validaciones

- `CUDA_VISIBLE_DEVICES=''` y máximo cuatro hilos:
  `venv/bin/pytest -q tests/test_wave58_open_diagnostic.py` → `22 passed`.
- `venv/bin/python scripts/lint_phideus_wiki.py` →
  `PASS: 18 páginas, 52 fuentes, IDs y enlaces válidos`.
- `git diff --check` → limpio antes de archivar este informe.

## Riesgo residual

La evidencia de Ola 58 sigue siendo adaptativa y post-selección sobre un único
draw ya abierto. El replay acredita determinismo, no transporte entre draws; la
degeneración de IDs tampoco identifica un selector causal. Esas limitaciones
están correctamente preservadas en la propagación pública y no cambian por los
dos defectos documentales señalados.
