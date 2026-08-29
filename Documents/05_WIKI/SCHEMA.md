# Esquema de mantenimiento de la wiki

## Propósito

Este esquema convierte la wiki en una capa mantenible por agentes y legible por
personas. La fuente de verdad sigue estando en los documentos y artefactos
enlazados. La wiki orienta, integra y hace visibles las tensiones; no sustituye
la verificación de evidencia.

## Frontmatter mínimo

Toda página sustantiva usa:

```yaml
---
schema_version: 1
id: identificador-estable
kind: front | roadmap | concept | map | context
page_status: current | historical
front_status: focus_active | residual_active | decision_ready | reopenable | incubated | closed | superseded | paused | projection | transversal
architecture_status: baseline | candidate | incubated | historical | not_applicable | mixed
experiment_status: running | analysis_pending | phase_closed | not_started | not_applicable | mixed
evidence_status: <alcance explícito de la evidencia>
decision_status: decided | pending_user | pending_analysis | not_applicable | preserved
updated: YYYY-MM-DD
verified_at: YYYY-MM-DD
valid_at: YYYY-MM-DD
recorded_at: YYYY-MM-DD
evidence_commit: <git-sha>
source_paths:
  - ruta/relativa/al/repo.md
depends_on: []
tangents: []
---
```

`page_status` describe la vigencia de la página. `front_status` describe el
estatuto científico u operativo del objeto documentado. Separarlos evita que
una página vigente sobre un frente cerrado sea confundida con un experimento
activo.

`valid_at` identifica el corte temporal que la página describe y
`recorded_at` cuándo esa lectura fue incorporada a la wiki. La diferencia
permite preservar un estado histórico sin presentarlo como presente.

`evidence_commit` fija el commit base contra el que se auditaron las fuentes
de evidencia. No pretende identificar el worktree que contiene la propia wiki:
hasta su commit, esa capa puede estar en edición.

Las páginas `kind: front` deben declarar además los cuatro estados ortogonales.
Una arquitectura puede estar incubada con una fase experimental cerrada,
evidencia multi-seed y una decisión todavía pendiente: reducir todo eso a una
sola etiqueta destruye información.

## Estados de frente

| Estado | Significado |
|---|---|
| `focus_active` | foco explícito del programa; tiene una tarea inmediata declarada |
| `residual_active` | única pregunta abierta que sobrevive dentro de un frente mayormente cerrado |
| `decision_ready` | evidencia cerrada; falta una decisión estratégica del usuario |
| `reopenable` | fase actual cerrada; existe un punto de reentrada bien definido |
| `incubated` | línea arquitectónica válida pero todavía no canónica para todo el programa |
| `closed` | pregunta o fase cerrada dentro de su alcance |
| `superseded` | absorbida o reformulada por una línea posterior |
| `paused` | preservada sin ejecución inmediata |
| `projection` | horizonte sin protocolo experimental activo |
| `transversal` | concepto o infraestructura que cruza varios frentes |

Ninguno de estos estados equivale por sí solo a GO/NO-GO. Esa decisión pertenece
al usuario.

## Régimen de evidencia

Las páginas deben separar, cuando corresponda:

- **Observación:** dato producido por un artefacto o documento fuente.
- **Hipótesis:** explicación posible que todavía requiere discriminación.
- **Inferencia:** lectura sostenida por varias observaciones dentro de un alcance.
- **Decisión:** selección explícita realizada por el usuario.

Todo número o claim experimental debe enlazar una fuente. Cuando dos documentos
discrepan, la wiki registra la contradicción y la autoridad relativa; no corrige
silenciosamente el pasado.

El registro machine-readable [sources.yaml](sources.yaml) asigna IDs estables a
las fuentes canónicas. Los claims centrales de `LLM_CONTEXT.md` citan esos IDs;
las páginas de frente conservan además links humanos directos.

## Operaciones

### Ingesta

1. Leer la fuente canónica completa pertinente.
2. Identificar desde [index.md](index.md) las páginas afectadas.
3. Actualizar síntesis, relaciones, tensiones y preguntas abiertas.
4. Renovar `verified_at`, `evidence_commit` y `source_paths`.
5. Añadir una entrada a [log.md](log.md).
6. Ejecutar `venv/bin/python scripts/lint_phideus_wiki.py --write-catalog`.

### Consulta

1. Leer [index.md](index.md).
2. Para una vista global, leer [LLM_CONTEXT.md](LLM_CONTEXT.md).
3. Leer las páginas de frente o concepto relevantes.
4. Volver a las fuentes enlazadas para números, decisiones o precisión fina.

### Mantenimiento

- Actualizar la wiki cuando cambie el estado, roadmap, framing o arquitectura de
  un frente.
- Preferir actualizar una página existente antes que crear una duplicada.
- Mantener toda página sustantiva enlazada desde `index.md`.
- Conservar páginas superseded o históricas cuando expliquen genealogía real.
- No convertir `log.md` en bitácora experimental.
- El linter valida esquema, SHAs, IDs, relaciones, fuentes registradas y enlaces;
  una wiki que no pasa el lint no debe considerarse actualizada.

## Reglas de autoridad

- El usuario conserva GO/NO-GO, prioridades y promociones arquitectónicas.
- Los informes y artefactos conservan resultados; la wiki los sintetiza.
- Los roadmaps conservan secuencias locales; la wiki explicita dependencias
  transversales.
- La bitácora conserva cronología; el log registra mantenimiento de la wiki.
- Los documentos históricos se contextualizan, no se reescriben para que
  parezcan predicciones correctas del presente.

## Auditoría semántica

El lint estructural no puede determinar por sí solo si una síntesis expresa
correctamente la evidencia. Cada actualización de estado debe incluir una
lectura completa de las fuentes afectadas y un contraste cruzado de la página
de frente, el mapa humano, el contexto LLM y el roadmap de portafolio.

Como defensa mínima, el linter rechaza rangos compactos que abrevien desde
`P0` hasta `P6`: pueden incluir por accidente una fase intermedia no ejecutada. Las
unidades completadas deben enumerarse. Para cambios de estado sustantivos se
requiere además una auditoría independiente posterior a la edición.
