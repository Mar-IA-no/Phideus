# Índice de la wiki de Phideus

> Actualizado: 2026-09-03
> Corte base de evidencia: `78e9377693bbe8b105ea5b356aac431fb4cc38a4`

## Entradas

- [Mapa visual del programa](MAPA_VISUAL_DEL_PROGRAMA.md): superficie humana,
  esquemática y comparativa.
- [Contexto integral para LLMs](LLM_CONTEXT.md): estado denso, autoridades,
  dependencias, tensiones y preguntas abiertas.
- [Mapa de portafolio y roadmaps](roadmaps/current-portfolio.md): relación entre
  foco, ramas residuales, decisiones, reaperturas y proyecciones.
- [Programa de arquitecturas y experimentos proporcionales](roadmaps/proportional-architecture-experiments.md):
  cartera concreta, primitive inmediata y escalera CPU-first.
- [Registro machine-readable de arquitecturas](architecture-registry.yaml): estados,
  primitives, controles y próximo discriminante sin borrar alternativas.

## Frentes

- [Escalón 1 y BIAS_CONTROL](fronts/escalon-1-bias-control.md)
- [Gate 6 AMT](fronts/gate-6-amt.md)
- [Escalón 2: Speech ↔ EGG](fronts/escalon-2.md)
- [Voz Expresiva Phideus](fronts/voz-expresiva.md)
- [Escalón 3: Audio XY ↔ Lissajous](fronts/escalon-3.md)
- [Atención Armónica](fronts/atencion-armonica.md)
- [Escalón 4: ECG ↔ PPG](fronts/escalon-4.md)
- [Líneas superseded, pausadas y cerradas](fronts/lineas-preservadas.md)

## Conceptos transversales

- [Tres vías de investigación](concepts/tres-vias-de-investigacion.md)
- [PPU y Natural Harmonic Geometry](concepts/ppu-geometria-armonica-natural.md)
- [Ground truth para geometría proporcional](concepts/ground-truth-geometria-proporcional.md)
- [Álgebra geométrica y ejecutores Clifford](concepts/algebra-geometrica-clifford.md)
- [Régimen de evidencia](concepts/regimen-de-evidencia.md)
- [Registro de tensiones documentales](concepts/contradicciones-documentales.md)

## Fuentes canónicas principales

- [README público](../../README.md)
- [Proyecto Estado Actual](../00_TRONCAL/Proyecto_Estado_Actual.md)
- [Índice general de documentación](../00_TRONCAL/INDICE_DOCUMENTACION.md)
- [Marco epistemológico](../../MARCO_EPISTEMOLOGICO_PHIDEUS.md)
- [Filosofía y proyección de Phideus](../04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/INFORME_FILOSOFIA_PROYECCION_PHIDEUS.md)
- [Historia de las representaciones de ratios](../04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md)
- [Geometría proporcional y bases de verdad](../04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/GEOMETRIA_PROPORCIONAL_BASES_DE_VERDAD.md)

## Mantenimiento

- [Esquema](SCHEMA.md)
- [Log](log.md)
- [Registro machine-readable de fuentes](sources.yaml)
- [Catálogo machine-readable de páginas](catalog.json)
- Validación: `python scripts/lint_phideus_wiki.py`

La recuperación canónica usa enlaces, front matter, `catalog.json` y el
registro arquitectónico. Con `18` páginas, una capa de embeddings duplicaría
estado sin resolver una falla observada de retrieval; se difiere hasta que una
evaluación de recuperación demuestre que el índice estructurado no alcanza.
