# Rosetta1 v1.0 - Archivado

**Estado**: Archivado (Enero 2026)
**Reemplazado por**: Rosetta1 2.0

---

## Contenido

Documentación del experimento Rosetta1 original. Estos documentos representan
el trabajo inicial que demostró cross-modal alignment (cos_sim = 0.766) pero
que fue identificado con debilidades metodológicas en el diagnóstico de
consistencia.

| Archivo | Descripción |
|---------|-------------|
| `INFORME_ROSETA_1_PARA_PUBLICACION.md` | Informe principal para publicación |
| `INFORME_ROSETA_1_HARMONIC_INFORMATION_THEORY.md` | Teoría de información armónica |
| `ANALISIS_EXPERIMENTO_ROSETA.md` | Análisis detallado del experimento |
| `PROPUESTA_ROSETA_2_AUDIO_CINEMATICA.md` | Propuesta para Roseta 2 (visual) |

## Por qué fue archivado

El diagnóstico GPT5.2Pro (Enero 2026) identificó:
1. Posible leakage en split (por frame vs por archivo)
2. z_private colapsado (varianza ~0)
3. Falta de controles negativos (shuffled, random)
4. Claims ambiguos sobre separación de regímenes

## Versión actual

Ver `Documents/Roseta/ROSETTA1_2.0_IMPLEMENTATION_PLAN.md` para el plan
de implementación actualizado que corrige estas debilidades.
