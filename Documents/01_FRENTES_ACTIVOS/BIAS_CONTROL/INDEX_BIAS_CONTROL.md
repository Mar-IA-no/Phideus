<div align="center">

# INDEX BIAS_CONTROL
### Navegación por fases del roadmap

![Estado](https://img.shields.io/badge/Estado-Reordenado-0A7E3B?style=for-the-badge)
![Foco](https://img.shields.io/badge/Foco-Escalon_1--C-F59E0B?style=for-the-badge)

</div>

> [!NOTE]
> **Corte operativo (2026-02-12 15:42 UTC):** Bloque A v1.1 sigue activo (`S0`, `Run A`, `Run B`, `Run C` y `Run D` completados, `Run D-02` en curso con best parcial ep18 `S=59.6%`, `hard_neg=91.0%`). Gate 4.2 mantiene implementacion paralela habilitada y screening posterior al foundation lock definitivo (`C5 vs D5 vs D-02(best)`).
>  
> Para comparar arquitectura/config de runs (cuadros preflight por `Run A/B/C/D` y recipe de `Run D-02`), ver:
> - `ROADMAP_BIAS_CONTROL.md` (seccion 7.1.b)
> - `05_PLAN_POST_DIAGNOSTICO_BLOQUE_A/PLAN_EJECUCION_POST_DEC005_v1.1.md` (seccion 2.1.b)
> - `06_GATE_4_2_RATIO_CENTRICO/PLANES/plan_gate_4.2.md` (Fase 0)
>
> Visualizaciones 3D publicadas: `https://altermundi.github.io/Phideus/` (adaptadas sobre `https://github.com/bbycroft/llm-viz`).

## Orden canónico

- `ROADMAP_BIAS_CONTROL.md`  
  Documento troncal de estado, métricas y decisiones.

- `01_GATES_0_2_5/`  
  Fase base del programa (fundación cross-modal).
  - `GATE_2_FOUNDATION/INFORME_GATE2_COMPLETO.md`
  - `GATE_2_FOUNDATION/EVIDENCIAS/`

- `02_GATE_3_DANN/`  
  Ciclo adversarial y cierre de la línea DANN.
  - `INFORME_GATE3_COMPLETO.md`
  - `COMPARISON_GATE3.md`
  - `EVIDENCIAS/`

- `03_GATE_4_4_1_RATIO/`  
  Planeamiento y variantes de ratio auxiliary (Gate 4 / 4.1).
  - `PLANES/`

- `04_DIAGNOSTICO_GATE_6_Y_GATE_4_2/`  
  Diagnóstico post Gate 4.1 + curaduría visual consolidada.
  - `INFORME_DEC005_DIAGNOSTICO_COMPLETO.md`
  - `INFORME_PLAN_CURADURIA_VISUAL_DEC005_CODEX.md`
  - `CURADURIA_VISUAL/`

- `05_PLAN_POST_DIAGNOSTICO_BLOQUE_A/`  
  Plan operativo vigente para la ejecución posterior al diagnóstico.
  - `PLAN_EJECUCION_POST_DEC005_v1.1.md`
  - `PLAN_EJECUCION_POST_DEC005_CODEX.md`
  - `Grafo_modelo_run_p0_a-b-c-d.md`

- `06_GATE_4_2_RATIO_CENTRICO/`  
  Etapa dedicada a exploración de descriptores de ratios sobre foundation sano post-Bloque A.
  - `README.md`
  - `PLANES/plan_gate_4.2.md`
  - `EVIDENCIAS/`
  - `RESULTADOS/`
  - `DECISIONES/`

Compatibilidad de ruta legacy:
- `plan_gate_4.2.md` (raiz de `BIAS_CONTROL`) redirige al plan canonico en `06_GATE_4_2_RATIO_CENTRICO/PLANES/`.

- `90_ARCHIVO_REFERENCIA/`  
  Históricos, auditorías previas y material de soporte.

## Carpeta espejo para compartir

- `resultados_compartir/`
  - Es una copia curada de material visual/audiovisual para revisión rápida.
  - Está ordenada con lógica de roadmap para recorrer resultados por etapa.
  - **No se versiona en git** (intencional, para uso local de difusión/descarga).

## Regla de mantenimiento

Cuando aparezcan nuevas visualizaciones o artefactos de exposición:

1. Guardar el resultado canónico en su fase correspondiente.
2. Duplicar en `resultados_compartir/` en la etapa equivalente.
3. Actualizar el manifiesto breve dentro de `resultados_compartir/`.
