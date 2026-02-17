<div align="center">

# INDEX BIAS_CONTROL
### Navegacion por fases y estado operativo

![Estado](https://img.shields.io/badge/Estado-Actualizado-0A7E3B?style=for-the-badge)
![Foco](https://img.shields.io/badge/Foco-Escalon_1--C-F59E0B?style=for-the-badge)

</div>

> [!NOTE]
> **Corte operativo (2026-02-17):** Gate 4.3 quedó cerrado con 13 brazos (best 5ep: `d4a4=69.8%`) y run largo `d4a4-scratch` completado en 30ep (`S=83.6%`, record). Fase 5 también cerrada en UNC (`A4r`, `D4r`, `A8`, `A9`). Siguiente transición: Gate 4.4 (arquitecturas mayores) con `a4r-scratch` en cola UNC.

## Orden canónico

- `ROADMAP_BIAS_CONTROL.md`  
  Documento troncal del frente: estado, métricas y decisiones.

- `01_GATES_0_2_5/`  
  Fundación cross-modal y baseline (`Gate 2`).

- `02_GATE_3_DANN/`  
  Línea adversarial cerrada por NO-GO.

- `03_GATE_4_4_1_RATIO/`  
  Gate 4/4.1 histórico (ratio auxiliary).

- `04_DIAGNOSTICO_GATE_6_Y_GATE_4_2/`  
  Diagnóstico causal y curaduría visual de cierre.

- `05_PLAN_POST_DIAGNOSTICO_BLOQUE_A/`  
  Bloque A v1.1 (cerrado con foundation lock).

- `06_GATE_4_2_RATIO_CENTRICO/`  
  Gate 4.2 cerrado (`D4 8ep`).

- `07_GATE_4_3_RATIO_RE_CENTRICO/`  
  Gate 4.3 cerrado (13 brazos + scratch):
  - `README.md`
  - `INFORME_GATE_4_3_RATIO_RE_CENTRICO.md`
  - `plan_gate_4.3.md`
  - `INFORME COMPLETO: d4a4-scratch 30 epochs.md`

- `08_GATE_4_4_ARQUITECTURAS_MAYORES/`  
  Third Tower + MoE (siguiente bloque de arquitectura).

- `09_GATE_5_LINEA_A_BARRIDO/`  
  Barrido + FiLM + cross-modal injection.

- `10_GATE_5_LINEA_B_SHOWCASE/`  
  Validación científica y showcase (13 tests).

- `90_ARCHIVO_REFERENCIA/`  
  Material histórico y auditorías previas.

## Carpeta espejo para compartir

- `resultados_compartir/`
  - espejo local de artefactos visuales;
  - no se versiona en git;
  - mantener sincronía mínima con hitos cerrados.

## Regla de mantenimiento

Cuando cambie el estado de un gate:

1. Actualizar primero `ROADMAP_BIAS_CONTROL.md`.
2. Actualizar después este índice con el nuevo corte.
3. Alinear documentos troncales (`Proyecto_Estado_Actual`, `HANDOFF`, `bitacora`).
