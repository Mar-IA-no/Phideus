<div align="center">

# INDEX BIAS_CONTROL
### Navegacion por fases y estado operativo

![Estado](https://img.shields.io/badge/Estado-Actualizado-0A7E3B?style=for-the-badge)
![Foco](https://img.shields.io/badge/Foco-Gate_6_AMT-1F6FEB?style=for-the-badge)

</div>

> [!NOTE]
> **Corte operativo (2026-03-08):** Gate 4.4 permanece cerrado como bloque arquitectural, Gate 4.5 queda en cierre operativo y **Gate 5B ya quedó completamente cerrado**: `Test02` cerró `4/4`, `Test11` cerró `2/2`, `Test13G-B` cerró `4/4` y la lectura final quedó fijada como “ventaja geométrica, no de feature richness”. **Gate 6 AMT** ya tiene una referencia local completa y un preflight UNC `v5` que obliga a checkpoint + auto-resubmit. **Gate 7** ya no está solo en fase de probe: `Exp 7.0` quedó completo (`MERT-330M=0.850`, `MERTLite=0.734`, `MERT-95M=0.659`) y `7.1a` ya cerró su pilot negativo útil (`75.0% ≈ 75.2%`). **Gate 8** ya dejó sus dos primeros brazos medidos localmente (`a4r-ctrl=79.2%`, `a4r-pcm=80.0%`) y migró los tres restantes a UNC. Gate 5A queda replanteado como línea oportunista, sin bloquear Escalón 2, que ya dejó atrás `S2-P2-control` y pasó a su fase descriptor-guided.
>
> **Navegación de Escalón 1:** para el mapa unificado del brazo Shazam + brazo neural usar `../ESCALON_1/INDICE_ESCALON1_COMPLETO.md`; este índice cubre solo `BIAS_CONTROL/`.

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
  Third Tower + FiLM + MoE (cerrado).

- `09_GATE_4_5_LR_SCHEDULE_OPTIMIZATION/`  
  Optimización de scheduler/LR y ventana temporal (50ep/60ep).

- `10_GATE_5_LINEA_A_BARRIDO/`  
  Replanteo Gate 5A: conditioned projections + combinatorios oportunistas.

- `11_GATE_5_LINEA_B_SHOWCASE/`  
  Validacion cientifica y showcase (13 tests), ya cerrada: incluye A/B pre-projection, cierre multi-seed, cierre causal de `Test02`, lectura negativa de `13G-B` y el informe completo del gate.

- `12_GATE_6_AMT/`  
  Validación downstream por AMT: baseline `Transkun` ya verificado, decoder VICReg enviado a UNC y línea `Transkun+A4` preparada.
  - `README.md`
  - `Explicacion_gate6.md`
  - `Briefing_para_claude_unc.md`

- `13_GATE_7_MERT_PROBE/`
  Gate 7 completo: probe lineal sobre MERTLite/MERT-95M/MERT-330M para medir accesibilidad de la envolvente espectral asociada a `A4`.
  - `README.md`
  - `Explicacion_resultados_fase0.md`
  - `DEBATES_FINALES.md`

- `14_GATE_7.1/`
  Gate 7.1 ya con `7.1a` cerrado y `7.1b` condicional: bifurcación explícita entre `D0` pilot con backbone congelado y una variante nueva `a4r-mert`.
  - `README.md`
  - `Plan_implementacion.md`

- `15_GATE_8_CONDITIONED_PROJECTIONS/`
  Promotion operativa de Gate 5A/C1: FiLM en projection heads. `a4r-ctrl` y `a4r-pcm` ya cerraron localmente; `pcd-zero`, `pcd` y `pca` siguen su cierre en UNC.
  - `README.md`

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
3. Alinear documentos troncales aplicables (`Proyecto_Estado_Actual`, `INDICE_DOCUMENTACION`, `bitacora`).
