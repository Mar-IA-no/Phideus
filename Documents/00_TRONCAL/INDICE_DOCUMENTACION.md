<div align="center">

# Índice de Documentación
### Proyecto Phideus v5.0

![Scope](https://img.shields.io/badge/Scope-Project_Documentation-1F6FEB?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-0A7E3B?style=for-the-badge)
![Updated](https://img.shields.io/badge/Updated-2026--06--28-F59E0B?style=for-the-badge)

</div>

> [!IMPORTANT]
> **Propósito**: referencia rápida de documentación operativa y de investigación.

## Navegación rápida

- [Documentos Troncales (Tier A)](#documentos-troncales-tier-a)
- [Documentos Principales](#documentos-principales)
- [Libro HIT](#libro-hit)
- [Skills Compartidas](#skills-compartidas)
- [Escalón 1: MAESTRO (Audio ↔ MIDI)](#escalón-1-maestro-audio--midi)
- [Escalón 2: Speech ↔ EGG](#escalón-2-speech--egg)
- [Voz Expresiva Phideus](#voz-expresiva-phideus)
- [Atención Armónica](#atención-armónica)
- [BIAS_CONTROL: Cross-Modal Learning con Control de Sesgo](#bias_control-cross-modal-learning-con-control-de-sesgo)
- [UOEMD / Rosetta (Histórico - NO-GO)](#uoemd--rosetta-histórico---no-go)
- [Experimentos Generales](#experimentos-generales)
- [Estructura de Directorios](#estructura-de-directorios)

---

## Documentos Troncales (Tier A)

Estos son los únicos documentos que llevan diseño visual reforzado de forma sistemática.

| Documento | Rol |
|-----------|-----|
| `README.md` | Entrada principal del repositorio |
| `Documents/00_TRONCAL/INDICE_DOCUMENTACION.md` | Mapa global de documentación |
| `Documents/00_TRONCAL/Proyecto_Estado_Actual.md` | Estado ejecutivo y decisiones vigentes |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md` | Plan maestro del experimento principal |

---

## Documentos Principales

| Documento | Ubicación | Descripción |
|-----------|-----------|-------------|
| **Estado Actual** | `Documents/00_TRONCAL/Proyecto_Estado_Actual.md` | Estado global del proyecto, ya sincronizado con `d4a4=84.0%±2.7pp` sobre 5 training seeds, Gate 6 `Transkun+A4` cerrado negativamente, Gate 10 completo, null mecanístico inicial de Escalón 2 ya cerrado, `Voz Expresiva Phideus` con `ZH` ya corrido y cierre analítico todavía pendiente, Atención Armónica con `Fase 0` y `0.5` ya cerradas, y Escalón 3 ya con línea geométrica `P5/P6` consolidada |
| **Este índice** | `Documents/00_TRONCAL/INDICE_DOCUMENTACION.md` | Mapa de documentación |
| **Bitácora** | `Documents/00_TRONCAL/bitacora_desarrollo.md` | Log de desarrollo |
| **Protocolo Codex ↔ Claude** | `Documents/00_TRONCAL/PROTOCOLO_OPERATIVO_CODEX_CLAUDE.md` | Reparto operativo recomendado: Codex como dueño de método/auditoría/documentación y Claude como dueño de implementación/ejecución/monitoreo |
| **Marco epistemológico** | `MARCO_EPISTEMOLOGICO_PHIDEUS.md` | Posición metodológica estable del programa |
| **Libro HIT (repo público)** | [AlterMundi/harmonic-information-theory](https://github.com/AlterMundi/harmonic-information-theory) | Repositorio público del libro HIT: manuscrito, arquitectura editorial, bibliografía de trabajo, fuente LaTeX y edición web en `hit.altermundi.net` |
| **Skills compartidas** | `Documents/Skills/README.md` | Índice público de skills reutilizables |
| **Paper** | `Documents/03_FRENTES_CERRADOS/UOEMD/UOEMD_Roseta_v2.2/PHIDEUS_RESEARCH_PROGRAM_2026.md` | Paper técnico de referencia |
| **★ Informe Histórico** | `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md` | **NUEVO**: Historia completa de representaciones de ratios |
| **Backpropagando Phideus** | `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/BACKPROPAGANDO_PHIDEUS.md` | Ideas y redefiniciones en discusión (no implementadas) |

---

## Libro HIT

El repo ya no contiene fisicamente el libro HIT. Esa formulacion larga vive ahora en un repositorio público independiente, pero sigue funcionando como pieza complementaria de la documentacion canonica de Phideus.

| Documento | Ubicación | Contenido |
|-----------|-----------|-----------|
| **Repositorio del libro HIT** | [AlterMundi/harmonic-information-theory](https://github.com/AlterMundi/harmonic-information-theory) | Repo público donde se mantienen la formulación larga del programa, la arquitectura editorial, la bibliografía de trabajo, la fuente LaTeX y la edición web del libro |

---

## Skills Compartidas

### Estado: 🟢 Índice público ya abierto. El repo comparte skills reutilizables de operación HPC/SLURM bajo `Documents/Skills/`

| Documento | Ubicación | Contenido |
|-----------|-----------|-----------|
| **Índice de skills** | `Documents/Skills/README.md` | Guía de instalación y catálogo de skills compartidas |
| **validate-sbatch** | `Documents/Skills/validate-sbatch/SKILL.md` | Skill pública para validar scripts SLURM antes de submitir |
| **slurm-handbook** | `Documents/Skills/slurm-handbook/SKILL.md` | Compendio operativo SLURM en formato skill |

---

## Escalón 1: MAESTRO (Audio ↔ MIDI)

### Estado: ✅ Escalón 1 cerrado y documentado en dos brazos complementarios: Shazam (1-A) con cierre formal e índice maestro nuevos, y BIAS_CONTROL (1-B/1-C) con Gate 5B ya cerrado y Gate 6 AMT activo como validación downstream

Punto de entrada canónico del escalón completo:
- `Documents/01_FRENTES_ACTIVOS/ESCALON_1/INDICE_ESCALON1_COMPLETO.md`

Decisión estructural vigente:
- no mover ni renombrar `Documents/01_FRENTES_ACTIVOS/ESCALON_1/` ni `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/`;
- `ESCALON_1/` documenta el brazo Shazam;
- `BIAS_CONTROL/` concentra la evidencia científica principal del brazo neural.

### Documentación

| Documento | Ubicación | Contenido |
|-----------|-----------|-----------|
| **Índice maestro Escalón 1** | `Documents/01_FRENTES_ACTIVOS/ESCALON_1/INDICE_ESCALON1_COMPLETO.md` | Punto de entrada unificado para Shazam + DANN + BIAS_CONTROL |
| **Cierre formal Shazam** | `Documents/01_FRENTES_ACTIVOS/ESCALON_1/CIERRE_ESCALON1_SHAZAM.md` | Cronología completa, resultados controlados, causa raíz y lecciones |
| Cronología histórica del brazo Shazam | `Documents/01_FRENTES_ACTIVOS/ESCALON_1/RESULTADOS_ESCALON_1.md` | Fases 1-11 reencuadradas como historial del brazo |
| Plan de implementación | `Documents/01_FRENTES_ACTIVOS/ESCALON_1/01_PLANIFICACION/Plan_implementacion.md` | 6 gates del experimento original |
| Plan de validación H3 | `Documents/01_FRENTES_ACTIVOS/ESCALON_1/01_PLANIFICACION/PLAN_VALIDACION_H3.md` | 4 fases: auditoría, replicación, escala, pipeline |
| Plan análisis errores | `Documents/01_FRENTES_ACTIVOS/ESCALON_1/01_PLANIFICACION/PLAN_ANALISIS_ERRORES.md` | 5 fases de análisis |
| Recomendaciones GPT | `Documents/01_FRENTES_ACTIVOS/ESCALON_1/02_CONSULTAS_GPT/Extractor_nuevos_enfoques_GPT5.2Think.md` | Specs de Route A y Route B |
| Resultados piloto Route A/B | `Documents/01_FRENTES_ACTIVOS/ESCALON_1/03_INFORMES_EXPERIMENTOS/RESULTADOS_NUEVOS_ENFOQUES.md` | N=10 original, con caveat de auditoría posterior |
| **Auditoría Fase A** | `Documents/01_FRENTES_ACTIVOS/ESCALON_1/03_INFORMES_EXPERIMENTOS/AUDITORIA_FASE_A.md` | Bug `t_anchor` y corrección de métricas infladas |
| **Informe Fases A-B** | `Documents/01_FRENTES_ACTIVOS/ESCALON_1/03_INFORMES_EXPERIMENTOS/INFORME_FASES_A_B.md` | Resultados corregidos + replicación N=20 |
| **Informe análisis errores** | `Documents/01_FRENTES_ACTIVOS/ESCALON_1/03_INFORMES_EXPERIMENTOS/INFORME_ANALISIS_ERRORES.md` | Diagnóstico completo y límite estructural |

### Scripts Principales

| Script | Propósito | Uso |
|--------|-----------|-----|
| `test_retrieval_routes.py` | Test Shazam-style retrieval | `python test_retrieval_routes.py --input-dir <dir>` |
| `analyze_errors.py` | Análisis de errores | `python analyze_errors.py --route A` |
| `analyze_overlap_deep.py` | Análisis de componentes | `python analyze_overlap_deep.py` |
| `ablation_chord_only.py` | Ablation por tipo token | `python ablation_chord_only.py` |
| `diagnose_hash_collision.py` | Diagnóstico colisiones | `python diagnose_hash_collision.py` |
| `compare_routes.py` | Comparación overlap | `python compare_routes.py` |

### Extractores

| Archivo | Descripción | Config actual |
|---------|-------------|---------------|
| `src/extractors/event_based_extractor.py` | **Route A**: Event-Based | DT_BIN=10, CHORD_TOL=5, BOOST=2.0 |
| `src/extractors/improved_tf_extractor.py` | **Route B**: Improved TF | Original |

### Datos

| Directorio | Contenido | Tamaño |
|------------|-----------|--------|
| `experiments/un_audio_un_midi/Varios_pares/` | 10 pares originales | ~2GB |
| `experiments/un_audio_un_midi/muestra_replicacion/` | 20 pares replicación | ~4GB |
| `data/maestro_v3/maestro-v3.0.0/` | Dataset completo | 121GB |

### Resultados Clave

| Fase | Métricas | Documento |
|------|----------|-----------|
| N=10 original (con bug) | 71-80% accuracy | `03_INFORMES_EXPERIMENTOS/RESULTADOS_NUEVOS_ENFOQUES.md` |
| N=10 corregido | 32-42% accuracy | `03_INFORMES_EXPERIMENTOS/AUDITORIA_FASE_A.md` |
| N=20 replicación | 21-27% accuracy | `03_INFORMES_EXPERIMENTOS/INFORME_FASES_A_B.md` |
| N=20 post-mejoras | 27% accuracy, 5.4x random | `03_INFORMES_EXPERIMENTOS/INFORME_ANALISIS_ERRORES.md` |

---

## Escalón 2: Speech ↔ EGG

### Estado: 🔵 Frente abierto con `S2-P0` y `S2-P1` completos, `S2-P2-control`, `S2-P2-main`, `S2-P2.5` y `S2-P2.5b` ya absorbidos en un null mecanístico inicial cerrado, y `S2-P3` ya corrido en primera pasada con `WavLM-Large` frozen; la tarea viva es el diagnóstico comparativo `P2 vs P3`

### Documentación

| Documento | Ubicación | Contenido |
|-----------|-----------|-----------|
| **README Escalón 2** | `Documents/01_FRENTES_ACTIVOS/ESCALON_2/README.md` | Estado canónico del frente Speech↔EGG, con null mecanístico inicial ya cerrado y `S2-P3` ya absorbido como primera pasada completa (`WavLM-Large` frozen) |
| **Roadmap Escalón 2** | `Documents/01_FRENTES_ACTIVOS/ESCALON_2/ROADMAP_ESCALON_2.md` | Desarrollo completo del frente, ya reencuadrado por la rectificación de armonía natural, el cierre formal de `P2.5/P2.5b`, la primera pasada de `P3` y el diagnóstico pendiente `P2 vs P3` |
| **Plan implementación** | `Documents/01_FRENTES_ACTIVOS/ESCALON_2/PLAN_IMPLEMENTACION_ESCALON2.md` | Plan base de apertura; hoy queda como documento histórico/superseded frente al README, roadmap y rectificación epistemológica |
| **Rectificación armonía natural** | `Documents/01_FRENTES_ACTIVOS/ESCALON_2/S2_P2/plan_rectificacion_armonia_natural.md` | Rediseño descriptorial de `S2-P2-main` con `V4-lin`, `H-series` y `A4-16k` |
| **★ Predicciones epistemológicas P2.5** | `Documents/01_FRENTES_ACTIVOS/ESCALON_2/S2_P2/PREDICCIONES_EPISTEMOLOGICAS_P25.md` | **NUEVO**: Preregistro interpretativo con regla operativa (bootstrap pareado Δ), matriz de predicciones y guardrails para nulls |
| **Auditoría epistemológica de segundo orden** | `Documents/01_FRENTES_ACTIVOS/ESCALON_2/Plan_revision_epistemologica.md` | Documento de auditoría que consolida la rectificación del frente y explicita jerarquía documental, taxonomía y triggers arquitectónicos |
| **Rosetta Triplescaloneta** | `Documents/00_TRONCAL/ROADMAP_GENERAL/Rosetta_triplescaloneta.md` | Justificación macro del escalón y estado actual dentro del programa |
| **Proyecto Estado Actual** | `Documents/00_TRONCAL/Proyecto_Estado_Actual.md` | Corte ejecutivo con `S2-P0` ya integrado |
| **Contracts roadmap** | `Documents/00_TRONCAL/ROADMAP_GENERAL/contracts/README.md` | Apertura del andamiaje de contratos para instancias futuras |

### Artefactos confirmados

| Artefacto | Ubicación | Estado |
|-----------|-----------|--------|
| Manifest clip-level | `data/lombard/manifest.json` | `9,120` clips, split speaker `28/5/5` |
| Segment index | `data/lombard/segment_index.json` | `108,536` segmentos con segmentación canónica `2s / hop 0.5s` |
| Alignment audit | `data/lombard/alignment_audit.json` | `lag_correction_samples=0`, `voiced_threshold=0.1494`, `0` clipping |
| Script S2-P0 | `experiments/bias_control/escalon2/s2_p0_manifest.py` | Ingesta, split y auditoría inicial |
| Script S2-P1 | `experiments/bias_control/escalon2/s2_p1_baseline_linear.py` | Baseline lineal sobre el protocolo ya congelado |
| Resultados S2-P1 | `data/lombard/p1_results/p1_results_noise0.json` | `CCA S=64.4%`, `raw cosine S=46.8%`, CI grouped |
| Control neural cerrado | `data/lombard/d0_control/` | `S2-P2-control` completo (`best S=77.8% @ ep25`, `CI=[72.0%, 80.8%]`) |
| Plan activo `S2-P2-main` | `Documents/01_FRENTES_ACTIVOS/ESCALON_2/S2_P2/plan_rectificacion_armonia_natural.md` | Rectificación por armonía natural con familias descriptoriales primarias |
| Preregistro P2.5 | `Documents/01_FRENTES_ACTIVOS/ESCALON_2/S2_P2/PREDICCIONES_EPISTEMOLOGICAS_P25.md` | Matriz de predicciones pre-registrada, regla operativa CI_Δ |

### Estado operativo real

- Dataset local inspeccionado: French Lombard `v1.1`, `38` speakers (`20F/18M`), ~`20h`.
- Piloto limpio disponible: `noise0` con `19,910` segmentos train, `3,624` validation y `3,629` test.
- Positivo canónico: misma ventana temporal del mismo clip (`speech[t0:t1] ↔ egg[t0:t1]`).
- Baseline lineal ya validado: `CCA` supera el azar (`7.8%`) por un margen amplio (`S=64.4%`).

---

## Voz Expresiva Phideus

### Estado: 🟡 Frente exploratorio ya abierto y ya con `Fase 0A`, `0B` y `1 EN` cerradas, más la réplica `ZH` ya ejecutada completa. La lectura vigente ya no es sólo “hay señal descriptorial”: `WavLM-only` levantó el techo del stack clásico y `concat` aportó robustamente sobre baseline en `N-strict`; la tarea viva pasó a ser el cierre analítico `EN ↔ ZH`, no sumar training sin consolidación

| Documento | Ubicación | Contenido |
|-----------|-----------|-----------|
| **README del frente** | `Documents/01_FRENTES_ACTIVOS/Voz_Expresiva_Phideus/README.md` | Estado canónico del frente: `Fase 1 EN` ya cerrada, `ZH` ya corrido, y cierre translingüístico todavía pendiente de consolidación analítica |
| **Roadmap general** | `Documents/01_FRENTES_ACTIVOS/Voz_Expresiva_Phideus/ROADMAP_VOZ_EXPRESIVA_PHIDEUS.md` | Estructura del frente por carriles y fases; `0A/0B/1 EN` ya cerradas, `ZH` ya ejecutado, y pregunta viva = lectura consolidada `EN ↔ ZH` |
| **Plan archivado Fase 1 ZH** | `Documents/01_FRENTES_ACTIVOS/Voz_Expresiva_Phideus/PLAN_FASE_1_ZH.md` | Plan canónico de la réplica `ZH`, con nota de estado post-training y pendientes de cierre analítico |
| **Antecedente exploratorio** | `Documents/01_FRENTES_ACTIVOS/EIR-EMR/README.md` | Apertura temprana preservada como antecedente conceptual; no es el nombre vigente del frente |
| **Pipeline de descriptores** | `src/voz_expresiva/README.md` | Módulo de extracción y composición descriptorial usado en `Fase 0A` |
| **Scripts 0A/0B/1** | `experiments/voz_expresiva/` | Extracción, análisis, clasificación clásica, precaches `WavLM` y training `SSL` del frente |
| **Reporte Fase 0A** | `data/visualizations/voz_expresiva/0A/REPORTE_0A.md` | Lectura exploratoria descriptor-only: señal univariada de la familia `A` frente al control `C` |
| **Reporte Fase 0B** | `data/voz_expresiva/0B/REPORTE_0B.md` | Lectura comparativa `N-strict` vs `N-adapt`: especificidad ratio sí, validación fuerte estricta todavía no |
| **Reporte Fase 1** | `data/voz_expresiva/1/REPORTE_1.md` | Cierre `SSL` sobre `ESD` English: `WavLM-only` como baseline real, `concat` positivo robusto en `N-strict`, `CKA` como lectura geométrica |
| **Explicación pipeline Fase 1** | `Documents/01_FRENTES_ACTIVOS/Voz_Expresiva_Phideus/EXPLICACION_PIPELINE_FASE_1.md` | Explicación pedagógica del pipeline `WavLM` + familia `A` + mecanismos `concat/FiLM/xattn` |

### Lectura útil del corte

- `Fase 0A` dejó un **GO direccional**: `A` superó al control `C` por ~5× en `eta²`.
- `Fase 0B` dejó una lectura dual:
  - en `N-strict`, el stack descriptor-only no valida generalización honesta a hablante nuevo;
  - en `N-adapt`, la familia `A` sí muestra especificidad frente al control y una mejora pequeña sobre `eGeMAPS`.
- `Fase 1` ya respondió la pregunta `SSL` en inglés:
  - `WavLM-only` levanta el techo de `N-strict`;
  - `concat` agrega robustamente sobre baseline;
  - `FiLM` y `xattn` quedan positivos pero no cerrados todavía en el régimen estricto.
- La pregunta viva correcta ya no es “si WavLM sirve” ni “si hay que correr ZH” en abstracto, sino si esa lectura **sobrevive al cierre analítico `EN ↔ ZH`** antes de pasar a un dominio naturalístico.

---

## Atención Armónica

### Estado: 🟡 Frente incubado con `Fase 0` y `Fase 0.5` ya cerradas. El `sweep` `v2.1` ya pasó, el `final_pool` quedó congelado con gate `PASS`, el training decisivo cerró `54/54` y el post-audit mostró que el cuello de `OOD-poly` está en `connected-components`, no en la calibración de `τ`

| Documento | Ubicación | Contenido |
|-----------|-----------|-----------|
| **README del frente** | `Documents/01_FRENTES_ACTIVOS/Atencion_Armonica/README.md` | Estado canónico local del frente: origen, correcciones `v1/v2/v2.1`, cierre `Fase 0` y `0.5`, resultado dual pair-state/triangle y cuello actual del clusterer |
| **Explicación arquitectónica** | `Documents/01_FRENTES_ACTIVOS/Atencion_Armonica/Explicacion_arq_RNA_codex.md` | Explicación conceptual de `Harmonic Pairformer`: plano token, plano par, `triangle update`, geometría relacional y caminos derivados |
| **Explicación Fase 0.5** | `Documents/01_FRENTES_ACTIVOS/Atencion_Armonica/Explicacion_fase_0_5_calibracion_codex.md` | Lectura conceptual del último hallazgo: el problema no era `τ`, sino la lectura por `connected-components` |
| **Roadmap general** | `Documents/01_FRENTES_ACTIVOS/Atencion_Armonica/ROADMAP_ATENCION_ARMONICA.md` | Marco del frente y lectura actual: `GO` acotado, `Fase 0.5` ya cerrada y foco siguiente en clusterers globales antes de CQT/audio |
| **Plan Fase 0.5** | `Documents/01_FRENTES_ACTIVOS/Atencion_Armonica/PLAN_FASE_0_5_CALIBRACION.md` | Plan ejecutado del post-audit: re-run con matrices/checkpoints, calibradores, reglas deployables y oráculos separados |
| **Plan Fase 0 v2.1** | `Documents/01_FRENTES_ACTIVOS/Atencion_Armonica/PLAN_FASE_0_v2_1.md` | Plan operativo ya ejecutado con `β>0`, amplitud randomizada, gate de feature-triviality, combo congelada, `final_pool` pasado y cierre threshold-free |
| **Plan v1 superseded** | `Documents/01_FRENTES_ACTIVOS/Atencion_Armonica/PLAN_FASE_0_v1_superseded.md` | Registro del diseño original que quedó invalidado por feature-triviality y se preserva solo como trazabilidad metodológica |

### Lectura útil del corte

- `v1` y `v2` no fallaron por detalles cosméticos sino por un problema de validez experimental: el dataset seguía dejando canales cerrados demasiado fuertes para `A-rich`.
- `v2.1` rompió ese problema con `β>0`, amplitud randomizada y un gate explícito de feature-triviality sobre todo lo que recibe `A-rich`, incluido `ratio_class_id`.
- El frente ya tiene una lectura de `Fase 0` y `0.5`: `B-minus ≫ A-rich` confirma el valor del pair-state, `B ≫ B-shuffle` muestra que la estructura del triángulo no es capacidad pura, `B > B-local` en `OOD-poly` threshold-free justifica un `GO` acotado, y el post-audit reubica el cuello desde `τ` hacia `connected-components` y la lectura global de la partición.

---

## Escalón 3: Audio XY ↔ Figuras de Lissajous

### Estado: 🟡 Frente activo ya abierto y ya corrido en su primera línea geométrica. `E3-P0` ya dejó generador reproducible y dataset materializado, `E3-P1` ya cerró aprendibilidad por `ratio`, `E3-P2` ya fijó baseline dual (`flat` canónico + `cqtshift` alternativo), `E3-P4` ya fue corrido, `P5/P6` ya devolvieron una primera lectura completa y `P5-cqtshift` queda como mejor brazo OOD actual

| Documento | Ubicación | Contenido |
|-----------|-----------|-----------|
| **README Escalón 3** | `Documents/01_FRENTES_ACTIVOS/ESCALON_3/README.md` | Estado canónico del frente Lissajous: baseline dual consolidado y primera lectura `P5/P6` ya incorporada |
| **Roadmap Escalón 3** | `Documents/01_FRENTES_ACTIVOS/ESCALON_3/ROADMAP_ESCALON_3.md` | Hoja de ruta del frente, ya reencuadrada por el baseline dual y por la primera lectura completa de la línea geométrica |
| **Resultados E3-P4** | `Documents/01_FRENTES_ACTIVOS/ESCALON_3/Resultados_E3_P4.md` | Resultado consolidado del régimen de probes: lectura útil sobre latente plano, sin cierre suficiente contra la línea geométrica |
| **Resultados E3-P5/P6** | `Documents/01_FRENTES_ACTIVOS/ESCALON_3/Resultados_E3_P5_P6.md` | Resultado consolidado de la primera pasada geométrica: `P5-cqtshift` mejor brazo OOD actual y `P6` no ganador bajo la receta vigente |
| **Plan P5/P6** | `Documents/01_FRENTES_ACTIVOS/ESCALON_3/PLAN_E3_P5_P6_GEOMETRIA_NO_PLANA.md` | Especificación metodológica completa de la línea geométrica no plana: `P5` mixto y `P6` toroidal completo |
| **Briefing operativo P5/P6** | `Documents/01_FRENTES_ACTIVOS/ESCALON_3/BRIEFING_OPERATIVO_P5_P6.md` | Versión corta y ejecutable del plan geométrico: orden de implementación, invariantes, entregables y checkpoints |
| **Plan Claude** | `Documents/01_FRENTES_ACTIVOS/ESCALON_3/Plan_Claude.md` | Plan operativo histórico de `E3-P0 + P1 + P2`; hoy sirve como trazabilidad, no como lectura canónica del estado vigente |
| **Plan inaugural Codex** | `Documents/01_FRENTES_ACTIVOS/ESCALON_3/Legacy/Plan_inaugural_construccion_dataset_Codex.md` | Diseño sintético-first del dataset; queda como documento de origen y criterio de lectura |
| **Protocolo operativo Codex ↔ Claude** | `Documents/00_TRONCAL/PROTOCOLO_OPERATIVO_CODEX_CLAUDE.md` | Regla práctica para correr `P4` y frentes siguientes sin mezclar diseño metodológico con ejecución operativa |
| **Generador E3-P0** | `experiments/escalon3/generate_lissajous_dataset.py` | Generador reproducible del banco canónico de scenes Lissajous |
| **Dataset materializado** | `data/escalon3/scenes/` | Banco `E3-P0` ya generado (`6,016` scenes; splits IID + OOD) |

---

## Escalón 4: ECG ↔ PPG

### Estado: ⚪ Frente todavía conceptual. Pasa a ocupar el lugar de expansión fisiológica fuera de acústica y se referencia hoy desde los roadmaps generales, no desde una carpeta propia de frente activo

| Documento | Ubicación | Contenido |
|-----------|-----------|-----------|
| **Rosetta Triplescaloneta** | `Documents/00_TRONCAL/ROADMAP_GENERAL/Rosetta_triplescaloneta.md` | Marco general del escalón fisiológico y su lugar dentro del programa |
| **Plan maestro v1.1** | `Documents/00_TRONCAL/ROADMAP_GENERAL/PLAN_AVANCE_TRIPLESCALONETA_v1.1.md` | Encaje metodológico del escalón fisiológico en la secuencia general |

---

## BIAS_CONTROL: Cross-Modal Learning con Control de Sesgo

### Estado: ✅ **Escalón 1-A/B completado** — Gate 3 cerrado. ✅ **Escalón 1-C cerrado** (post-diagnóstico, Bloque A v1.1 cerrado con D-02 y lock formal; Gate 4.2/4.3/4.4 cerrados; Gate 5B ya cerrado con `Test05`, `Test02` 4/4, `Test11` ya integrado en su lectura completa y `Test13G-B` 4/4). 🔵 **Gate 6 AMT activo** ya con la rama `Transkun+A4` cerrada negativamente (`Exp A` + `Exp B`) y `Exp C` como única línea abierta, ✅ **Gate 8** ya cerrado `5/5` como línea positiva paralela, 🟡 **Gate 9 / revisión `A10`** ya con datos retrospectivos, y ✅ **Gate 10** ya documentado como barrido causal completo con lectura final `concat > FiLM/pca >> attn_bias`.

Marco de referencia:
- `Documents/00_TRONCAL/ROADMAP_GENERAL/Rosetta_triplescaloneta.md`
- `Documents/00_TRONCAL/ROADMAP_GENERAL/PLAN_AVANCE_TRIPLESCALONETA_v1.1.md` (vigente)
- `Documents/00_TRONCAL/ROADMAP_GENERAL/PLAN_AVANCE_TRIPLESCALONETA_v1.md` (archivado historico)
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`

### Documentación

| Documento | Ubicación | Contenido |
|-----------|-----------|-----------|
| **Roadmap** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md` | Plan maestro y criterios GO/NO-GO (incluye cierre Gate 4.3 y transición a 4.4/5) |
| **Índice por fases (nuevo)** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/INDEX_BIAS_CONTROL.md` | Navegación del roadmap por árbol de directorios |
| **Auditoría Codex (v1 + addendums)** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/90_ARCHIVO_REFERENCIA/AUDITORIA_BIAS_CONTROL_CODEX.md` | Auditoría histórica + addendums operativos |
| **Plan post-diagnóstico v1.1** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/05_PLAN_POST_DIAGNOSTICO_BLOQUE_A/PLAN_EJECUCION_POST_DEC005_v1.1.md` | Plan operativo de Bloque A (cerrado con D-02) |
| **Plan Gate 4.2 (final)** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/06_GATE_4_2_RATIO_CENTRICO/plan_gate_4.2.md` | Exploración ratio-céntrica post Bloque A (v2.1) |
| **Plan Gate 4.3** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/07_GATE_4_3_RATIO_RE_CENTRICO/plan_gate_4.3.md` | Bloque causal corto bifurcado (MIDI temperado / Audio armonía natural / Dual) |
| **Gate 4.4 (arquitecturas mayores)** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/08_GATE_4_4_ARQUITECTURAS_MAYORES/README.md` | Third Tower + FiLM + MoE con Ratio Expert |
| **Gate 4.5 (LR schedule optimization)** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/09_GATE_4_5_LR_SCHEDULE_OPTIMIZATION/README.md` | Corridas 50ep/60ep + comparación de scheduler |
| **Gate 5 Linea A** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/10_GATE_5_LINEA_A_BARRIDO/README.md` | Replanteo Gate 5A: conditioned projections + combinatorios oportunistas |
| **Gate 5 Linea B** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/README.md` | Batería de validación científica (13 tests) |
| **Gate 6 AMT** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/12_GATE_6_AMT/README.md` | Validación downstream por AMT: baseline `Transkun`, rama `Transkun+A4` ya cerrada negativamente (`Exp A` + `Exp B`) y decoder VICReg aún activo (`Exp C`) |
| **Gate 8 conditioned projections** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/15_GATE_8_CONDITIONED_PROJECTIONS/README.md` | Promotion operativa de Gate 5A/C1: FiLM en projection heads, ya cerrada `5/5` con `pcd > pca > pcd-zero > pcm > ctrl` |
| **Gate 9 natural harmony** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/16_GATE_9_NAT_HARM_DESCRIPTOR/PLAN_GATE9.md` | Piloto retrospectivo `A7r/A9r` ya con datos para releer armonía natural en música bajo el mecanismo ganador |
| **Gate 10 mechanism sweep** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/17_GATE_10_MECHANISM_SWEEP/README.md` | Barrido descriptor × mecanismo para separar contenido de inyección en audio-only, ya completo con lectura final `concat > FiLM/pca >> attn_bias` |
| **Revisión A10** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/16_GATE_9_NAT_HARM_DESCRIPTOR/PLAN_GATE9_DESCRIPTOR_REVISION.md` | Extensión continua ontology-free (`A10d/A10e`) y controles explícitos para música / voz |
| **Explicación Gate 6** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/12_GATE_6_AMT/Explicacion_gate6.md` | Lectura narrativa de por qué Gate 6 abre después del cierre Gate 5B |
| **Briefing Gate 6 para UNC** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/12_GATE_6_AMT/Briefing_para_claude_unc.md` | Referencia operativa e histórica de UNC para Gate 6; hoy debe leerse con `Exp A` + `Exp B` ya cerrados negativamente y `Exp C` como única línea downstream abierta |
| **Explicación Pre-Proj A/B** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/Explicaccion_pre-projection_test.md` | Lectura del bottleneck de proyección e information retention ratio (`D0` vs `a4r`) |
| **Explicación Test 13G** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/Explicacion_test_13G.md` | Explicación narrativa del dual-objective generative encoder, el cierre de su Phase A y el pivot hacia features pre-pooling |
| **Explicación Test 13G Phase B** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/Explicacion_test_13G_faseB.md` | Diseño y lectura metodológica del decoder post-hoc sobre features pre-pooling |
| **Informe Gate 5B completo** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/INFORME_COMPLETO_GATE5B.md` | Síntesis exhaustiva del cierre Gate 5B con cadena de evidencia completa |
| **Resultados Test02 + 13G-B** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/Explicacion_resultados_test13g_y_02.md` | Lectura detallada del cierre causal de Test02 y del resultado negativo de 13G-B |
| **Informe Gate 5B (corte inicial Test01/Test12)** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/INFORME_EJECUCION_TEST01_TEST12_2026-02-25.md` | Corte inicial de cierre operativo (scoreboard + causal ablation + avance temprano de transposition) |
| **Exploración Foundation (script)** | `experiments/bias_control/explore_foundation.py` | Probes cualitativos (retrieval, UMAP, pairs, similarity, per-piece, interpolation) post-lock |
| **Estructura Gate 4.2** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/06_GATE_4_2_RATIO_CENTRICO/README.md` | Estructura operativa de la fase (planes, evidencias, resultados, decisiones) |
| **Curaduría visual** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/04_DIAGNOSTICO_GATE_6_Y_GATE_4_2/CURADURIA_VISUAL/INDEX_VISUAL.md` | Snapshot visual técnico de resultados cerrados |
| **★ Informe Gate 3 completo** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/02_GATE_3_DANN/INFORME_GATE3_COMPLETO.md` | **Evaluación comparativa 4 Runs + decisión** |
| **Comparación Gate 3** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/02_GATE_3_DANN/COMPARISON_GATE3.md` | Tabla comparativa (6 checkpoints) |
| **Informe Runs A/B** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/02_GATE_3_DANN/INFORME_GATE3_DANN_SIN_NORM.md` | Runs A (sin norm) y B (F.normalize) |
| **Informe Gate 2** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/01_GATES_0_2_5/GATE_2_FOUNDATION/INFORME_GATE2_COMPLETO.md` | Informe exhaustivo Gate 2 |
| **Fast test results** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/90_ARCHIVO_REFERENCIA/BIAS_CONTROL_FAST_TEST_RESULTS.md` | 3 epochs, Gap: 0.026 |
| **Medium test results** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/90_ARCHIVO_REFERENCIA/BIAS_CONTROL_MEDIUM_TEST_RESULTS.md` | 61 epochs, Gap: 0.478 best |
| **Plan Gate 4 (Claude)** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/03_GATE_4_4_1_RATIO/PLANES/plan_gate4.md` | Plan operativo Gate 4 |
| **Revisión Gate 4 (Codex)** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/03_GATE_4_4_1_RATIO/PLANES/plan_gate4_codex.md` | Observaciones técnicas y riesgos |
| **VibeTensor spike plan** | `Documents/02_FRENTES_PAUSADOS/VIBETENSOR_SPIKE_PLAN/VIBETENSOR_SPIKE_PLAN.md` | Plan de infraestructura (actualmente pausado) |

Nota operativa:
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/resultados_compartir/` es espejo local de visualizaciones para revisión/descarga y no se versiona en git.
- Visualizaciones 3D publicadas en `https://altermundi.github.io/Phideus/` (adaptación sobre `https://github.com/bbycroft/llm-viz`).

### Gobernanza operativa

- Claude: implementación y ejecución experimental.
- Codex: mantenimiento y actualización de documentación del repositorio.

### Módulo Principal: `src/bias_control/`

| Componente | Archivos | Descripción |
|------------|----------|-------------|
| Encoders | `encoders/mert_encoder.py`, `midi_encoder.py`, `projection.py` | MERT, Transformer MIDI, MLPs |
| Losses | `losses/dann.py` | DANN + Gradient Reversal Layer |
| Modelos | `architectures/cross_modal_model.py` | CrossModalModel con VICReg |
| Datos | `datasets/maestro_segments.py` | Dataset MAESTRO segmentado |

### Experimentos: `experiments/bias_control/`

| Script | Gate | Descripción |
|--------|------|-------------|
| `gate0_data_integrity.py` | 0 | Verificación datos y alignment |
| `gate1_intra_modal.py` | 1 | Baselines Audio→Audio, MIDI→MIDI |
| `gate2_foundation.py` | 2 | VICReg cross-modal |
| `gate2_5_embedding_analysis.py` | 2.5 | t-SNE/UMAP diagnóstico |
| `gate3_dann.py` | 3 | Domain adversarial training |
| `gate4_ratio_auxiliary.py` | 4 | Multi-view con ratios |
| `evaluate_structured_pool.py` | - | Pool estructurado (test definitivo) |
| `compare_gate3_checkpoints.py` | 3 | **Comparación 6+ checkpoints Gate 3** |
| `compare_layer_drift.py` | 6 | Drift por capas entre Gate2, RB0, RA5 y R1 |
| `extract_multigate_embeddings.py` | 6 | Extracción unificada de embeddings multi-checkpoint |
| `h426_prered_test.py` | 4.2 | Pre-red dual-domain (`P0/P1`) |
| `run_all_gates.py` | - | Orquestador completo |

### Arquitectura

```
Audio → MERT (frozen, 330M) → Projection → Embedding (256d)
MIDI  → Transformer (4L, 8H) → Projection → Embedding (256d)
       └──────────────────────────────────────────────┘
                              │
                    VICReg Loss + DANN (opcional)
```

### Comandos

```bash
# Pipeline completo
python experiments/bias_control/run_all_gates.py \
    --maestro-dir data/maestro_v3/maestro-v3.0.0 \
    --output data/bias_control
```

---

## UOEMD / Rosetta (Histórico - NO-GO)

### Estado: 🔴 Cerrado - Dataset insuficiente

### Documentación

| Documento | Ubicación | Contenido |
|-----------|-----------|-----------|
| Roadmap revisionismo | `Documents/03_FRENTES_CERRADOS/UOEMD/UOEMD_Revisionismo/ROADMAP.md` | Plan de 4 fases |
| Fase 0 | `Documents/03_FRENTES_CERRADOS/UOEMD/UOEMD_Revisionismo/Fase_0/` | Tests sintéticos |
| Fase 1 | `Documents/03_FRENTES_CERRADOS/UOEMD/UOEMD_Revisionismo/Fase_1/` | Extractor v2.2 |
| Fase 2 | `Documents/03_FRENTES_CERRADOS/UOEMD/UOEMD_Revisionismo/Fase_2/` | Re-entrenamiento |
| Fase 3A | `Documents/03_FRENTES_CERRADOS/UOEMD/UOEMD_Revisionismo/Fase_3A/` | Constellation tokens |
| Resultados v2.2 | `Documents/03_FRENTES_CERRADOS/UOEMD/UOEMD_Roseta_v2.2/` | Métricas extractor |
| Planes Claude | `Documents/03_FRENTES_CERRADOS/UOEMD/Planes Claude/` | Planes de implementación |

### Conclusión UOEMD

El dataset UOEMD (128 muestras de motor diésel) no demostró cross-modality:
- Gap pre-red: 0.691 (extractor v2.2)
- Gap post-red: 0.007 (modelo no aprende)
- Top-1 retrieval: 0.78% (= random)

---

## Experimentos Generales

### Documentación

| Documento | Ubicación | Contenido |
|-----------|-----------|-----------|
| Comparativo 4.1 vs 5.0 | `Documents/90_ARCHIVO_GLOBAL/Experimentos/REPORTE_COMPARATIVO_4.1_vs_5.0.md` | Analizadores |
| HRM vs VAE masivo | `Documents/90_ARCHIVO_GLOBAL/Experimentos/RESULTADOS_HRM_VS_VAE_MASIVO.md` | 4 arquitecturas |
| HRM training | `Documents/90_ARCHIVO_GLOBAL/Experimentos/RESULTADOS_HRM_TRAINING.md` | Hierarchical model |

### Scripts

| Script | Propósito |
|--------|-----------|
| `experiments/run_experiments_5.0.py` | Comparación 4 arquitecturas |
| `experiments/run_roseta_experiment.py` | Experimento Roseta |
| `experiments/evaluate_retrieval.py` | Evaluación retrieval |
| `experiments/evaluate_cross_reconstruction.py` | Cross-reconstruction |

---

## Código Fuente Principal

### Analizadores

| Archivo | Descripción |
|---------|-------------|
| `src/analizador/analizador_5.0.py` | Principal - escala lineal + temporal |
| `src/analizador/analizador_roseta.py` | Dual-domain para Roseta |
| `src/analizador/analizador_maestro.py` | Extracción MAESTRO (no usado) |

### Modelos RNA

| Archivo | Descripción |
|---------|-------------|
| `src/RNA/roseta_vae.py` | VAE cross-modal |
| `src/RNA/constellation_vae.py` | ConstellationVAE |
| `src/RNA/jepa_lite.py` | JEPA sin decoder |
| `src/RNA/vicreg.py` | VICReg loss |
| `src/RNA/barlow_twins.py` | Barlow Twins loss |

### Datasets

| Archivo | Descripción |
|---------|-------------|
| `src/datasets/temporal_dataset_5.py` | Loader NPZ/JSON |
| `src/datasets/roseta_dataset.py` | Loader dual-domain |
| `src/datasets/maestro_dataset.py` | Loader MAESTRO |

### Utilidades

| Archivo | Descripción |
|---------|-------------|
| `src/utils/midi_utils.py` | Parseo MIDI, piano roll |

---

## Estructura de Directorios

```text
<repo-root>/
├── Documents/
│   ├── 00_TRONCAL/                  # Índice, estado, bitácora, roadmap general
│   ├── 01_FRENTES_ACTIVOS/          # BIAS_CONTROL, Escalones 1/2/3 y Voz Expresiva
│   ├── Skills/                      # Skills públicas compartidas
│   ├── 02_FRENTES_PAUSADOS/         # VIBETENSOR spike
│   ├── 03_FRENTES_CERRADOS/         # UOEMD / Rosetta no-go
│   ├── 04_TRANSVERSAL/              # teoría, análisis externos, overviews
│   └── 90_ARCHIVO_GLOBAL/           # legado histórico y experimentos archivados
├── src/
│   ├── analizador/
│   ├── extractors/
│   ├── datasets/
│   ├── RNA/
│   └── utils/
├── experiments/
│   ├── bias_control/                # Scripts Gates cross-modal
│   ├── un_audio_un_midi/            # Scripts Escalón 1
│   └── *.py
├── data/
│   ├── maestro_v3/
│   └── datasets/
└── models/
```

---

## Comandos Frecuentes

### Setup

```bash
cd <repo-root>
source venv/bin/activate
```

### Escalón 1

```bash
# Test retrieval
python experiments/un_audio_un_midi/test_retrieval_routes.py \
    --input-dir experiments/un_audio_un_midi/muestra_replicacion

# Análisis de errores
python experiments/un_audio_un_midi/analyze_errors.py --route A

# Ablation
python experiments/un_audio_un_midi/ablation_chord_only.py
```

### Git

```bash
git status
git log --oneline -10
git diff
```

---

## Histórico de Decisiones

| Fecha | Decisión | Razón |
|-------|----------|-------|
| 2026-04-09 | **d4a4 training multi-seed + libro HIT público** | Escalón 1 fija su cierre canónico en `d4a4=84.0%±2.7pp` sobre `5` training seeds y el libro HIT pasa a funcionar como pieza pública consolidada con edición web en `hit.altermundi.net` |
| 2026-01 | NO-GO UOEMD | Dataset muy pequeño (128 muestras) |
| 2026-02-04 | Auditar experimento N=10 | Resultados sospechosamente altos |
| 2026-02-04 | Bug t_anchor encontrado | 71% → 42% accuracy |
| 2026-02-04 | Replicar con N=20 | Confirmar resultados |
| 2026-02-04 | Análisis de errores | Accuracy baja (27%) |
| 2026-02-04 | Mejoras A+B | Overlap +8pp, accuracy +0.4pp |
| 2026-02-04 | **Pausa Escalón 1** | Rendimientos decrecientes |
| 2026-02-04 | **BIAS_CONTROL** | Nuevo enfoque: soft matching con embeddings |
| 2026-02-04 | Fast test BIAS_CONTROL | Gap: 0.026 (3 epochs) |
| 2026-02-04 | Medium test inicio | 30 epochs, 200 bat/ep |
| 2026-02-04 | Migración tmux | Resume capability añadida |
| 2026-02-05 | Escalar a 1000 bat/ep | Más data coverage |
| 2026-02-05 | Recalibrar criterios (v1.3) | Pool estructurado como test definitivo |
| 2026-02-05 | **Gate 2 completado - GO** | Gap 0.478, R@10 34.4%, Hard neg 80.4% |
| 2026-02-05 | **Gate 3 smoke test - GO** | Script validado, métricas sin degradación |
| 2026-02-05 | **Gate 3 DANN training** | 30 epochs lanzado en tmux |
| 2026-02-05 | **Gate 6 añadido al roadmap** | Retroanálisis embeddings vs ratios (v1.6) |
| 2026-02-05 | Gate 3 epoch 7 **nuevo best** | Domain acc 62.7%, R@10 7.3% |
| 2026-02-06 | **Gate 3 Run A detenido ep10** | Fix normalización, lanzar Run B |
| 2026-02-06 | **Gate 3 Run B (norm) completado** | F.normalize antes domain head |
| 2026-02-06 | **Gate 3 Run C (λ=0.8) detenido ep27** | Sobre-regularización, no mejora |
| 2026-02-06 | **Evaluación comparativa completada** | 6 checkpoints, pool estructurado: Gate 2 ≈ Run C ep4 |
| 2026-02-06 | **Gate 3 Run D (λ=0.3) lanzado** | Último experimento DANN |
| 2026-02-07 | **Gate 3 Run D completado** | R@10 27.4% — peor que Gate 2 |
| 2026-02-07 | **Gate 3 CERRADO** | DANN no mejora en ningún régimen |
| 2026-02-09 | **Marco Escalón 1-A/B/C fijado** | BIAS_CONTROL se formaliza como Escalón 1; cierre con Gate 4 + Gate 6 + auditoría final |
| 2026-02-09 | **Ingreso operativo de Codex + protocolo collab v1.0** | Se crea gobernanza Claude↔Codex con ON/OFF y TURN_SUMMARY obligatorio |
| 2026-02-09 | **Piloto collab exitoso** | DEC-001 y DEC-002 cerradas; plan Gate 4 v2 consolidado |
| 2026-02-10 | **Gate 4 hardening pre-run** | Fix de device mismatch en evaluación y guardado de checkpoint antes de eval |
| 2026-02-10 | **Gate 4 Run A lanzado (30 epochs)** | Régimen 1000/846 con seed 42 para comparación causal A/B |
| 2026-02-10 | **DEC-003 cerrada** | Playbook collab v1 operativo (A-B-C-D + E opcional), métricas M1/M2/M3 y umbral de aplicación |
| 2026-02-10 | **Gobernanza de roles Claude/Codex** | Claude implementa/ejecuta; Codex mantiene documentación del repo |
| 2026-02-11 | **Diagnóstico post Gate 4.1 completado** | Gate 6 confirma asimetría por audio congelado; Gate 4.2 pre-red queda NO-GO |
| 2026-02-11 | **Plan post-diagnóstico v1.1 aprobado** | Bloque A (S0/A/B/C) definido con criterios de corte y protocolo anti-variable-fantasma |
| 2026-02-12 | **Gate 4.2 integrado al árbol documental** | Plan final consolidado en `06_GATE_4_2_RATIO_CENTRICO/plan_gate_4.2.md` y sincronización troncal |
| 2026-02-12 | **Run D completado (Bloque A)** | Mejor single-seed: `S=51.0%`; foundation provisional en `D(ep5)` |
| 2026-02-12 | **Run D-02 lanzado (Bloque A)** | Extensión full-unfreeze a 30 epocas en curso; lock final diferido a comparativa `C5 vs D5 vs D-02(best)` |
| 2026-02-12 | **Visualizaciones 3D publicadas** | Sitio activo en `https://altermundi.github.io/Phideus/` con reconocimiento a `bbycroft/llm-viz` |
| 2026-02-12 | **Corte D-02 ep18** | Nuevo best parcial `S=59.6%`, `hard_neg=91.0%`; lock final sigue pendiente al cierre de corrida |
| 2026-02-13 | **Run D-02 cerrado (30 ep)** | Best single-seed en `epoch25` (`S=61.8%`, empate S con `epoch26`) y cierre de Bloque A v1.1 |
| 2026-02-13 | **Foundation lock formal** | Checkpoint inmutable: `data/bias_control_medium/training_outputs/foundation_locked_e25.pt` |
| 2026-02-13 | **Explore foundation ejecutado** | 6 probes completados en `resultados_compartir/` con resumen en `explore_summary.json` |
| 2026-02-14 | **Bifurcación Gate 4.3/4.4 aprobada** | Separación explícita de paradigmas: rama MIDI temperada y rama Audio de armonía natural |
| 2026-02-14 | **Gate 4.2 cerrado (D4 8ep)** | `S_best=64.2%` (e7), `hard_neg_best=91.6%`; Gate 4.3 avanza en corrida 6 brazos (`D0`/`D4` cerrados, `A4` en curso) |
| 2026-02-15 | **Gate 4.3 amplía fase experimental** | Se incorporan `D4x`, `d4a4`, `d4a4cm` y Fase 5 (`A4r`, `D4r`, `A8`, `A9`) |
| 2026-02-16 | **Gate 4.3 CERRADO** | 13 brazos completados; mejor 5ep `d4a4=69.8%` |
| 2026-02-16 | **d4a4-scratch 30ep COMPLETO** | Nuevo record `S=83.6%` (e30), referencia eval-seed `84.1% +/- 2.3pp` sobre un checkpoint |
| 2026-02-16 | **Roadmap distribuido LOCAL+UNC operativo** | Protocolo de ramas `main/unc`, release foundation y ejecución Fase 5 en UNC |
| 2026-02-17 | **A4r-scratch en cola UNC** | Siguiente punto de decisión antes del arranque efectivo de Gate 4.4 |
| 2026-02-18 | **Gate 4.4 screening completado** | Tabla 5ep cerrada para 24 brazos (incluye MoE v2/v3/v4) con métricas comparables |
| 2026-02-19 | **Runs largos 30ep cerrados + nuevo bloque 60ep** | Cierre de `d4-a4r/t3-wt/moe-dual` y apertura de batch 60ep + `t3-wt` 50ep hold |
| 2026-02-22 | **Gate 4.5 formalizado + reorden de árbol BIAS_CONTROL** | Secuencia oficial `4.4 -> 4.5 -> 5A -> 5B`, con nuevas rutas `09/10/11` |
