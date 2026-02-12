<div align="center">

# Índice de Documentación
### Proyecto Phideus v5.0

![Scope](https://img.shields.io/badge/Scope-Project_Documentation-1F6FEB?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-0A7E3B?style=for-the-badge)
![Updated](https://img.shields.io/badge/Updated-2026--02--12-F59E0B?style=for-the-badge)

</div>

> [!IMPORTANT]
> **Propósito**: referencia rápida de documentación operativa y de investigación.

## Navegación rápida

- [Documentos Troncales (Tier A)](#documentos-troncales-tier-a)
- [Documentos Principales](#documentos-principales)
- [Escalón 1: MAESTRO (Audio ↔ MIDI)](#escalón-1-maestro-audio--midi)
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
| `Documents/00_TRONCAL/HANDOFF.md` | Continuidad operativa entre sesiones e instancias |
| `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md` | Plan maestro del experimento principal |

---

## Documentos Principales

| Documento | Ubicación | Descripción |
|-----------|-----------|-------------|
| **Estado Actual** | `Documents/00_TRONCAL/Proyecto_Estado_Actual.md` | Estado global del proyecto |
| **Este índice** | `Documents/00_TRONCAL/INDICE_DOCUMENTACION.md` | Mapa de documentación |
| **Handoff operativo** | `Documents/00_TRONCAL/HANDOFF.md` | Estado breve verificable + próximo paso único |
| **CLAUDE.md** | `CLAUDE.md` | Instrucciones para Claude Code |
| **CODEX.md** | `CODEX.md` | Reglas operativas de Codex (collab/contexto/hardware/documentación) |
| **Bitácora** | `Documents/00_TRONCAL/bitacora_desarrollo.md` | Log de desarrollo |
| **Paper** | `Documents/03_FRENTES_CERRADOS/UOEMD/UOEMD_Roseta_v2.2/PHIDEUS_RESEARCH_PROGRAM_2026.md` | Paper técnico de referencia |
| **★ Informe Histórico** | `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/INFORME_HISTORICO_REPRESENTACIONES_RATIOS.md` | **NUEVO**: Historia completa de representaciones de ratios |
| **Backpropagando Phideus** | `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/BACKPROPAGANDO_PHIDEUS.md` | Ideas y redefiniciones en discusión (no implementadas) |

---

## Escalón 1: MAESTRO (Audio ↔ MIDI)

### Estado: 🟡 Escalón 1-C en curso (diagnóstico post Gate 4.1 completado + Bloque A v1.1 con `Run D-02` en curso y foundation lock en cierre)

### Documentación

| Documento | Ubicación | Contenido |
|-----------|-----------|-----------|
| Plan de implementación | `Documents/01_FRENTES_ACTIVOS/ESCALON_1/Plan_implementacion.md` | 6 Gates del experimento |
| Plan de validación H3 | `Documents/01_FRENTES_ACTIVOS/ESCALON_1/PLAN_VALIDACION_H3.md` | 4 fases: Auditoría, Replicación, Escala, Pipeline |
| Recomendaciones GPT | `Documents/01_FRENTES_ACTIVOS/ESCALON_1/Extractor_nuevos_enfoques_GPT5.2Think.md` | Specs de Route A y Route B |
| Resultados preliminares | `Documents/01_FRENTES_ACTIVOS/ESCALON_1/RESULTADOS_NUEVOS_ENFOQUES.md` | Resultados N=10 (preliminares) |
| **Auditoría Fase A** | `Documents/01_FRENTES_ACTIVOS/ESCALON_1/AUDITORIA_FASE_A.md` | Bug t_anchor encontrado |
| **Informe Fases A-B** | `Documents/01_FRENTES_ACTIVOS/ESCALON_1/INFORME_FASES_A_B.md` | Resultados corregidos + replicación |
| **Plan análisis errores** | `Documents/01_FRENTES_ACTIVOS/ESCALON_1/PLAN_ANALISIS_ERRORES.md` | 5 fases de análisis |
| **Informe análisis errores** | `Documents/01_FRENTES_ACTIVOS/ESCALON_1/INFORME_ANALISIS_ERRORES.md` | **Diagnóstico completo, causa raíz** |

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
| N=10 original (con bug) | 71-80% accuracy | `RESULTADOS_NUEVOS_ENFOQUES.md` |
| N=10 corregido | 32-42% accuracy | `AUDITORIA_FASE_A.md` |
| N=20 replicación | 21-27% accuracy | `INFORME_FASES_A_B.md` |
| N=20 post-mejoras | 27% accuracy, 5.4x random | `INFORME_ANALISIS_ERRORES.md` |

---

## BIAS_CONTROL: Cross-Modal Learning con Control de Sesgo

### Estado: ✅ **Escalón 1-A/B completado** — Gate 3 cerrado. 🟡 **Escalón 1-C en curso** (post-diagnóstico, Bloque A v1.1 con S0/A/B/C/D cerrados + Gate 4.2 ratio-céntrico listo para screening post-lock)

Marco de referencia:
- `Documents/00_TRONCAL/ROADMAP_GENERAL/Rosetta_triplescaloneta.md`
- `Documents/00_TRONCAL/ROADMAP_GENERAL/PLAN_AVANCE_TRIPLESCALONETA_v1.1.md` (vigente)
- `Documents/00_TRONCAL/ROADMAP_GENERAL/PLAN_AVANCE_TRIPLESCALONETA_v1.md` (archivado historico)
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md`

### Documentación

| Documento | Ubicación | Contenido |
|-----------|-----------|-----------|
| **Roadmap** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md` | Plan completo y criterios GO/NO-GO (v2.1) |
| **Índice por fases (nuevo)** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/INDEX_BIAS_CONTROL.md` | Navegación del roadmap por árbol de directorios |
| **Auditoría Codex (v1 + addendums)** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/90_ARCHIVO_REFERENCIA/AUDITORIA_BIAS_CONTROL_CODEX.md` | Auditoría histórica + addendums operativos |
| **Plan post-diagnóstico v1.1** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/05_PLAN_POST_DIAGNOSTICO_BLOQUE_A/PLAN_EJECUCION_POST_DEC005_v1.1.md` | Plan operativo vigente (S0/A/B/C) |
| **Plan Gate 4.2 (final)** | `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/06_GATE_4_2_RATIO_CENTRICO/PLANES/plan_gate_4.2.md` | Exploración ratio-céntrica post Bloque A (v2.1) |
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

### Protocolo Claude + Codex

Estado actual: `COLLAB OFF` con protocolo consolidado y `TURN_SUMMARY`.

| Documento | Ubicación | Contenido |
|-----------|-----------|-----------|
| Protocolo collab | `COLLAB/README.md` | Reglas ON/OFF, task-claim, TURN_SUMMARY |
| Snapshot collab | `COLLAB/STATUS.md` | Estado por agente y modo activo |
| Decisiones collab | `COLLAB/DECISIONS.md` | DEC-001..DEC-005 (protocolo, Gate 4.x y diagnóstico actual) |

Gobernanza operativa vigente:
- Claude: implementación y ejecución experimental.
- Codex: mantenimiento y actualización de documentación del repositorio.

### Skill documental (Codex)

| Artefacto | Ubicación | Rol |
|-----------|-----------|-----|
| Skill `phideus-doc-maintainer` | `tools/skills/phideus-doc-maintainer/SKILL.md` | Actualización documental dinámica por frente |
| Detección de frente | `tools/skills/phideus-doc-maintainer/scripts/detect_front.py` | `auto + override` con evidencia |
| Selección de targets | `tools/skills/phideus-doc-maintainer/scripts/select_targets.py` | Política "frente + global mínima" |
| Verificación de consistencia | `tools/skills/phideus-doc-maintainer/scripts/consistency_check.py` | Validaciones de políticas locales |

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
/mnt/m2-1TB/Phideus/
├── Documents/
│   ├── 00_TRONCAL/                  # Índice, estado, bitácora, roadmap general
│   ├── 01_FRENTES_ACTIVOS/          # BIAS_CONTROL y Escalón 1
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
cd /mnt/m2-1TB/Phideus
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
| 2026-02-12 | **Gate 4.2 integrado al árbol documental** | Plan final consolidado en `06_GATE_4_2_RATIO_CENTRICO/PLANES/plan_gate_4.2.md` y sincronización troncal |
| 2026-02-12 | **Run D completado (Bloque A)** | Mejor single-seed: `S=51.0%`; foundation provisional en `D(ep5)` |
| 2026-02-12 | **Run D-02 lanzado (Bloque A)** | Extensión full-unfreeze a 30 epocas en curso; lock final diferido a comparativa `C5 vs D5 vs D-02(best)` |
