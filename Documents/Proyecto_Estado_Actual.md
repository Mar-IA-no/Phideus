<div align="center">

# Proyecto Estado Actual
### Phideus v5.0

![Program](https://img.shields.io/badge/Program-Research_Active-0A7E3B?style=for-the-badge)
![Current Focus](https://img.shields.io/badge/Focus-Escalon_1--C-1F6FEB?style=for-the-badge)
![Bias Control](https://img.shields.io/badge/BIAS_CONTROL-Gate_4.1_%2B_Gate_6-F59E0B?style=for-the-badge)

</div>

> [!IMPORTANT]
> **Actualizado**: 2026-02-10  
> **Estado**: ✅ Escalón 1-A/B completado (Gate 3 cerrado) — 🟡 Escalón 1-C en curso (Gate 4.1 + Gate 6)

## Navegación rápida

- [Resumen Ejecutivo](#resumen-ejecutivo)
- [Pipeline de Gates BIAS_CONTROL](#pipeline-de-gates-bias_control)
- [Marco Rosetta (alineación operativa)](#marco-rosetta-alineación-operativa)
- [Archivos de Referencia](#archivos-de-referencia)
- [Métricas Clave del Proyecto](#métricas-clave-del-proyecto)

---

## Lectura por frentes

| Frente | Estado | Documento eje |
|--------|--------|---------------|
| Pre-análisis (hashing/ratios) | Pausado como línea principal | `Documents/ESCALON_1/Plan_implementacion.md` |
| BIAS_CONTROL (cross-modal) | Línea principal activa | `Documents/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md` |
| UOEMD revisionismo | Histórico NO-GO | `Documents/UOEMD/UOEMD_Revisionismo/ROADMAP.md` |

---

## Resumen Ejecutivo

### Estado de Hipótesis

| Hipótesis | Estado | Evidencia |
|-----------|--------|-----------|
| H1: Estructura de ratios | **VALIDADA** | Distribuciones no aleatorias |
| H2: Aprendibilidad | **VALIDADA** | VAE/HRM val_loss < 0.5 |
| H3: Cross-modality | 🟢 **PROMETEDOR** | BIAS_CONTROL: Gap 0.478, Hard neg acc 80.4% |

### Situación Actual (2026-02-10)

**BIAS_CONTROL Gate 3 (DANN) CERRADO** — 4 Runs completados, DANN no mejora:

- **Gate 2**: GO (Gap 0.478, Recall@10 34.4%, Hard neg acc 80.4%) — **MEJOR CHECKPOINT**
- **Gate 3**: ❌ CERRADO — 4 Runs de DANN, ninguno mejora sobre Gate 2
- **Gate 4 (base)**: ✅ completado (Run A 30 épocas, señal mixta)
- **Escalón 1-C**: Gate 4.1 en curso (matriz causal DEC-004) + Gate 6 pendiente

### Actualizaciones del día (2026-02-10)

- Integración operativa de Codex al repo y reglas persistentes en `CODEX.md`.
- Protocolo Claude↔Codex consolidado con `DEC-003`:
  - Playbook v1 para tareas de impacto (`A->B->C->D`, con `E` opcional de spot-check post-ejecución).
  - Métricas por ciclo: `M1` bloqueantes pre-ejecución, `M2` issues que habrían causado fallo, `M3` desacuerdo residual (objetivo `0`).
  - Estado actual: `COLLAB OFF`.
- Gobernanza de roles acordada:
  - Claude enfocado en implementación/ejecución.
  - Codex responsable de mantener actualizada la documentación del repositorio.
- Gate 4: se aplicaron ajustes de robustez para evitar pérdida de progreso:
  - fix de device mismatch en evaluación (`piece_idx`/`segment_idx` a CPU),
  - guardado de checkpoint antes de `evaluate()`.
- Gate 4 Run A finalizado (30 épocas) y evaluado en structured pool:
  - `RA5`: A2M R@10 31.4%, M2A R@10 40.6%, hard neg 79.0%
  - `RA30`: A2M R@10 29.2%, M2A R@10 36.4%, hard neg 74.8%
- Se cerró `DEC-004` (Gate 4.1):
  - Fase 0 bloqueante: `RB0` (`ratio_weight=0.0`, 5 épocas, régimen idéntico a `RA5`)
  - Continuidad condicionada por umbral causal (`S=min(R@10 a2m,m2a)` + hard neg)
- Se consolidó documentación de auditoría y encuadre Escalón 1-A/B/C para mantener decisiones consistentes.

#### Resultado Definitivo Gate 3 (Structured Pool, 7 checkpoints)

| Checkpoint | R@10 a2m | R@10 m2a | Hard Neg | MRR a2m |
|-----------|---------|---------|----------|---------|
| **gate2_ep45 (sin DANN)** | **34.4%** | 37.6% | **80.4%** | 0.138 |
| runA_best_ep7 (sin norm) | 27.8% | 35.4% | 74.8% | 0.132 |
| runB_ep5 (λ~0.17) | 24.6% | 32.0% | 70.4% | 0.112 |
| runB_ep10 (λ~0.33) | 29.8% | 34.6% | 73.6% | 0.130 |
| runC_best_ep4 (λ~0.3 trans.) | 34.6% | **39.2%** | **81.2%** | **0.148** |
| runC_ep13 (λ=0.8) | 32.2% | 38.0% | 76.6% | 0.144 |
| **runD_best_ep12 (λ=0.3 sost.)** | **27.4%** | 36.4% | 73.2% | 0.134 |

**Conclusión**: Gate 2 (sin DANN) es el mejor. La separabilidad modal (92.7%) no es el factor limitante del retrieval. DANN destruye información útil sin compensar.

---

## Pipeline de Gates BIAS_CONTROL

| Gate | Descripción | Estado | Resultado |
|------|-------------|--------|-----------|
| 0 | Data Integrity | ✅ Completado | GO |
| 1 | Intra-Modal Baselines | ✅ Completado | GO |
| **2** | **VICReg Training** | ✅ **Completado** | **GO — Mejor checkpoint** |
| 2.5 | Embedding Analysis | ✅ Completado | 92.7% separabilidad |
| **3** | **DANN Training** | ❌ **CERRADO** | **DANN no mejora (4 Runs)** |
| **4 (base)** | **Ratio Auxiliary** | ✅ **COMPLETADO** | Señal mixta; ep5 mejor que ep30 |
| **4.1 (DEC-004)** | **Matriz causal por fases** | ⏳ **EN CURSO** | Bloqueante: ejecutar `RB0` |
| 5 | Curriculum (opcional) | ⏸ HOLD | No bloquea cierre de escalón |
| 6 | Retroanálisis | ⏳ Prioridad post-Gate 4.1 | Embeddings vs representaciones |

### Próximos Pasos

1. ⏳ **Gate 4.1 Fase 0**: ejecutar `RB0` y comparar `RA5 vs RB0` bajo regla causal DEC-004.
2. ⏳ **Gate 4.1 Fase 1** (si GO): screening de variantes `R1-R4` a 5 épocas.
3. ⏳ **Gate 6**: ejecutar retroanálisis representacional (RSA/CKA/probes/disagreement).
4. ⏳ Cerrar auditoría final de BIAS_CONTROL al completar Gate 4.1 + Gate 6.

## Marco Rosetta (alineación operativa)

`BIAS_CONTROL` se ubica en el `Escalón 1` de `Documents/Rosetta_triplescaloneta.md`, con subfases:

- `Escalón 1-A`: Gates 0/1/2
- `Escalón 1-B`: Gate 3
- `Escalón 1-C`: Gate 4 base + Gate 4.1 + Gate 6

Criterio de cierre de Escalón 1:
- Gate 4.1 completo con evidencia causal.
- Gate 6 completo con evidencia explicativa.
- Auditoría final consolidada.

---

## 🟢 BIAS_CONTROL: Gate 2 — MEJOR CHECKPOINT ACTUAL

**Checkpoint**: `data/bias_control_medium/training_outputs/gate2/checkpoint_epoch45.pt`

| Dirección | R@1 | R@10 | MRR |
|-----------|-----|------|-----|
| Audio→MIDI | 4.4% | 34.4% | 0.138 |
| MIDI→Audio | 5.2% | 37.6% | 0.158 |

| Test | Accuracy |
|------|----------|
| vs Same-Piece-Diff-Time | **80.4%** |
| vs Random | 87.0% |

---

## 🟡 ESCALÓN 1: MAESTRO (Hashing) - PAUSADO

| Métrica | Valor | Status |
|---------|-------|--------|
| Piece Accuracy | 27% | ✗ Insuficiente |
| vs Random | 5.4× | ✓ Señal detectada |
| Causa raíz | Resolución temporal onset | Identificada |

---

## 🔴 REVISIONISMO UOEMD - COMPLETADO (NO-GO)

| Fase | Resultado |
|------|-----------|
| Fase 0: Tests sintéticos | ✓ GO |
| Fase 1: Extractor v2.2 | ✓ Gap 0.691 |
| Fase 2: Re-entrenamiento | ✗ Gap 0.007 |
| Fase 3A: Constellation tokens | ✗ Random level |

---

## Archivos de Referencia

### BIAS_CONTROL

| Archivo | Descripción |
|---------|-------------|
| `Documents/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md` | Arquitectura y gates (v2.0) |
| `Documents/BIAS_CONTROL/AUDITORIA_BIAS_CONTROL_CODEX.md` | Auditoría técnica v1 + addendum Gate 4.1 |
| `Documents/BIAS_CONTROL/Gate3_DANN_Results/INFORME_GATE3_COMPLETO.md` | **Informe exhaustivo Gate 3 (4 Runs)** |
| `Documents/BIAS_CONTROL/INFORME_GATE2_COMPLETO.md` | Informe exhaustivo Gate 2 |
| `experiments/bias_control/compare_gate3_checkpoints.py` | Comparación Gate 3 |
| `experiments/bias_control/evaluate_structured_pool.py` | Pool estructurado |

### Colaboración de agentes

| Archivo | Descripción |
|---------|-------------|
| `COLLAB/README.md` | Protocolo colaborativo Claude↔Codex |
| `COLLAB/DECISIONS.md` | Decisiones cerradas de protocolo + `DEC-004` (Gate 4.1) |
| `CODEX.md` | Reglas locales de Codex para este repo |

---

## Métricas Clave del Proyecto

```
┌────────────────────────────────────────────────────────────────┐
│        BIAS_CONTROL — ESTADO ACTUAL                            │
├────────────────────────────────────────────────────────────────┤
│  Gate 2 (MEJOR CHECKPOINT):                                    │
│    Gap: 0.478 | R@10 pool256: 34.4% | Hard neg: 80.4%        │
│                                                                 │
│  Gate 3 (DANN) CERRADO:                                        │
│    4 Runs, ninguno mejora sobre Gate 2                         │
│    Separabilidad modal ≠ factor limitante                      │
│                                                                 │
│  Escalón 1-C: Gate 4.1 en curso + Gate 6 pendiente             │
│                                                                 │
│  Gate 2: GO | Gate 3: CERRADO | Gate 4.1: EN CURSO | Gate 6: PEND│
└────────────────────────────────────────────────────────────────┘
```

---

*Documento actualizado: 2026-02-10 (Escalón 1-C en curso: Gate 4.1 + Gate 6)*
