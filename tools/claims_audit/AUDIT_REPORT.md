# Phideus Numeric Claims Audit Report
Generated: 2026-04-03T14:04:27.912053

## Executive Summary
- **Total claims**: 55
- **PASS**: 53 (96%)
- **FAIL**: 0 (0%)
- **WARN**: 1 (2%)
- **MISSING**: 0 (0%)
- **DOC_ONLY**: 0 (0%)
- **BITACORA_ONLY**: 0 (0%)
- **STALE**: 1 (2%)

## Tier 1: Docs Canónicos Públicos

| Status | Claim ID | Documented | Extracted | Delta | Source | Notes |
|--------|----------|-----------|-----------|-------|--------|-------|
| ✅ PASS | T1_G5B_MULTI_D0_S_MEAN | 75.2 | 75.2 | 0.0 | final_results.json... |  |
| ✅ PASS | T1_G5B_MULTI_A4R_S_MEAN | 80.7 | 80.72 | 0.02 | final_results.json... |  |
| ✅ PASS | T1_G5B_MULTI_D4A4R_S_MEAN | 81.2 | 81.16 | 0.04 | final_results.json... |  |
| ✅ PASS | T1_G8_PCD_S | 84.2 | 84.2 | 0.0 | final_results.json |  |
| ✅ PASS | T1_G8_PCA_S | 82.6 | 82.6 | 0.0 | final_results.json |  |
| ✅ PASS | T1_G8_PCM_S | 80.0 | 80.0 | 0.0 | final_results.json |  |
| ✅ PASS | T1_G8_CTRL_S | 79.2 | 79.2 | 0.0 | final_results.json |  |
| ✅ PASS | T1_G8_PCDZERO_S | 81.8 | 81.8 | 0.0 | final_results.json |  |
| ✅ PASS | T1_G10_a7_concat_S | 76.4 | 76.4 | 0.0 | final_results.json |  |
| ✅ PASS | T1_G10_a10a_concat_S | 75.6 | 75.6 | 0.0 | final_results.json |  |
| ✅ PASS | T1_G10_a10d_concat_S | 75.4 | 75.4 | 0.0 | final_results.json |  |
| ✅ PASS | T1_G10_a7_FiLM_S | 71.6 | 71.6 | 0.0 | final_results.json |  |
| ✅ PASS | T1_G10_a10a_FiLM_S | 74.0 | 74.0 | 0.0 | final_results.json |  |
| ✅ PASS | T1_G10_a10d_FiLM_S | 73.2 | 73.2 | 0.0 | final_results.json |  |
| ✅ PASS | T1_G10_a7_attn_bias_S | 55.8 | 55.8 | 0.0 | final_results.json |  |
| ✅ PASS | T1_G10_a10a_attn_bias_S | 59.6 | 59.6 | 0.0 | final_results.json |  |
| ✅ PASS | T1_G10_a10d_attn_bias_S | 57.4 | 57.4 | 0.0 | final_results.json |  |
| ✅ PASS | T1_G7_MERT_R2 | 0.85 | 0.8498 | 0.0002 | probe_results.json |  |
| ✅ PASS | T1_G71_FROZEN_S | 75.0 | 75.0 | 0.0 | d0_mert330m_seed42 |  |
| ✅ PASS | T1_G9_a7r_S | 70.4 | 70.4 | 0.0 | final_results.json |  |
| ✅ PASS | T1_G9_a9r_S | 71.6 | 71.6 | 0.0 | final_results.json |  |
| ✅ PASS | T1_G9_a10ar_S | 70.6 | 70.6 | 0.0 | final_results.json |  |
| ✅ PASS | T1_G9_a10br_S | 70.0 | 70.0 | 0.0 | final_results.json |  |
| ✅ PASS | T1_G9_a10cr_S | 69.2 | 69.2 | 0.0 | final_results.json |  |
| ✅ PASS | T1_G9_a10dr_S | 70.2 | 70.2 | 0.0 | final_results.json |  |
| ✅ PASS | T1_G9_a10er_S | 71.8 | 71.8 | 0.0 | final_results.json |  |
| ✅ PASS | T1_E2_D0_S | 77.8 | 77.8 | 0.0 | lombard |  |
| ✅ PASS | T1_G6_EXPA_BASELINE_F1 | 0.3186 | 0.3186 | 0.0 | baseline_results.json |  |
| ✅ PASS | T1_G6_EXPA_A4EVENT_F1 | 0.3186 | 0.3186 | 0.0 | training_results.json |  |
| 🕐 STALE | T1_E2_P3_STATE | P3-D0 corriendo / sin lectura | — | — | p3_full_results.json | Doc says 'P3-D0 corriendo / sin lectura' but results file ex |
| ⚠️ WARN | T1_G5B_MULTI_D4A4_S_MEAN | 84.1 | 84.08 | 0.02 | eval_seed123.json... | WARN: Recovered artifacts are eval-seed (5 evals on 1 checkp |

## Tier 2: Informes Autoritativos

| Status | Claim ID | Documented | Extracted | Delta | Source | Notes |
|--------|----------|-----------|-----------|-------|--------|-------|
| ✅ PASS | T2_G5B_T12_d4a4_S | 83.8 | 83.8 | 0.0 | test12_scoreboard.json |  |
| ✅ PASS | T2_G5B_T12_D0_S | 73.4 | 73.4 | 0.0 | test12_scoreboard.json |  |
| ✅ PASS | T2_G5B_T12_a4r_S | 82.0 | 82.0 | 0.0 | test12_scoreboard.json |  |
| ✅ PASS | T2_G5B_T12_d4-a4r_S | 79.8 | 79.8 | 0.0 | test12_scoreboard.json |  |
| ✅ PASS | T2_G5B_T01_d4a4_ZERO_AUDIO | 7.8 | 7.8 | 0.0 | test01_causal_ablation.json |  |
| ✅ PASS | T2_G5B_T01_a4r_ZERO_AUDIO | 4.4 | 4.4 | 0.0 | test01_causal_ablation.json |  |
| ✅ PASS | T2_G5B_T01_d4-a4r_ZERO_AUDIO | 4.4 | 4.4 | 0.0 | test01_causal_ablation.json |  |
| ✅ PASS | T2_G5B_T02_real_S | 83.0 | 83.0 | 0.0 | final_results.json |  |
| ✅ PASS | T2_G5B_T02_random_S | 73.6 | 73.6 | 0.0 | final_results.json |  |
| ✅ PASS | T2_G5B_T02_zero_S | 75.0 | 75.0 | 0.0 | final_results.json |  |
| ✅ PASS | T2_G5B_T02_SHUFFLED_S | 73.6 | 73.6 | 0.0 | final_results.json | Raw JSON gives 73.2; docs use 73.6* (early convergence opera |
| ✅ PASS | T2_G5B_T06_D0_CKA | 0.435 | 0.4353 | 0.0003 | test06_rsa_cka.json |  |
| ✅ PASS | T2_G5B_T06_d4a4_CKA | 0.659 | 0.6594 | 0.0004 | test06_rsa_cka.json |  |
| ✅ PASS | T2_G5B_T06_a4r_CKA | 0.766 | 0.7662 | 0.0002 | test06_rsa_cka.json |  |
| ✅ PASS | T2_G5B_T06_d4-a4r_CKA | 0.794 | 0.7939 | 0.0001 | test06_rsa_cka.json |  |
| ✅ PASS | T2_G5B_T04_D0_TRANS | 36.8 | 36.7847 | 0.0153 | test04_transposition.json |  |
| ✅ PASS | T2_G5B_T04_d4a4_TRANS | 51.3 | 51.3126 | 0.0126 | test04_transposition.json |  |
| ✅ PASS | T2_G5B_T04_a4r_TRANS | 59.3 | 59.2683 | 0.0317 | test04_transposition.json |  |
| ✅ PASS | T2_G5B_T04_d4-a4r_TRANS | 59.0 | 59.0226 | 0.0226 | test04_transposition.json |  |
| ✅ PASS | T2_G5B_T11_D0_IR | 0.597 | 0.5971 | 0.0001 | test11_preproj_ab.json | May be in test11_preproj_ab_summary.json or test11_preproj_a |
| ✅ PASS | T2_G5B_T11_a4r_IR | 0.712 | 0.7122 | 0.0002 | test11_preproj_ab.json | May be in test11_preproj_ab_summary.json or test11_preproj_a |
| ✅ PASS | T2_E3_P2_flat_IID_S | 0.583 | 0.5833 | 0.0003 | final_results.json | Check both final_results.json and final_results_v2.json. Ali |
| ✅ PASS | T2_E3_P2_cqtshift_IID_S | 0.515 | 0.5146 | 0.0004 | final_results.json | Check both final_results.json and final_results_v2.json. Ali |
| ✅ PASS | T2_E3_P5_CQT_SCALE_OOD | 0.508 | 0.508 | 0.0 | final_results.json |  |

## Estado Documental Desactualizado (STALE)

- **T1_E2_P3_STATE**: Doc says 'P3-D0 corriendo / sin lectura' but results file exists

---

## Correcciones Documentales Requeridas (para Codex)

### CORRECCIÓN 1 — OBLIGATORIA: Gate 8 pca epoch incorrecto

**Qué dice**: pca=82.6% @e30
**Qué debería decir**: pca=82.6% @e25
**Fuente de verdad**: `Documents/BITACORA_UNC.md:983,1013,1031` (fija @e25)
**Archivo de respaldo**: `results_unc/gate8_conditioned_projections/a4r-pca_seed42/final_results.json`

Archivos a corregir:
- `Documents/00_TRONCAL/Proyecto_Estado_Actual.md:17` — cambiar @e30 → @e25 para pca

### CORRECCIÓN 2 — OBLIGATORIA: Escalón 2 P3 stale-state

**Qué dicen**: P3-D0 "corriendo" o "sin lectura de resultados"
**Qué debería decir**: P3 completado, resultados disponibles
**Fuente de verdad**: `data/lombard/p3_interpretation/p3_full_results.json`

Archivos a corregir:
- `Documents/01_FRENTES_ACTIVOS/ESCALON_2/README.md:14` — actualizar estado P3
- `Documents/01_FRENTES_ACTIVOS/ESCALON_2/README.md:250` — actualizar estado P3
- `Documents/01_FRENTES_ACTIVOS/ESCALON_2/ROADMAP_ESCALON_2.md:4,7` — actualizar estado P3
- `Documents/00_TRONCAL/Proyecto_Estado_Actual.md:18` — actualizar estado Escalón 2

### CORRECCIÓN 3 — ALTA: Test02 shuffled asterisco

**Qué dice en algunos docs**: shuffled=73.6% (sin asterisco)
**Qué debería decir siempre**: shuffled=73.6%* (con asterisco de convergencia operativa)
**Fuente de verdad**: `results_unc/gate5b_param_matched/shuffled/final_results.json` → `training.best_S = 0.732` (73.2% raw)
**Contexto**: El JSON crudo da 73.2%. La documentación usa 73.6%* como valor operativo por convergencia temprana. Toda cita sin asterisco es incorrecta.

Archivos a auditar (verificar presencia del *):
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md:752`
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/INFORME_COMPLETO_GATE5B.md` (buscar "shuffled" y verificar *)
- `Documents/00_TRONCAL/Proyecto_Estado_Actual.md` (buscar "shuffled")
- `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/PHIDEUS_MASTER_BRIEFING.md` (buscar "shuffled")

### ADVERTENCIA 4 — ALTA: d4a4 multi-seed semántica

**Qué dicen los docs**: d4a4=84.1%±2.3pp (5 seeds) — implica 5 trainings independientes
**Qué respaldan los artefactos**: 5 evaluaciones (eval_seed*.json) sobre un único checkpoint, con seeds 42,123,456,789,2026 — NO 5 trainings

**Discrepancias específicas**:

| | Docs (INFORME_GATE5B) | Artefactos recuperados |
|---|---|---|
| Seeds | 42, 123, 456, 789, **1337** | 42, 123, 456, 789, **2026** |
| Valores | 83.6, 86.4, 84.0, 82.0, 84.4 | 83.6, 88.4, 83.0, 82.6, 82.8 |
| Tipo | Multi-seed training (implícito) | Multi-seed evaluation (1 checkpoint) |

**Archivos que citan d4a4 multi-seed y necesitan revisión**:
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/11_GATE_5_LINEA_B_SHOWCASE/INFORME_COMPLETO_GATE5B.md:199,203` — seeds y valores individuales
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/RANKING_DESCRIPTORES_UNIFICADO.md:457,459` — seeds y valores
- `Documents/00_TRONCAL/Proyecto_Estado_Actual.md:19` — claim 84.1%±2.3pp
- `Documents/00_TRONCAL/INDICE_DOCUMENTACION.md:520` — claim multi-seed
- `Documents/00_TRONCAL/HANDOFF.md:315,869` — claim multi-seed
- `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/PHIDEUS_MASTER_BRIEFING.md:227` — claim multi-seed
- `Documents/NOTAS_CLAUDE-CODEX.md:12` — seeds canónicas

**Opciones de resolución**:
1. Corregir la narrativa: explicitar que d4a4 es eval-seed (varianza del evaluador), no training-seed (varianza del training), y que los otros 3 arms sí son training-seed. Esto es honesto pero debilita el claim.
2. Correr d4a4 training multi-seed real (5 trainings independientes con seeds 42,123,456,789,1337). Esto cierra el agujero pero requiere ~5×36h de GPU.
3. Mantener como está con nota metodológica explícita en el informe Gate 5B.

**Decisión requerida del usuario.**

### LIMPIEZA 5 — BAJA: Gate 6 documentación incompleta

Los raw artifacts de Gate 6 Exp A/B ahora están en `results_unc/gate6_amt/`. Los docs del Gate 6 podrían actualizarse para referenciar estos archivos explícitamente.

Archivo a actualizar:
- `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/12_GATE_6_AMT/README.md` — agregar referencia a `results_unc/gate6_amt/expA/` y `expB/`
