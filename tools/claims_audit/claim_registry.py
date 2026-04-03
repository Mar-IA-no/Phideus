"""
Phideus Numeric Claims Registry.
Every auditable claim as a structured dataclass.
"""
from dataclasses import dataclass, field
from typing import Optional
import os

BASE = "/mnt/m2-1TB/Phideus"
RESULTS_UNC = f"{BASE}/results_unc"
DATA = f"{BASE}/data"
DOCS = f"{BASE}/Documents"
TRONCAL = f"{DOCS}/00_TRONCAL"
BC = f"{DOCS}/01_FRENTES_ACTIVOS/BIAS_CONTROL"


@dataclass
class Claim:
    id: str
    tier: int
    source_doc: str
    source_location: str
    claim_text: str
    expected_value: float
    tolerance: float
    category: str  # retrieval, causal, geometric, mechanistic, downstream
    measurement_semantics: str  # best, final, operational_best, mean_sd, derived
    evaluation_protocol: str
    alias_of: str  # e.g. "p2_baseline_seed42" for "P2-flat"
    claim_kind: str  # numeric, state, protocol
    evidence_level: str  # raw_json, raw_log, summary_json, bitacora_only, docs_bitacora, doc_only
    data_source_path: str
    extraction_spec: str  # JSON key path or function name
    notes: str = ""


def build_registry() -> list[Claim]:
    """Build the complete claims registry."""
    claims = []

    # =========================================================================
    # TIER 1: Docs canónicos públicos
    # =========================================================================

    pea = f"{TRONCAL}/Proyecto_Estado_Actual.md"
    readme = f"{BASE}/README.md"
    roadmap = f"{BC}/ROADMAP_BIAS_CONTROL.md"

    # --- Gate 5B multi-seed (from Proyecto_Estado_Actual / README) ---
    claims.append(Claim(
        id="T1_G5B_MULTI_D4A4_S_MEAN",
        tier=1, source_doc=pea, source_location="line ~19",
        claim_text="d4a4=84.1%±2.3pp (5 seeds)",
        expected_value=84.1, tolerance=0.2,
        category="retrieval", measurement_semantics="mean_sd",
        evaluation_protocol="pool=256,queries=500,seed=varied,structured",
        alias_of="gate5b_multiseed_d4a4",
        claim_kind="numeric", evidence_level="summary_json",
        data_source_path=f"{DATA}/gate5b_results/d4a4/multiseed/eval_seed*.json",
        extraction_spec="multiseed_mean:gate_metrics.S",
        notes="WARN: Recovered artifacts are eval-seed (5 evals on 1 checkpoint), "
              "NOT training-seed (5 independent runs) like D0/a4r/d4-a4r. "
              "Seeds differ: docs say 42,123,456,789,1337; files have 42,123,456,789,2026. "
              "Individual values differ: docs 83.6,86.4,84.0,82.0,84.4 vs files 83.6,88.4,83.0,82.6,82.8. "
              "Mean coincidence (84.1 vs 84.08) is likely accidental. "
              "True multi-seed training for d4a4 may never have been run."
    ))

    claims.append(Claim(
        id="T1_G5B_MULTI_D0_S_MEAN",
        tier=1, source_doc=pea, source_location="line ~19",
        claim_text="D0 baseline: 75.2%±2.3pp",
        expected_value=75.2, tolerance=0.2,
        category="retrieval", measurement_semantics="mean_sd",
        evaluation_protocol="pool=256,queries=500,seed=varied,structured",
        alias_of="gate5b_multiseed_D0",
        claim_kind="numeric", evidence_level="raw_json",
        data_source_path=f"{RESULTS_UNC}/gate5b_multiseed/D0_seed*/final_results.json",
        extraction_spec="multiseed_mean:training.best_S",
    ))

    claims.append(Claim(
        id="T1_G5B_MULTI_A4R_S_MEAN",
        tier=1, source_doc=pea, source_location="line ~19",
        claim_text="a4r: 80.7%±1.9pp",
        expected_value=80.7, tolerance=0.2,
        category="retrieval", measurement_semantics="mean_sd",
        evaluation_protocol="pool=256,queries=500,seed=varied,structured",
        alias_of="gate5b_multiseed_a4r",
        claim_kind="numeric", evidence_level="raw_json",
        data_source_path=f"{RESULTS_UNC}/gate5b_multiseed/a4r_seed*/final_results.json",
        extraction_spec="multiseed_mean:training.best_S",
    ))

    claims.append(Claim(
        id="T1_G5B_MULTI_D4A4R_S_MEAN",
        tier=1, source_doc=pea, source_location="line ~19",
        claim_text="d4-a4r: 81.2%±2.5pp",
        expected_value=81.2, tolerance=0.2,
        category="retrieval", measurement_semantics="mean_sd",
        evaluation_protocol="pool=256,queries=500,seed=varied,structured",
        alias_of="gate5b_multiseed_d4-a4r",
        claim_kind="numeric", evidence_level="raw_json",
        data_source_path=f"{RESULTS_UNC}/gate5b_multiseed/d4-a4r_seed*/final_results.json",
        extraction_spec="multiseed_mean:training.best_S",
    ))

    # --- Gate 8 conditioned projections ---
    claims.append(Claim(
        id="T1_G8_PCD_S",
        tier=1, source_doc=pea, source_location="Gate 8 section",
        claim_text="pcd=84.2% (record)",
        expected_value=84.2, tolerance=0.2,
        category="retrieval", measurement_semantics="best",
        evaluation_protocol="pool=256,queries=500,seed=42,structured",
        alias_of="gate8_pcd",
        claim_kind="numeric", evidence_level="raw_json",
        data_source_path=f"{DATA}/gate8_pcd_unc/final_results.json",
        extraction_spec="json_field:training.best_S",
    ))

    claims.append(Claim(
        id="T1_G8_PCA_S",
        tier=1, source_doc=pea, source_location="Gate 8 section",
        claim_text="pca=82.6%",
        expected_value=82.6, tolerance=0.2,
        category="retrieval", measurement_semantics="best",
        evaluation_protocol="pool=256,queries=500,seed=42,structured",
        alias_of="gate8_pca",
        claim_kind="numeric", evidence_level="raw_json",
        data_source_path=f"{RESULTS_UNC}/gate8_conditioned_projections/a4r-pca_seed42/final_results.json",
        extraction_spec="json_field:training.best_S",
    ))

    claims.append(Claim(
        id="T1_G8_PCM_S",
        tier=1, source_doc=pea, source_location="Gate 8 section",
        claim_text="pcm=80.0%",
        expected_value=80.0, tolerance=0.2,
        category="retrieval", measurement_semantics="best",
        evaluation_protocol="pool=256,queries=500,seed=42,structured",
        alias_of="gate8_pcm",
        claim_kind="numeric", evidence_level="raw_json",
        data_source_path=f"{DATA}/gate8_results/a4r-pcm_seed42/final_results.json",
        extraction_spec="json_field:training.best_S",
    ))

    claims.append(Claim(
        id="T1_G8_CTRL_S",
        tier=1, source_doc=pea, source_location="Gate 8 section",
        claim_text="ctrl=79.2%",
        expected_value=79.2, tolerance=0.2,
        category="retrieval", measurement_semantics="best",
        evaluation_protocol="pool=256,queries=500,seed=42,structured",
        alias_of="gate8_ctrl",
        claim_kind="numeric", evidence_level="raw_json",
        data_source_path=f"{DATA}/gate8_results/a4r-ctrl_seed42/final_results.json",
        extraction_spec="json_field:training.best_S",
    ))

    claims.append(Claim(
        id="T1_G8_PCDZERO_S",
        tier=1, source_doc=pea, source_location="Gate 8 section",
        claim_text="pcd-zero=81.8%",
        expected_value=81.8, tolerance=0.2,
        category="retrieval", measurement_semantics="best",
        evaluation_protocol="pool=256,queries=500,seed=42,structured",
        alias_of="gate8_pcd-zero",
        claim_kind="numeric", evidence_level="raw_json",
        data_source_path=f"{RESULTS_UNC}/gate8_conditioned_projections/a4r-pcd-zero_seed42/final_results.json",
        extraction_spec="json_field:training.best_S",
    ))

    # --- Gate 10 mechanism sweep (now synced from UNC) ---
    g10_base = f"{RESULTS_UNC}/gate10_mechanism_sweep"
    g10_arms = [
        ("a7_seed42", "a7-concat", 76.4),
        ("a10a_seed42", "a10a-concat", 75.6),
        ("a10d_seed42", "a10d-concat", 75.4),
        ("a7-pca_seed42", "a7-FiLM", 71.6),
        ("a10a-pca_seed42", "a10a-FiLM", 74.0),
        ("a10d-pca_seed42", "a10d-FiLM", 73.2),
        ("a7-ab_seed42", "a7-attn_bias", 55.8),
        ("a10a-ab_seed42", "a10a-attn_bias", 59.6),
        ("a10d-ab_seed42", "a10d-attn_bias", 57.4),
    ]
    for arm_dir, arm_name, expected_s in g10_arms:
        claims.append(Claim(
            id=f"T1_G10_{arm_name.replace('-','_')}_S",
            tier=1, source_doc=f"{BC}/17_GATE_10_MECHANISM_SWEEP/README.md",
            source_location="Gate 10 results table",
            claim_text=f"Gate10 {arm_name}={expected_s}%",
            expected_value=expected_s, tolerance=0.2,
            category="mechanistic", measurement_semantics="best",
            evaluation_protocol="pool=256,queries=500,seed=42,structured",
            alias_of=f"gate10_{arm_dir}",
            claim_kind="numeric", evidence_level="raw_json",
            data_source_path=f"{g10_base}/{arm_dir}/final_results.json",
            extraction_spec="json_field:training.best_S",
        ))

    # --- Gate 7 ---
    claims.append(Claim(
        id="T1_G7_MERT_R2",
        tier=1, source_doc=pea, source_location="Gate 7 section",
        claim_text="MERT R²=0.850",
        expected_value=0.850, tolerance=0.001,
        category="mechanistic", measurement_semantics="best",
        evaluation_protocol="segment_level",
        alias_of="gate7_mert330m",
        claim_kind="numeric", evidence_level="raw_json",
        data_source_path=f"{DATA}/gate7_results/probe_results/probe_results.json",
        extraction_spec="json_field:segment_level.MERT-v1-330M.r2_global_mean",
    ))

    claims.append(Claim(
        id="T1_G71_FROZEN_S",
        tier=1, source_doc=pea, source_location="Gate 7.1 section",
        claim_text="frozen S=75.0%",
        expected_value=75.0, tolerance=0.2,
        category="retrieval", measurement_semantics="best",
        evaluation_protocol="pool=256,queries=500,seed=42",
        alias_of="gate71_d0_mert330m",
        claim_kind="numeric", evidence_level="raw_json",
        data_source_path=f"{DATA}/gate71_results/d0_mert330m_seed42",
        extraction_spec="best_from_eval_epochs",
    ))

    # --- Gate 9/A10 (recovered from backup) ---
    g9_base = f"{DATA}/gate9_results"
    g9_arms = [
        ("a7r_seed42", "a7r", 70.4),
        ("a9r_seed42", "a9r", 71.6),
        ("a10ar_seed42", "a10ar", 70.6),
        ("a10br_seed42", "a10br", 70.0),
        ("a10cr_seed42", "a10cr", 69.2),
        ("a10dr_seed42", "a10dr", 70.2),
        ("a10er_seed42", "a10er", 71.8),
    ]
    for arm_dir, arm_name, expected_s in g9_arms:
        claims.append(Claim(
            id=f"T1_G9_{arm_name}_S",
            tier=1, source_doc=pea, source_location="Gate 9 section",
            claim_text=f"Gate9 {arm_name}={expected_s}%",
            expected_value=expected_s, tolerance=0.2,
            category="retrieval", measurement_semantics="best",
            evaluation_protocol="pool=256,queries=500,seed=42,structured",
            alias_of=f"gate9_{arm_dir}",
            claim_kind="numeric", evidence_level="raw_json",
            data_source_path=f"{g9_base}/{arm_dir}/final_results.json",
            extraction_spec="json_field:training.best_S",
        ))

    # --- Escalón 2 ---
    claims.append(Claim(
        id="T1_E2_D0_S",
        tier=1, source_doc=pea, source_location="Escalon 2 section",
        claim_text="D0 baseline: S=77.8%",
        expected_value=77.8, tolerance=0.2,
        category="retrieval", measurement_semantics="best",
        evaluation_protocol="lombard,structured",
        alias_of="e2_d0_control",
        claim_kind="numeric", evidence_level="raw_json",
        data_source_path=f"{DATA}/lombard",
        extraction_spec="lombard_d0_best_s",
    ))

    # --- Escalón 2 P3 stale state ---
    claims.append(Claim(
        id="T1_E2_P3_STATE",
        tier=1, source_doc=pea, source_location="Escalon 2 section",
        claim_text="P3-D0 corriendo / sin lectura",
        expected_value=0, tolerance=0,
        category="retrieval", measurement_semantics="best",
        evaluation_protocol="",
        alias_of="e2_p3_state",
        claim_kind="state", evidence_level="raw_json",
        data_source_path=f"{DATA}/lombard/p3_interpretation/p3_full_results.json",
        extraction_spec="file_exists",
        notes="STALE: docs say running but results already exist"
    ))

    # --- Gate 6 (now synced from UNC) ---
    claims.append(Claim(
        id="T1_G6_EXPA_BASELINE_F1",
        tier=1, source_doc=f"{BC}/12_GATE_6_AMT/README.md",
        source_location="Exp A results",
        claim_text="Gate6 ExpA baseline F1=0.3186",
        expected_value=0.3186, tolerance=0.001,
        category="downstream", measurement_semantics="final",
        evaluation_protocol="transkun,note_offset",
        alias_of="gate6_expA_baseline",
        claim_kind="numeric", evidence_level="raw_json",
        data_source_path=f"{RESULTS_UNC}/gate6_amt/expA/baseline_seed42/baseline_results.json",
        extraction_spec="json_field:results.note_offset_F1",
    ))
    claims.append(Claim(
        id="T1_G6_EXPA_A4EVENT_F1",
        tier=1, source_doc=f"{BC}/12_GATE_6_AMT/README.md",
        source_location="Exp A results",
        claim_text="Gate6 ExpA A4-event F1=0.3186 (= baseline, NEGATIVE)",
        expected_value=0.3186, tolerance=0.001,
        category="downstream", measurement_semantics="final",
        evaluation_protocol="transkun,note_offset",
        alias_of="gate6_expA_a4event",
        claim_kind="numeric", evidence_level="raw_json",
        data_source_path=f"{RESULTS_UNC}/gate6_amt/expA/A4-event_seed42/training_results.json",
        extraction_spec="json_field:best_f1",
    ))

    # =========================================================================
    # TIER 2: Informes autoritativos — Gate 5B
    # =========================================================================

    g5b_doc = f"{BC}/11_GATE_5_LINEA_B_SHOWCASE/INFORME_COMPLETO_GATE5B.md"

    # Test 12 Scoreboard (single seed)
    for arm, expected_s in [("d4a4", 83.8), ("D0", 73.4), ("a4r", 82.0), ("d4-a4r", 79.8)]:
        arm_dir = arm.lower() if arm != "D0" else "D0"
        claims.append(Claim(
            id=f"T2_G5B_T12_{arm}_S",
            tier=2, source_doc=g5b_doc, source_location="Test 12 Scoreboard",
            claim_text=f"{arm} S={expected_s}%",
            expected_value=expected_s, tolerance=0.2,
            category="retrieval", measurement_semantics="best",
            evaluation_protocol="pool=256,queries=500,seed=42,structured",
            alias_of=f"gate5b_test12_{arm}",
            claim_kind="numeric", evidence_level="raw_json",
            data_source_path=f"{DATA}/gate5b_results/{arm_dir}/test12_scoreboard.json",
            extraction_spec="json_field:S",
        ))

    # Test 01 Causal Ablation
    for arm in ["d4a4", "a4r", "d4-a4r"]:
        arm_dir = arm
        claims.append(Claim(
            id=f"T2_G5B_T01_{arm}_ZERO_AUDIO",
            tier=2, source_doc=g5b_doc, source_location="Test 01 Causal Ablation",
            claim_text=f"{arm} zero_audio",
            expected_value={"d4a4": 7.8, "a4r": 4.4, "d4-a4r": 4.4}[arm],
            tolerance=0.2,
            category="causal", measurement_semantics="final",
            evaluation_protocol="pool=256,queries=500,seed=42",
            alias_of=f"gate5b_test01_{arm}",
            claim_kind="numeric", evidence_level="raw_json",
            data_source_path=f"{DATA}/gate5b_results/{arm_dir}/test01_causal_ablation.json",
            extraction_spec="json_field:ablations.zero_audio.S",
        ))

    # Test 02 Parameter-matched
    for variant, expected_s in [("real", 83.0), ("random", 73.6), ("zero", 75.0)]:
        claims.append(Claim(
            id=f"T2_G5B_T02_{variant}_S",
            tier=2, source_doc=g5b_doc, source_location="Test 02 Parameter-Matched",
            claim_text=f"Test02 {variant}={expected_s}%",
            expected_value=expected_s, tolerance=0.5,
            category="causal", measurement_semantics="best",
            evaluation_protocol="pool=256,queries=500,seed=42,structured",
            alias_of=f"gate5b_test02_{variant}",
            claim_kind="numeric", evidence_level="raw_json",
            data_source_path=f"{RESULTS_UNC}/gate5b_param_matched/{variant}/final_results.json",
            extraction_spec="json_field:training.best_S",
        ))

    # Test 02 shuffled (special: asterisk convention)
    claims.append(Claim(
        id="T2_G5B_T02_SHUFFLED_S",
        tier=2, source_doc=g5b_doc, source_location="Test 02 Parameter-Matched",
        claim_text="Test02 shuffled=73.6%* (operational best, raw=73.2%)",
        expected_value=73.6, tolerance=0.5,
        category="causal", measurement_semantics="operational_best",
        evaluation_protocol="pool=256,queries=500,seed=42,structured",
        alias_of="gate5b_test02_shuffled",
        claim_kind="numeric", evidence_level="raw_json",
        data_source_path=f"{RESULTS_UNC}/gate5b_param_matched/shuffled/final_results.json",
        extraction_spec="json_field:training.best_S",
        notes="Raw JSON gives 73.2; docs use 73.6* (early convergence operational value)"
    ))

    # Test 06 CKA
    for arm, expected_cka in [("D0", 0.435), ("d4a4", 0.659), ("a4r", 0.766), ("d4-a4r", 0.794)]:
        arm_dir = arm if arm != "D0" else "D0"
        claims.append(Claim(
            id=f"T2_G5B_T06_{arm}_CKA",
            tier=2, source_doc=g5b_doc, source_location="Test 06 CKA",
            claim_text=f"{arm} CKA cross-enc mean={expected_cka}",
            expected_value=expected_cka, tolerance=0.002,
            category="geometric", measurement_semantics="derived",
            evaluation_protocol="cross-encoder CKA matrix mean [0:4,4:8]",
            alias_of=f"gate5b_test06_{arm}",
            claim_kind="numeric", evidence_level="raw_json",
            data_source_path=f"{DATA}/gate5b_results/{arm_dir}/test06_rsa_cka.json",
            extraction_spec="cka_cross_mean",
        ))

    # Test 04 Transposition Invariance
    for arm, expected_ret in [("D0", 36.8), ("d4a4", 51.3), ("a4r", 59.3), ("d4-a4r", 59.0)]:
        arm_dir = arm if arm != "D0" else "D0"
        claims.append(Claim(
            id=f"T2_G5B_T04_{arm}_TRANS",
            tier=2, source_doc=g5b_doc, source_location="Test 04 Transposition",
            claim_text=f"{arm} ±3semi retention={expected_ret}%",
            expected_value=expected_ret, tolerance=0.5,
            category="geometric", measurement_semantics="derived",
            evaluation_protocol="±3 semitone transposition retention = mean(S[-3],S[+3])/S_baseline",
            alias_of=f"gate5b_test04_{arm}",
            claim_kind="numeric", evidence_level="raw_json",
            data_source_path=f"{DATA}/gate5b_results/{arm_dir}/test04_transposition.json",
            extraction_spec="transposition_retention",
        ))

    # Test 11 Info Retention
    for arm, expected_ir in [("D0", 0.597), ("a4r", 0.712)]:
        arm_dir = arm if arm != "D0" else "D0"
        claims.append(Claim(
            id=f"T2_G5B_T11_{arm}_IR",
            tier=2, source_doc=g5b_doc, source_location="Test 11 Info Retention",
            claim_text=f"{arm} info_retention={expected_ir}",
            expected_value=expected_ir, tolerance=0.002,
            category="mechanistic", measurement_semantics="derived",
            evaluation_protocol="pre-projection decodability ratio",
            alias_of=f"gate5b_test11_{arm}",
            claim_kind="numeric", evidence_level="raw_json",
            data_source_path=f"{DATA}/gate5b_results/{arm_dir}",
            extraction_spec="test11_info_retention",
            notes="May be in test11_preproj_ab_summary.json or test11_preproj_ab.json"
        ))

    # =========================================================================
    # TIER 2: Escalón 3 results
    # =========================================================================

    e3_p2 = f"{DOCS}/01_FRENTES_ACTIVOS/ESCALON_3/Resultados_E3_P2.md"
    e3_p56 = f"{DOCS}/01_FRENTES_ACTIVOS/ESCALON_3/Resultados_E3_P5_P6.md"

    for arm_name, arm_dir, expected_s in [
        ("P2-flat", "p2_baseline_seed42", 0.583),
        ("P2-cqtshift", "p2_cqtshift_seed42", 0.515),
    ]:
        claims.append(Claim(
            id=f"T2_E3_{arm_name.replace('-','_')}_IID_S",
            tier=2, source_doc=e3_p2, source_location="P2 baselines",
            claim_text=f"{arm_name} IID S={expected_s}",
            expected_value=expected_s, tolerance=0.002,
            category="retrieval", measurement_semantics="best",
            evaluation_protocol="escalon3,structured",
            alias_of=arm_dir,
            claim_kind="numeric", evidence_level="raw_json",
            data_source_path=f"{DATA}/escalon3/{arm_dir}/final_results.json",
            extraction_spec="escalon3_iid_s",
            notes=f"Check both final_results.json and final_results_v2.json. Alias: {arm_dir}"
        ))

    # P5-cqtshift scale_ood (best OOD)
    claims.append(Claim(
        id="T2_E3_P5_CQT_SCALE_OOD",
        tier=2, source_doc=e3_p56, source_location="P5 results",
        claim_text="P5-cqtshift scale_ood=0.508",
        expected_value=0.508, tolerance=0.002,
        category="geometric", measurement_semantics="best",
        evaluation_protocol="escalon3,structured,mixed_geometry",
        alias_of="p5_mixed_cqtshift_seed42",
        claim_kind="numeric", evidence_level="raw_json",
        data_source_path=f"{DATA}/escalon3/p5_mixed_cqtshift_seed42/final_results.json",
        extraction_spec="escalon3_scale_ood_s",
    ))

    return claims
