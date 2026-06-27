"""
Claim verifier: compare documented claims against extracted values.
"""
from dataclasses import dataclass
from typing import Optional
from . import source_extractors as ext


@dataclass
class VerificationResult:
    claim_id: str
    status: str  # PASS, FAIL, WARN, MISSING, DOC_ONLY, BITACORA_ONLY, STALE
    documented_value: float
    extracted_value: Optional[float]
    source_file: str
    delta: Optional[float]
    notes: str


def verify_claim(claim) -> VerificationResult:
    """Verify a single claim against its data source."""

    # Handle non-numeric claims (state checks)
    if claim.claim_kind == "state":
        if claim.extraction_spec == "file_exists":
            exists = ext.file_exists(claim.data_source_path)
            if exists and "corriendo" in claim.claim_text.lower():
                return VerificationResult(
                    claim_id=claim.id, status="STALE",
                    documented_value=claim.expected_value,
                    extracted_value=None,
                    source_file=claim.data_source_path,
                    delta=None,
                    notes=f"Doc says '{claim.claim_text}' but results file exists"
                )
            return VerificationResult(
                claim_id=claim.id,
                status="PASS" if exists else "MISSING",
                documented_value=claim.expected_value,
                extracted_value=None,
                source_file=claim.data_source_path,
                delta=None, notes=""
            )
        # Generic state claim without backing
        if not claim.data_source_path:
            return VerificationResult(
                claim_id=claim.id, status="DOC_ONLY",
                documented_value=claim.expected_value,
                extracted_value=None, source_file="",
                delta=None, notes=claim.notes
            )

    # Handle claims with no data source
    if claim.evidence_level in ("doc_only",):
        return VerificationResult(
            claim_id=claim.id, status="DOC_ONLY",
            documented_value=claim.expected_value,
            extracted_value=None, source_file="",
            delta=None, notes=claim.notes
        )

    if claim.evidence_level in ("bitacora_only",):
        return VerificationResult(
            claim_id=claim.id, status="BITACORA_ONLY",
            documented_value=claim.expected_value,
            extracted_value=None, source_file="",
            delta=None, notes=claim.notes
        )

    if claim.evidence_level in ("docs_bitacora",):
        return VerificationResult(
            claim_id=claim.id, status="DOC_ONLY",
            documented_value=claim.expected_value,
            extracted_value=None, source_file="",
            delta=None,
            notes=f"docs+bitacora backed, no raw artifact. {claim.notes}"
        )

    # Extract value based on extraction_spec
    extracted = None
    source_file = claim.data_source_path

    spec = claim.extraction_spec
    if spec.startswith("json_field:"):
        key_path = spec.split(":", 1)[1]
        extracted = ext.extract_json_field(claim.data_source_path, key_path)

    elif spec.startswith("multiseed_mean:"):
        key_path = spec.split(":", 1)[1]
        stats = ext.extract_multiseed_stats(claim.data_source_path, key_path)
        if stats:
            extracted = stats['mean'] * 100  # convert to percentage
            source_file = f"{stats['n']} files: {stats['files'][0]}..."

    elif spec == "cka_cross_mean":
        extracted = ext.extract_cka_cross_mean(claim.data_source_path)

    elif spec == "escalon3_iid_s":
        extracted = ext.extract_escalon3_iid_s(claim.data_source_path)

    elif spec == "escalon3_scale_ood_s":
        extracted = ext.extract_escalon3_scale_ood_s(claim.data_source_path)

    elif spec == "best_from_eval_epochs":
        extracted = ext.extract_best_from_eval_epochs(claim.data_source_path)

    elif spec == "lombard_d0_best_s":
        extracted = ext.extract_lombard_d0_best_s(claim.data_source_path)

    elif spec == "transposition_retention":
        extracted = ext.extract_transposition_retention(claim.data_source_path)

    elif spec == "test11_info_retention":
        # Try summary first, then main file
        import os, glob
        base = claim.data_source_path
        for candidate in [
            os.path.join(base, 'test11_preproj_ab_summary.json'),
            os.path.join(base, 'test11_preproj_ab.json'),
        ]:
            if os.path.isfile(candidate):
                val = ext.extract_json_field(candidate, 'info_retention_events.ratio')
                if val is not None:
                    extracted = val
                    source_file = candidate
                    break
                # Try alternate key paths
                val = ext.extract_json_field(candidate, 'retention_ratio')
                if val is not None:
                    extracted = val
                    source_file = candidate
                    break

    # Handle missing extraction
    if extracted is None:
        return VerificationResult(
            claim_id=claim.id, status="MISSING",
            documented_value=claim.expected_value,
            extracted_value=None,
            source_file=source_file,
            delta=None,
            notes=f"Could not extract value. {claim.notes}"
        )

    # Normalize scale: if extracted is fraction (< 1.0) and expected is percentage (> 1.0)
    if extracted is not None and claim.expected_value > 1.0 and extracted < 1.0:
        extracted = extracted * 100
    # Also handle inverse: expected is fraction, extracted is percentage
    if extracted is not None and claim.expected_value < 1.0 and extracted > 1.0:
        extracted = extracted / 100

    # Compare
    delta = abs(extracted - claim.expected_value)
    if delta <= claim.tolerance:
        status = "PASS"
    elif delta <= claim.tolerance * 2:
        status = "WARN"
    else:
        status = "FAIL"

    # Force WARN if evidence level indicates methodological concerns
    if status == "PASS" and claim.evidence_level == "summary_json" and "WARN:" in claim.notes:
        status = "WARN"

    return VerificationResult(
        claim_id=claim.id, status=status,
        documented_value=claim.expected_value,
        extracted_value=round(extracted, 4),
        source_file=source_file,
        delta=round(delta, 4),
        notes=claim.notes
    )
