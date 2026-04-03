#!/usr/bin/env python3
"""
Phideus Numeric Claims Audit — Orchestrator.
Run: python -m tools.claims_audit.run_audit
"""
import sys
import os
from datetime import datetime
from collections import Counter

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.claims_audit.claim_registry import build_registry
from tools.claims_audit.claim_verifier import verify_claim


def generate_report(results, claims):
    """Generate markdown audit report."""
    lines = []
    lines.append("# Phideus Numeric Claims Audit Report")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("")

    # Executive summary
    status_counts = Counter(r.status for r in results)
    total = len(results)
    lines.append("## Executive Summary")
    lines.append(f"- **Total claims**: {total}")
    for status in ["PASS", "FAIL", "WARN", "MISSING", "DOC_ONLY", "BITACORA_ONLY", "STALE"]:
        count = status_counts.get(status, 0)
        pct = count / total * 100 if total > 0 else 0
        lines.append(f"- **{status}**: {count} ({pct:.0f}%)")
    lines.append("")

    # Results by tier
    claim_map = {c.id: c for c in claims}
    for tier in range(1, 5):
        tier_results = [(r, claim_map[r.claim_id]) for r in results if claim_map[r.claim_id].tier == tier]
        if not tier_results:
            continue
        tier_names = {
            1: "Docs Canónicos Públicos",
            2: "Informes Autoritativos",
            3: "Docs Transversales",
            4: "Bitácoras y Notas Operativas",
        }
        lines.append(f"## Tier {tier}: {tier_names.get(tier, '')}")
        lines.append("")
        lines.append("| Status | Claim ID | Documented | Extracted | Delta | Source | Notes |")
        lines.append("|--------|----------|-----------|-----------|-------|--------|-------|")
        for r, c in sorted(tier_results, key=lambda x: x[0].status):
            doc_val = f"{r.documented_value}" if c.claim_kind == "numeric" else c.claim_text[:30]
            ext_val = f"{r.extracted_value}" if r.extracted_value is not None else "—"
            delta = f"{r.delta}" if r.delta is not None else "—"
            src = os.path.basename(r.source_file)[:40] if r.source_file else "—"
            notes = r.notes[:60] if r.notes else ""
            status_icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "MISSING": "🔍",
                           "DOC_ONLY": "📄", "BITACORA_ONLY": "📓", "STALE": "🕐"}.get(r.status, "?")
            lines.append(f"| {status_icon} {r.status} | {r.claim_id} | {doc_val} | {ext_val} | {delta} | {src} | {notes} |")
        lines.append("")

    # Traceability holes
    holes = [(r, claim_map[r.claim_id]) for r in results
             if r.status in ("MISSING", "DOC_ONLY", "BITACORA_ONLY")]
    if holes:
        lines.append("## Agujeros de Trazabilidad")
        lines.append("")
        lines.append("| Claim ID | Status | Evidence Level | Description | Action Needed |")
        lines.append("|----------|--------|---------------|-------------|---------------|")
        for r, c in holes:
            lines.append(f"| {r.claim_id} | {r.status} | {c.evidence_level} | {c.claim_text[:50]} | {r.notes[:50]} |")
        lines.append("")

    # Failures
    failures = [(r, claim_map[r.claim_id]) for r in results if r.status == "FAIL"]
    if failures:
        lines.append("## Inconsistencias Numéricas (FAIL)")
        lines.append("")
        for r, c in failures:
            lines.append(f"### {r.claim_id}")
            lines.append(f"- **Documented**: {r.documented_value}")
            lines.append(f"- **Extracted**: {r.extracted_value}")
            lines.append(f"- **Delta**: {r.delta} (tolerance: {c.tolerance})")
            lines.append(f"- **Source**: `{r.source_file}`")
            lines.append(f"- **Doc**: `{c.source_doc}`")
            if r.notes:
                lines.append(f"- **Notes**: {r.notes}")
            lines.append("")

    # Stale state
    stale = [(r, claim_map[r.claim_id]) for r in results if r.status == "STALE"]
    if stale:
        lines.append("## Estado Documental Desactualizado (STALE)")
        lines.append("")
        for r, c in stale:
            lines.append(f"- **{r.claim_id}**: {r.notes}")
        lines.append("")

    # UNC sync list
    unc_items = [(r, claim_map[r.claim_id]) for r in results
                 if r.status in ("MISSING", "DOC_ONLY", "BITACORA_ONLY")
                 and claim_map[r.claim_id].claim_kind == "numeric"]
    if unc_items:
        lines.append("## Lista de Sincronización UNC")
        lines.append("")
        lines.append("Archivos a solicitar a Claude UNC para cerrar agujeros de trazabilidad:")
        lines.append("")
        seen_gates = set()
        for r, c in unc_items:
            gate = c.alias_of.split("_")[0] if "_" in c.alias_of else c.alias_of
            if gate not in seen_gates:
                seen_gates.add(gate)
                lines.append(f"- **{gate}**: {c.claim_text} → {c.notes or 'raw outputs needed'}")
        lines.append("")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("PHIDEUS NUMERIC CLAIMS AUDIT")
    print("=" * 60)

    # Build registry
    print("\n[1/3] Building claims registry...")
    claims = build_registry()
    print(f"  → {len(claims)} claims registered")

    # Verify all claims
    print("\n[2/3] Verifying claims against data sources...")
    results = []
    for c in claims:
        r = verify_claim(c)
        results.append(r)
        icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "MISSING": "🔍",
                "DOC_ONLY": "📄", "BITACORA_ONLY": "📓", "STALE": "🕐"}.get(r.status, "?")
        print(f"  {icon} {r.claim_id}: {r.status}"
              + (f" (doc={r.documented_value}, ext={r.extracted_value}, Δ={r.delta})" if r.extracted_value is not None else ""))

    # Generate report
    print("\n[3/3] Generating report...")
    report = generate_report(results, claims)
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AUDIT_REPORT.md")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  → Report saved to {report_path}")

    # Summary
    status_counts = Counter(r.status for r in results)
    print(f"\n{'='*60}")
    print("SUMMARY:")
    for s in ["PASS", "FAIL", "WARN", "MISSING", "DOC_ONLY", "BITACORA_ONLY", "STALE"]:
        if status_counts.get(s, 0) > 0:
            print(f"  {s}: {status_counts[s]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
