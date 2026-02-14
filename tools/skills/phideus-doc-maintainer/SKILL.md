---
name: phideus-doc-maintainer
description: Dynamic documentation maintenance for Phideus with active-front detection and policy-based target selection. Use when updating project docs after experiment progress, gate decisions, roadmap changes, status snapshots, protocol changes, or document restructures. Supports hybrid front detection (auto + manual override), minimal global updates, and legacy-safe behavior.
---

# Phideus Doc Maintainer

## Overview

Maintain Phideus documentation with a consistent policy that adapts to the active experiment front.
Detect the active front, select target documents, enforce local rules, and produce a traceable update summary.

## Workflow

1. Classify the change event (`gate_result_update`, `roadmap_or_strategy_update`, `protocol_or_policy_update`, `new_doc_or_major_reorg`, `status_snapshot_update`).
2. Detect active front with `scripts/detect_front.py`.
3. If detection is ambiguous, ask the user to confirm front before editing.
4. Select target docs with `scripts/select_targets.py`.
5. Apply edits using tiered style rules from `references/style_profiles.md`.
6. Validate changed paths with `scripts/consistency_check.py`.
7. Report: detected front, evidence, edited docs, skipped docs by policy.

## Front Detection

Run:

```bash
python3 tools/skills/phideus-doc-maintainer/scripts/detect_front.py \
  --front auto \
  --hints gate4 ratio auxiliary \
  --paths Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/ROADMAP_BIAS_CONTROL.md experiments/bias_control/gate4_ratio_auxiliary.py
```

Use `--front <id>` to override auto detection.
Supported known ids: `bias_control`, `escalon_1`, `uoemd`, `experimentos`, `legacy`.
Custom ids are allowed and treated as manual overrides.

Exit codes:
- `0`: resolved
- `2`: ambiguous (must ask user)

## Target Selection

Run:

```bash
python3 tools/skills/phideus-doc-maintainer/scripts/select_targets.py \
  --front bias_control \
  --event-type gate_result_update \
  --collab-mode off \
  --experimental-advance \
  --technical-decision
```

Policy implemented:
- Front docs + minimal global docs.
- Global minimal set uses event flags:
  - `Documents/00_TRONCAL/bitacora_desarrollo.md` for experimental advance or technical decision.
  - `Documents/00_TRONCAL/Proyecto_Estado_Actual.md` for status/focus/decision changes.
  - `Documents/00_TRONCAL/INDICE_DOCUMENTACION.md` only on doc structure change.
  - `README.md` only for trunk milestone or focus change.
- Legacy (`Documents/90_ARCHIVO_GLOBAL/Legacy/**`, `Documents/03_FRENTES_CERRADOS/UOEMD/**`) excluded by default unless front is legacy/uoemd or request is explicit.

## Consistency Checks

Run before final output:

```bash
python3 tools/skills/phideus-doc-maintainer/scripts/consistency_check.py \
  --changed Documents/00_TRONCAL/bitacora_desarrollo.md \
  --changed Documents/00_TRONCAL/Proyecto_Estado_Actual.md \
  --front bias_control \
  --collab-mode off
```

Checks include:
- forbidden paths (`PENDIENTES.md`, `CLAUDE.md`)
- `COLLAB/*` touched while collab is off
- legacy updates without explicit allowance
- tier-A coherence warnings

## Style Rules

Read `references/style_profiles.md` before writing.

- Tier A docs: reinforced design and high readability.
- Tier B docs: sober and concise style.
- Narrative explanatory style is mandatory for:
  - `README.md`
  - `Documents/00_TRONCAL/bitacora_desarrollo.md`
  - `Documents/00_TRONCAL/Proyecto_Estado_Actual.md`
  - `Documents/04_TRANSVERSAL/TEORIA_Y_FUNDAMENTOS/**`
  (reference tone: `Documents/01_FRENTES_ACTIVOS/BIAS_CONTROL/06_GATE_4_2_RATIO_CENTRICO/Explicacion_gate4.2_claude.md`).
- Keep moderate length by default.
- Preserve the existing visual design of each target document.
- Do not edit `PENDIENTES.md` or `CLAUDE.md`.
- Respect `CODEX.md` policy and collab mode.

## References

- `references/front_registry.yaml`: front records and paths
- `references/detection_rules.yaml`: scoring rules and fallback
- `references/global_update_policy.yaml`: event policy and safeguards
- `references/style_profiles.md`: editorial tiers
- `references/update_workflow.md`: operational checklist
