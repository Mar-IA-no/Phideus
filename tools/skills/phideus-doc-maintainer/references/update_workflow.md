# Update Workflow

1. Capture request context.
2. Classify event type.
3. Detect active front with `detect_front.py`.
4. If result is `ambiguous`, request confirmation before edits.
5. Select targets with `select_targets.py`.
6. Apply updates with tiered style rules.
   - If a target matches narrative-doc policy, enforce explanatory narrative without changing visual design.
7. Validate with `consistency_check.py`.
8. Report:
   - detected front
   - confidence and evidence
   - changed docs
   - skipped docs by policy

## Event type quick map

- `gate_result_update`: experiment/gate output, metrics, run decisions.
- `roadmap_or_strategy_update`: plan changes, phase reshaping, priorities.
- `protocol_or_policy_update`: operating rules, documentation policy, agent policy.
- `new_doc_or_major_reorg`: moved/new/renamed documentation files.
- `status_snapshot_update`: state-only refresh without major strategy change.

## Maintenance routine

1. Edit the blueprint under `tools/skills/phideus-doc-maintainer/`.
2. Validate with:
   `python3 /root/.codex/skills/.system/skill-creator/scripts/quick_validate.py tools/skills/phideus-doc-maintainer`
3. Sync runtime copy:
   `rsync -a --delete tools/skills/phideus-doc-maintainer/ /root/.codex/skills/phideus-doc-maintainer/`
4. Restart Codex when needed so the updated skill metadata is reloaded.
