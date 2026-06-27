---
name: validate-hyperparams
description: "Validate hyperparameter and eval protocol consistency BEFORE launching any training run. Prevents ghost variables that make runs non-comparable. Use when setting up a new training run, comparing configs, or auditing comparability post-hoc."
argument-hint: "<config-or-output-dir> [--baseline <baseline-config>]"
allowed-tools: Read, Grep, Glob, Bash(cat *), Bash(python3 -c *), Bash(jq *), Bash(ls *), Bash(test *), Bash(git diff *), Bash(git log *), Bash(git show *)
---

# Hyperparameter Consistency Validator — Phideus

## CRITICAL CONTEXT

A single mismatched hyperparameter makes two runs NON-COMPARABLE.
A "ghost variable" — a param that changed silently between runs — can invalidate
weeks of GPU time and lead to wrong scientific conclusions.
This validator exists to catch every ghost variable BEFORE training starts.

## INVOCATION

**Mode 1 — Explicit baseline:**
```
/validate-hyperparams data/gate10_results/a7_concat_seed42/config.json --baseline data/gate9_results/a7r_seed42/config.json
```

**Mode 2 — Auto-detect baseline (from output dir):**
```
/validate-hyperparams data/gate10_results/a7_concat_seed42/
```

When invoked as `/validate-hyperparams`, parse `$ARGUMENTS` to determine the config
path and optional baseline. If a directory is given, look for `config.json` inside it.
If insufficient info, ask the user.

---

## PHASE 1: Run Identity

Read the config.json (from the path given, or `<output-dir>/config.json`).
If the config does not exist yet, ask the user for the planned CLI arguments and
construct the config dict manually from defaults + overrides.

### Checks

- [ ] `descriptor` value exists in the known descriptor list (see Reference Data below)
- [ ] `output` directory: if it already exists AND contains `best_model.pt` or
      `checkpoint_epoch*.pt`, flag as **WARNING** ("output dir occupied — will overwrite")
- [ ] If output dir exists but is empty or has only config.json: OK

### Report

```
PHASE 1 — Run Identity
  Descriptor:  a7r
  Gate:        10
  Seed:        42
  Output:      data/gate10_results/a7_concat_seed42/
  Status:      [PASS / WARNING: output dir occupied]
```

---

## PHASE 2: Hyperparameter Consistency with Baseline

### Baseline Resolution

If `--baseline` was given explicitly, use that config.json.

Otherwise, auto-detect by experiment context using escalon-specific keys:

**Escalon detection**:
- If config has `lr_audio_unfreeze` → Escalon 1 (Audio-MIDI)
- If config has `lr_enc` → Escalon 2 (Speech-EGG)

**Auto-detect baselines:**

| Context | Baseline path |
|---------|---------------|
| Gate 4.2/4.3/5B (Audio-MIDI) | `data/bias_control_medium/training_outputs/gate43/gate43_d4a4_scratch_30ep/config.json` |
| Gate 8 ctrl | `data/gate8_results/ctrl_seed42/config.json` |
| Gate 9 / A10 / Gate 10 | `data/gate9_results/a7r_seed42/config.json` |
| S2 (Lombard) D0 | First available `data/lombard/d0_seed42/config.json` |

If no baseline can be resolved: **WARNING** — report all params but skip comparison.

### Parameters to Compare

Read both config.json files.

**Critical params** (mismatch = BLOCKER):

| Param (S1) | Param (S2) | Notes |
|------------|------------|-------|
| `lr_audio_unfreeze` | `lr_enc` | Must match exactly |
| `lr_audio_low` | — | Must match exactly |
| `lr_midi` | — | Must match exactly |
| `lr_proj` | `lr_proj` | Must match exactly |
| `lr_ratio` | — | Must match exactly |
| `batch_size` | `batch_size` | Must match exactly |
| `epochs` | `epochs` | Must match exactly |
| `max_batches_per_epoch` | — | Must match exactly |
| `max_val_batches` | — | Must match exactly |
| `warmup_steps` | `warmup_steps` | Must match exactly |
| `freeze_policy` | — | Must match exactly |
| `from_scratch` | — | Must match exactly |
| `ratio_weight` | — | Must match exactly |

**Non-critical params** (mismatch = OK, just note):

| Param | Why OK |
|-------|--------|
| `gate` | Label only |
| `output` | Path only |
| `descriptor` | Experimental variable — expected to differ |
| `seed` | Expected to differ for multi-seed |
| `checkpoint_type`, `eval_compatible` | Metadata |
| `device`, `num_workers`, `embed_batch_size` | Environment/performance |
| `mode`, `maestro_dir`, `family` | Invocation context |

**Smart defaults** (treat as equivalent to MISSING):

| Param | Default value |
|-------|---------------|
| `lr_floor` | `0.0` |
| `lr_hold_fraction` | `0.0` |
| `lr_cosine_ref_epochs` | `0` |
| `lr_tail_end` | `0.0` |

If one config has `lr_floor: 0.0` and the other lacks the key: **OK**.
If one config has `lr_floor: 0.001` and the other lacks it: **BLOCKER**.

### Cross-Escalon Guard

If S1 config is compared against S2 config: emit **BLOCKER**.
"Cannot compare Escalon 1 (Audio-MIDI) with Escalon 2 (Speech-EGG)."

### Report Format

```
PHASE 2 — Hyperparameter Consistency
  Baseline: data/gate9_results/a7r_seed42/config.json

  | Param                  | Baseline | This Run | Match? |
  |------------------------|----------|----------|--------|
  | lr_audio_unfreeze      | 1e-05    | 1e-05    | OK     |
  | batch_size             | 16       | 16       | OK     |
  | lr_floor               | 0.0      | MISSING  | OK (default) |
  ...

  Status: [PASS / FAIL]
```

---

## PHASE 3: Eval Protocol Consistency

Compare evaluation parameters that affect metric comparability.

### Checks

**structured_eval_epochs**:
- Identical: OK
- This run is a SUPERSET of baseline: **WARNING** (extra eval points are fine,
  but comparing "best over N epochs" vs "best over M epochs" introduces selection bias)
- This run is a SUBSET: **BLOCKER** (missing eval points = cannot compare at same epochs)
- Disjoint sets: **BLOCKER**

**Eval defaults** (from eval script, not config.json):
- `pool_size`: 256 (S1) or 128 (S2)
- `n_queries`: 500 (S1)
- `seed`: 42

**Resume status**:
- If one run was resumed and the other was not: **WARNING** — resume changes CUDA
  trajectory. Expected variance ~1-2pp for same seed.

### Report Format

```
PHASE 3 — Eval Protocol Consistency
  structured_eval_epochs:
    Baseline:  [5, 10, 15, 20, 25, 28, 29, 30]
    This Run:  [5, 10, 15, 20, 25, 28, 29, 30]
    Status:    OK (identical)

  Resume status:
    Baseline:  resumed from epoch 10
    This Run:  not resumed (ran straight through)
    Status:    WARNING (different CUDA trajectory — expect ~1-2pp variance)

  Status: [PASS / WARNING]
```

---

## PHASE 4: Code Consistency

Check whether the training script has been modified since the baseline was trained.

### Procedure

1. Identify the training script:
   - S1 (Audio-MIDI): `experiments/bias_control/gate43_scratch/gate43_scratch_training.py`
   - S2 (Speech-EGG): `experiments/bias_control/escalon2/train_escalon2*.py`

2. Check for uncommitted changes:
   ```bash
   git diff HEAD -- <training_script> | head -100
   ```

3. If there are changes since baseline, categorize:
   - **Training loop / loss / optimizer**: BLOCKER
   - **Model forward pass / architecture**: BLOCKER
   - **DataLoader / collate / augmentation**: BLOCKER
   - **New descriptor/mechanism (not used by this run)**: OK
   - **Logging / CLI args**: OK
   - **Eval code changes**: WARNING

### Report Format

```
PHASE 4 — Code Consistency
  Training script: experiments/bias_control/gate43_scratch/gate43_scratch_training.py
  Uncommitted changes: None
  Changes since baseline: +319 lines (all new descriptors/mechanisms, not used by this arm)
  Status: [PASS / WARNING / BLOCKER]
```

---

## PHASE 5: Report + Verdict

Compile results from all phases.

```
=== HYPERPARAMETER VALIDATION REPORT ===
Run: <descriptor> seed<N> gate<G>
Baseline: <baseline path>
Date: <current date>

PHASE 1 — Run Identity:           [PASS/WARNING/FAIL]
PHASE 2 — Hyperparameter Match:   [PASS/FAIL]
PHASE 3 — Eval Protocol Match:    [PASS/WARNING]
PHASE 4 — Code Consistency:       [PASS/WARNING/BLOCKER]

MISMATCHES:
  [P2] lr_floor: baseline=MISSING, this_run=0.0 → OK (default 0.0 = absent)
  [P3] resume: baseline=resumed_from_ep10, this_run=straight → WARNING (~1-2pp variance)

VERDICT: [COMPARABLE / NOT COMPARABLE / COMPARABLE WITH CAVEATS]
```

### Verdict Logic

- **COMPARABLE**: All phases PASS. No critical mismatches.
- **NOT COMPARABLE**: Any BLOCKER in Phase 2, 3, or 4.
- **COMPARABLE WITH CAVEATS**: No blockers, but WARNINGs exist. List each caveat.

If NOT COMPARABLE, show the exact CLI arguments needed to make the run match the baseline.

---

## REFERENCE DATA

### Known Descriptors (S1 — Audio-MIDI)

Single: `d0`, `d1`, `d4`, `a4`, `a4r`, `a7`, `a7r`, `a9`, `a9r`,
`a10a`, `a10ar`, `a10b`, `a10br`, `a10c`, `a10cr`, `a10d`, `a10dr`, `a10e`, `a10er`
Dual: `d4a4`, `d4-a4r`
Mechanism variants: `*-pca`, `*-ab` (Gate 10)

### Known Descriptors (S2 — Speech-EGG)

`v4_lin`, `v4_log`, `h_series`, `a4_16k`

### S1 Canonical Hyperparameters

```json
{
  "lr_audio_unfreeze": 1e-05,
  "lr_audio_low": 5e-06,
  "lr_midi": 5e-05,
  "lr_proj": 0.0001,
  "lr_ratio": 0.0005,
  "batch_size": 16,
  "epochs": 30,
  "max_batches_per_epoch": 1000,
  "max_val_batches": 846,
  "warmup_steps": 200,
  "freeze_policy": "run-d",
  "from_scratch": true,
  "ratio_weight": 0.1
}
```

### S2 Canonical Hyperparameters

```json
{
  "lr_enc": 0.0005,
  "lr_proj": 0.001,
  "batch_size": 64,
  "epochs": 30,
  "warmup_steps": 500,
  "seed": 42
}
```

### Known Batch Size Confound (S2)

The `attn_bias` mechanism runs in S2 used `batch_size=48` instead of 64.
This is a KNOWN CONFOUND documented in the 2026-03-14 audit.
If validating an `attn_bias` S2 run: emit WARNING referencing this confound.
