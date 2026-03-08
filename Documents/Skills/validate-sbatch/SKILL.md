---
name: validate-sbatch
description: "Validate SLURM sbatch scripts before submission on Mendieta/CCAD. MUST be invoked before ANY sbatch submission. Checks paths, directives, environment, dependencies. Prevents wasting queue slots on preventable errors."
argument-hint: "<script-path>"
allowed-tools: Read, Grep, Glob, Bash(test *), Bash(ls *), Bash(stat *), Bash(bash -n *), Bash(sbatch --test-only *), Bash(python -c *), Bash(pip show *), Bash(conda *), Bash(module *), Bash(seff *), Bash(scontrol *), Bash(sinfo *), Bash(squeue *)
---

# SLURM sbatch Validator — Mendieta/CCAD UNC

## CRITICAL CONTEXT

Queue slots on Mendieta are PRECIOUS. Jobs wait hours or DAYS in queue.
A script that fails in 13 seconds wastes that entire wait.
This validator exists to make that IMPOSSIBLE.

## EXECUTION PROTOCOL

Run ALL 5 phases sequentially. If ANY phase produces a BLOCKER, STOP and report.
Do NOT suggest submitting until all phases pass.
After Phase 5 report, present the STRATEGIC OPTIONS panel to help the user
decide the best submission strategy.

When invoked as `/validate-sbatch <path>`, validate the script at `$ARGUMENTS`.
If no argument given, ask the user which script to validate.

---

## PHASE 1: Static Analysis (read the script, check patterns)

Read the entire script. Then check each item below. Mark as BLOCKER, WARNING, or OK.

### 1.1 Shebang & Shell Safety
- [ ] First line is `#!/bin/bash`
- [ ] `set -eo pipefail` present (or `set -e` minimum)
- [ ] **BLOCKER**: `set -u` or `set -euo` present → MUST be removed. Reason: `/etc/profile` on Mendieta has undefined `LC_ALL`. With `-u`, the script dies immediately on `. /etc/profile`.
- [ ] Run `bash -n <script>` to check for syntax errors → any error is a BLOCKER

### 1.2 SLURM Directives
- [ ] `--partition=` is `multi` or `short` (only valid partitions on Mendieta)
- [ ] **BLOCKER**: `--time` exceeds partition limit (`short`: 1h, `multi`: 2 days / `2-00:00:00`)
- [ ] **WARNING**: `--mem=0` → blocks scheduling on `mix`-state nodes. Recommend specific value (`32G`, `48G`, `60G`)
- [ ] **BLOCKER**: `--gpus-per-task` used WITHOUT `--gres=gpu:N`. Since 2026-02-25, SLURM rejects `--gpus-per-task` alone. Fix: replace with `--gres=gpu:1` (or `gpu:2` for DDP)
- [ ] `--output=` specified with valid parent directory
- [ ] **WARNING**: `--error=` NOT specified → Python logs go to stderr, will be LOST or merged with bash stdout. Must add `--error=` line
- [ ] `--cpus-per-task` is between 1-20 (Mendieta nodes have 20 cores)
- [ ] **WARNING**: `--cpus-per-task` > 10 → leaves less room for system processes and co-located jobs
- [ ] If job is long (>6h): `--signal=B:SIGTERM@595` or similar is present
- [ ] `--nodes=1` (multi-node only for advanced DDP, verify intent)

### 1.3 Environment Setup Block
Check that these appear IN THIS ORDER after SBATCH directives:

```
. /etc/profile                    ← MUST come first
module load gcc cuda              ← MUST come after /etc/profile
source .../activate phideus       ← conda activation
export OMP_NUM_THREADS=...        ← prevent thread explosion
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1         ← MANDATORY for log visibility
```

- [ ] **BLOCKER**: `. /etc/profile` missing or comes AFTER `module load` → modules won't load, script fails immediately
- [ ] **BLOCKER**: `module load` missing → no CUDA, no compiler
- [ ] **BLOCKER**: conda activation missing or references wrong env
- [ ] **BLOCKER**: `PYTHONUNBUFFERED=1` missing → Python buffers ALL output, logs appear empty during job, monitoring impossible. This is mandatory.
- [ ] **WARNING**: `OMP_NUM_THREADS` not set → OpenMP may spawn excessive threads
- [ ] **WARNING**: `PYTORCH_CUDA_ALLOC_CONF` not set → risk of CUDA OOM from fragmentation

### 1.4 Array Job Logic (if `--array` present)
- [ ] Array indices match the number of items being decoded
- [ ] Index arithmetic is correct (division/modulo for nested arrays)
- [ ] All array elements are valid (descriptor names, seed values, config names)
- [ ] No off-by-one: `--array=0-N` means N+1 jobs

---

## PHASE 2: Path Verification (the #1 cause of wasted queue slots)

This is the MOST IMPORTANT phase. Extract EVERY path from the script, resolve variables, and verify existence on disk.

### 2.1 Variable Resolution
Parse variable assignments in order. Build a resolution map. Example:
```
REPO=<repo-root>
MAESTRO_SRC=$REPO/data/maestro_v3/maestro-v3.0.0
→ resolves to: <repo-root>/data/maestro_v3/maestro-v3.0.0
```

### 2.2 Check EVERY resolved path

For each path found in the script, run `test -e <path>` or `ls <path>`.

**Paths to extract and verify:**

| What | How to find | What to check |
|------|-------------|---------------|
| MAESTRO_SRC | `grep MAESTRO` | Directory exists, contains `maestro-v3.0.0.json` |
| Python script | `grep "srun python"` or `grep "python "` | File exists and is readable |
| Foundation checkpoint | `grep "foundation\|\.pt"` | File exists (if referenced) |
| Model checkpoints | `grep "best_model\|checkpoint\|\.pt"` | Each file exists |
| Output log dir | `--output=` parent directory | Directory exists |
| Error log dir | `--error=` parent directory | Directory exists |
| Conda activate path | `grep "activate"` | File exists |
| Any `cp` source | `grep "^cp "` | Source exists |

For EACH path that doesn't exist: **BLOCKER** — report the path, what it should be, and suggest the fix.

### 2.3 Known Correct Paths on Mendieta (2026-03 reference)

Use this table to suggest corrections when paths are wrong:

| Resource | CORRECT path on Mendieta | WRONG paths seen before |
|----------|--------------------------|-------------------------|
| MAESTRO dataset | `$REPO/data/maestro_v3/maestro-v3.0.0` | `/home/$USER/data/maestro-v3.0.0` (flat), `/home/$USER/data/maestro_v3/maestro-v3.0.0` (absolute fuera del repo) |
| Project repo | `<repo-root>` | — |
| Conda activate | `$HOME/miniconda3/bin/activate` | — |
| Conda env | `phideus` | — |
| Logs directory | `$REPO/logs/` | — |
| Gate 5B models | `$REPO/models/gate5b/{D0,d4a4,a4r,d4-a4r}/best_model.pt` | — |
| Gate 5B results | `$REPO/data/gate5b_results/` | — |
| Gate 6 results | `$REPO/data/gate6_results/` | — |
| Scratch (runtime) | `/scratch/$SLURM_JOB_ID` | Cannot verify pre-submission (created by SLURM) |

### 2.4 Path Origin Detection
If a path looks like it came from LOCAL server (the other dev environment):
- LOCAL uses `/home/$USER/data/maestro-v3.0.0` (flat, no maestro_v3 subdirectory)
- LOCAL uses `--workers 14` (12th gen i5, different core count)
- LOCAL may use `set -euo pipefail` (no /etc/profile issue there)

Flag these as: "This path appears to be from LOCAL server. On Mendieta, use: ..."

---

## PHASE 3: Dependency Verification

### 3.1 Python Package Check
Extract the Python script path from Phase 2. Read its import statements.
For each non-standard import, verify it's installed:

```bash
source $HOME/miniconda3/bin/activate phideus
pip show <package> 2>/dev/null | head -2
```

**Known required packages by gate:**
- Gate 4.x/5B: `torch`, `torchaudio`, `transformers` (MERT), `librosa`, `mir_eval`
- Gate 6 Exp A/B: `transkun`, `pretty_midi`, `midi2audio`
- Gate 6 Exp C: `torch`, `mir_eval` (lighter deps)

Any missing package is a **BLOCKER**.

### 3.2 Conda Environment Check
```bash
conda env list | grep phideus
```
If env doesn't exist: **BLOCKER**.

### 3.3 CUDA/Module Check
Verify modules are available (this runs on login node, modules should be loadable):
```bash
module avail cuda 2>&1 | head -5
```

### 3.4 Runtime Sanity Checks for Transkun / Gate 6
If the sbatch script launches `transkun_a4_finetune.py`, `gate6_transkun_*.sh`, or any
Transkun-based training, inspect the target Python file and verify these two runtime fixes:

- [ ] **BLOCKER**: variable-length audio batching handled before `torch.stack`
  - Look for `min_len = min(...)` + truncation, explicit padding, or a collate path that already normalizes shapes.
  - Reason: MAESTRO v3 mixes 44.1 kHz and 48 kHz sources. Raw chunks can have different lengths in the same batch.
- [ ] **BLOCKER**: input audio marked with `requires_grad_(True)` if the model uses `torch.utils.checkpoint` internally and the wrapper/frozen backbone path needs it
  - Look for `.requires_grad_(True)` after `.to(device)` on the audio tensor, or equivalent logic.
  - Reason: some Transkun paths fail in backward with `element 0 of tensors does not require grad...`

If either fix is missing, mark it as a BLOCKER even if the sbatch script itself is syntactically valid.

---

## PHASE 4: SLURM Dry Run & Scheduling Intelligence

### 4.1 Syntax Validation
```bash
sbatch --test-only <script-path>
```
- If it returns a job ID estimate: PASS
- If it returns an error: **BLOCKER** — report the error
- Note: `--test-only` time estimates are PESSIMISTIC (worst-case). Do not alarm the user about long wait times.

### 4.2 Time Request Optimization
Check if `--time` is significantly over-estimated for the workload. Overly generous
time requests HURT scheduling — SLURM cannot backfill a 24h job into a 6h gap.

If there are completed jobs of the same type, run `seff <jobid>` to check actual
wall-clock usage vs requested time. Suggest tighter `--time` if actual usage was
<50% of requested.

**Reference durations (empirical, Mendieta A30):**
- Gate 4.x screening 5ep: ~2-3h (request 6h)
- Gate 4.x 30ep scratch: ~12-25h depending on architecture (request 48h)
- Gate 5B test13g posthoc decoder: ~5h (request 12h)
- Gate 5B test11 pre-proj: ~9h (request 24h)
- Gate 6 Exp C per arm: ~4-6h estimated (request 24h — could reduce to 8-10h)
- MAESTRO copy to scratch: ~22 min overhead always

### 4.3 Queue Position Check
Run:
```bash
squeue -p multi -t PENDING -o "%.10i %.12u %.8Q %R" --sort=-Q | head -20
```
Show the user where their job would land in queue relative to other pending jobs.

---

## PHASE 5: Final Report

Present results in this exact format:

```
=== SBATCH VALIDATION REPORT ===
Script: <path>
Date: <date>

PHASE 1 — Static Analysis:     [PASS/FAIL]
PHASE 2 — Path Verification:   [PASS/FAIL]  ← most critical
PHASE 3 — Dependencies:        [PASS/FAIL]
PHASE 4 — SLURM Dry Run:       [PASS/FAIL]

BLOCKERS (must fix before submission):
  1. [P2] MAESTRO_SRC=/home/$USER/data/... does not exist
     Fix: MAESTRO_SRC=$REPO/data/maestro_v3/maestro-v3.0.0
  2. ...

WARNINGS (recommended fixes):
  1. [P1] --error not specified, Python logs will be lost
     Fix: Add #SBATCH --error=.../logs/<name>_%j.err
  2. ...

VERDICT: [READY TO SUBMIT / DO NOT SUBMIT — N blockers found]
```

If VERDICT is READY TO SUBMIT, present the STRATEGIC OPTIONS panel below.
If VERDICT is DO NOT SUBMIT, offer to fix all blockers automatically.

---

## STRATEGIC OPTIONS PANEL

After validation passes, present these options to the user. These are NOT mandatory
steps — they are tools to choose from depending on the situation. Assess which ones
are relevant and recommend accordingly.

### Option A: Direct Submit (`sbatch` to `multi`)
**When to use:** Script is well-tested, same type of job has succeeded before, queue is short.
**Risk:** If something fails at runtime (GPU, data loading), you lose your queue slot.
```bash
sbatch <script-path>
```

### Option B: Preflight on `short` Partition
**When to use:** New script type, first time running this experiment, paths recently changed,
dependencies recently installed. Validates EVERYTHING on a real compute node.
**Cost:** ~30 min (MAESTRO copy + 1 epoch). `short` jobs enter faster via backfill.
**How:** Generate a `short` version of the script that runs the same setup + 1 epoch/iteration,
then wait for it to complete. Only submit the real job if preflight passes.
```bash
# Auto-generate: same env/paths, but --partition=short, --time=00:55:00,
# --epochs 1 or minimal iterations, single arm only (not full array)
```

### Option C: Interactive Debug Session (`srun`)
**When to use:** Something is failing and you need to poke around interactively on a
compute node — check GPU, test imports, inspect scratch, try commands manually.
**Cost:** Ties up a terminal. Enters via queue like any job.
```bash
srun -p short --gres=gpu:1 --cpus-per-task=10 --mem=32G --time=00:30:00 --pty bash
# Then manually: module load gcc cuda, activate phideus, test commands
```
Alternative with `salloc` (allocates node, you run commands on it):
```bash
salloc -p short --gres=gpu:1 --time=00:30:00
# Then: srun python -c "import torch; print(torch.cuda.is_available())"
```

### Option D: Nabucodonosor (no queue)
**When to use:** Queue is completely saturated, job is small enough for 1×A30,
or you need interactive GPU access with internet for debugging/installing.
**Specs:** 10 cores, 64GB RAM, 1×A30 24GB, NO SLURM, direct SSH access, has internet.
**Access:** `ssh mfmendez@nabucodonosor.ccad.unc.edu.ar`
**Requires:** Explicit access request to CCAD support (soporte@ccad.unc.edu.ar or
https://ccadunc.zulipchat.com/). May not be available if not previously authorized.
**Limitations:** Single GPU (no array jobs), shared with other ML users, no job scheduling.

### Option E: Optimize `--time` for Better Backfill
**When to use:** Job is in queue with (Priority) and waiting a long time.
SLURM's backfill scheduler can squeeze shorter jobs into gaps between running jobs.
A job requesting 8h can backfill into gaps that a 24h job cannot.
**How:** Check actual duration of similar completed jobs with `seff`, then set `--time`
to actual_duration × 1.3 (30% margin) instead of the partition maximum.
```bash
seff <similar-completed-jobid>    # Check actual wall-clock time
# Then adjust: --time=HH:MM:00 (tighter fit = better backfill chances)
```

### Option F: Post-Submit Verification (`scontrol`)
**When to use:** Always, right after submitting. Confirms SLURM parsed your
directives correctly (partition, time, memory, GPU, array indices).
```bash
scontrol show job <JOBID> | grep -E "Partition|TimeLimit|NumCPUs|Gres|ArrayTaskId|Command"
```
Catches silent misparses (e.g., a `#SBATCH` line was ignored because of a typo).

### Option G: Post-Completion Efficiency Audit (`seff`)
**When to use:** After a job completes, to tune future resource requests.
Over-requesting resources hurts both scheduling and cluster fairness.
```bash
seff <JOBID>
# Shows: CPU efficiency, memory efficiency, wall-clock vs requested time
# If CPU efficiency <50% or memory <30%: consider reducing requests
```

### Recommendation Logic

Present a specific recommendation based on context:

1. **Is this the first time running this type of job?** → Recommend B (preflight) or C (interactive)
2. **Did a similar job succeed recently?** → Recommend A (direct submit) + F (post-submit check)
3. **Is the queue saturated (all nodes alloc)?** → Recommend E (optimize --time) + mention D (Nabucodonosor)
4. **Is the user debugging a failure?** → Recommend C (interactive session)
5. **Is this a quick test (<1h)?** → Suggest submitting directly to `short` partition instead of `multi`

---

## MANDATORY INVOCATION RULE

This skill MUST be run before ANY `sbatch` command in this environment.
If Claude is about to run `sbatch`, it MUST run `/validate-sbatch` first.
No exceptions. The cost of validation (~30 seconds) is negligible compared to the cost of a failed job (hours or days of wasted queue time).
