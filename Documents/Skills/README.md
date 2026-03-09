# Shared Claude Code Skills — Phideus Project

Reusable skills for [Claude Code](https://claude.com/claude-code), built from real operational experience on HPC clusters and ML research workflows.

Each skill lives in its own directory as a `SKILL.md` file. Install by copying it from this repository into your local Claude Code skills directory, then invoke with `/<skill-name>` in Claude Code.

## What We Share Here

`Documents/Skills/` is the public/shared layer of operational know-how that Phideus exports from real work.

- Shared/public skills: reusable outside this repo, mainly HPC/SLURM operations.
- Source of truth for new HPC lessons: [BITACORA_UNC.md](../BITACORA_UNC.md), which stays as the operational log; only stable, reusable lessons get promoted from there into shared skills.

## Available Skills

### `/validate-sbatch` — SLURM Script Validator

**Purpose**: Validate SLURM sbatch scripts before submission. Catches errors that would waste hours of queue time.

**What it does**: Runs 5 sequential validation phases on any sbatch script:

1. **Static analysis** — shebang, shell safety flags, SLURM directives, syntax check (`bash -n`)
2. **Path verification** — resolves all variables, checks files/dirs exist on disk
3. **Dependency verification** — pip packages, conda envs, CUDA modules
4. **SLURM dry run** — `sbatch --test-only` + queue position estimate
5. **Final report** — BLOCKER / WARNING / OK for each item, overall VERDICT

After validation, presents 7 strategic options (direct submit, preflight on short partition, interactive debug, etc.).

**Latest Mendieta lessons included**:
- catches `--gres`/`set -u`/`--mem=0` class blockers,
- adds runtime checks for Transkun/Gate 6 paths with mixed sample rates,
- verifies the `torch.utils.checkpoint` input-grad fix when that training path needs it.

**Built for**: CCAD Mendieta (Universidad Nacional de Cordoba), but adaptable to any SLURM cluster by editing partition names and paths.

**Invoke**: `/validate-sbatch <script-path>`

---

### `/slurm-handbook` — SLURM Operations Handbook

**Purpose**: Comprehensive reference for operating ML training jobs on SLURM clusters. Interactive menu with 14 sections.

**Sections**:

1. HPC Architecture — login vs compute nodes, hardware specs, partitions
2. sbatch Script Template — complete template with mandatory setup order
3. Known Traps — 12+ real errors and how to avoid them
4. Data Staging — rsync vs cp, verification, scratch storage
5. Checkpoint & Resume — bash patterns, SIGTERM handlers, auto-resubmit
6. Array Jobs — 1D, 2D cartesian product, logging, individual requeue
7. Memory — sizing by job type, page cache vs OOM, profiling
8. Scheduling — backfill strategy, `--time` tuning, `scontrol update`
9. Monitoring — squeue formats, scontrol, seff, interpreting PENDING reasons
10. Debugging — preflight strategy on short partition, interactive sessions
11. CCAD Wiki Links — organized references for infrastructure and tutorials
12. Pre-submit Checklist — 17 quick verification items
13. Templates — 4 ready-to-copy scripts (simple, long with resume, array, preflight)
14. Quick Reference — most-used commands in a single table

**Latest Mendieta lessons included**:
- mixed sample rates in MAESTRO v3 and their impact on `torch.stack`,
- `torch.utils.checkpoint` failures when no input reaches backward with grad,
- stronger preflight guidance: dataloader + collate + first backward, not just imports.

**Built for**: CCAD Mendieta, but most sections are cluster-agnostic.

**Invoke**: `/slurm-handbook` or `/slurm-handbook <topic>`

---

## Shared Skills
- `/validate-sbatch`
- `/slurm-handbook`

These are the skills we currently treat as exportable and worth sharing outside the immediate repo workflow.

## Update Policy

When UNC/Mendieta produces a new operational lesson, promote it to a shared skill only if it is:

1. recurrent or costly enough to justify codification,
2. reusable beyond a single experiment,
3. stable enough to be stated as procedure and not just anecdote.

Examples already promoted from UNC:

- avoid `set -u` in Mendieta sbatch scripts,
- avoid `--mem=0` for normal jobs,
- use `--gres=gpu:N`,
- validate `stderr` logging and `PYTHONUNBUFFERED=1`,
- handle mixed sample rates before `torch.stack`,
- check `torch.utils.checkpoint` input-grad requirements in Transkun-like paths.

---

## Installation

```bash
# Copy a skill from this repo to your local Claude Code skills directory
mkdir -p "$HOME/.claude/skills/validate-sbatch"
cp Documents/Skills/validate-sbatch/SKILL.md "$HOME/.claude/skills/validate-sbatch/"

mkdir -p "$HOME/.claude/skills/slurm-handbook"
cp Documents/Skills/slurm-handbook/SKILL.md "$HOME/.claude/skills/slurm-handbook/"
```

## Contributing

Skills are written as Markdown files following the [Claude Code skill format](https://docs.anthropic.com/en/docs/claude-code). Each skill has a YAML frontmatter header with `name`, `description`, and optionally `argument-hint` and `allowed-tools`.

All skills in this collection were built from real operational experience — every trap documented was hit at least once, every pattern was validated in production.
