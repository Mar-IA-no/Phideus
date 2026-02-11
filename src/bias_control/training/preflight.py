"""
MVP Anti-Variable-Fantasma: Training validation utilities for Bloque A.

Three components:
1. validate_training_setup(): 6-point preflight check before training starts.
2. DriftSentinel: Detects parameter drift after epoch 1 to catch frozen-by-accident.
3. validate_checkpoint_load(): Asserts that missing/unexpected keys match expected patterns.
"""

import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Expected trainable param count ranges by mode
PARAM_RANGES = {
    'run-a': (14_500_000, 15_600_000),    # MIDI + proj + adapters (~15.0M)
    'run-b': (39_000_000, 40_500_000),    # MIDI + proj + audio layers 2-3 (~39.7M)
    'run-c': (39_300_000, 40_700_000),    # MIDI + proj + audio layers 2-3 + adapters 0-1 (~40.0M)
}

# Module groups for drift reporting (reused from compare_layer_drift.py)
MODULE_GROUPS = {
    "audio_encoder.feature_extractor": "Audio CNN",
    "audio_encoder.pos_embedding":     "Audio PosEmb",
    "audio_encoder.transformer":       "Audio Transformer",
    "audio_projection":                "Audio Projection",
    "midi_encoder.event_embedding":    "MIDI Embedding",
    "midi_encoder.output_norm":        "MIDI OutputNorm",
    "midi_encoder.pos_encoding":       "MIDI PosEncoding",
    "midi_encoder.transformer":        "MIDI Transformer",
    "midi_projection":                 "MIDI Projection",
    "audio_adapters":                  "Audio Adapters",
}


def _classify_param(name: str) -> str:
    """Classify a parameter name into a module group."""
    for prefix, group in MODULE_GROUPS.items():
        if prefix in name:
            return group
    return "Other"


def validate_training_setup(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    mode: str,
    frozen_prefixes: List[str],
    trainable_prefixes: List[str],
) -> None:
    """
    6-point preflight validation before training starts.

    Checks:
    1. Module existence: All prefixes exist in the model.
    2. requires_grad match: Trainable modules have requires_grad=True,
       frozen modules have requires_grad=False.
    3. Optimizer coverage: Every requires_grad=True param is in optimizer.
    4. No frozen-in-optimizer: No requires_grad=False param is in optimizer.
    5. No duplicates: No param appears in more than one param group.
    6. Param count in expected range for mode.

    Args:
        model: The model to validate.
        optimizer: The optimizer to validate.
        mode: One of 'run-a', 'run-b', 'run-c'.
        frozen_prefixes: List of parameter name prefixes that MUST be frozen.
        trainable_prefixes: List of parameter name prefixes that MUST be trainable.

    Raises:
        RuntimeError: If any check fails.
    """
    errors = []

    # Collect all parameter names and their properties
    all_params = {name: param for name, param in model.named_parameters()}

    # --- Check 1: Module existence ---
    for prefix in frozen_prefixes + trainable_prefixes:
        found = any(name.startswith(prefix) for name in all_params)
        if not found:
            errors.append(f"[CHECK 1] Module prefix '{prefix}' not found in model")

    # --- Check 2: requires_grad match ---
    trainable_count = 0
    frozen_count = 0
    for name, param in all_params.items():
        for prefix in frozen_prefixes:
            if name.startswith(prefix) and param.requires_grad:
                errors.append(
                    f"[CHECK 2] Param '{name}' should be FROZEN but has requires_grad=True"
                )
        for prefix in trainable_prefixes:
            if name.startswith(prefix) and not param.requires_grad:
                errors.append(
                    f"[CHECK 2] Param '{name}' should be TRAINABLE but has requires_grad=False"
                )

        if param.requires_grad:
            trainable_count += param.numel()
        else:
            frozen_count += param.numel()

    # --- Check 3: Optimizer coverage ---
    optimizer_param_ids = set()
    for group in optimizer.param_groups:
        for p in group['params']:
            optimizer_param_ids.add(id(p))

    for name, param in all_params.items():
        if param.requires_grad and id(param) not in optimizer_param_ids:
            errors.append(
                f"[CHECK 3] Trainable param '{name}' NOT in optimizer"
            )

    # --- Check 4: No frozen-in-optimizer ---
    for name, param in all_params.items():
        if not param.requires_grad and id(param) in optimizer_param_ids:
            errors.append(
                f"[CHECK 4] Frozen param '{name}' found IN optimizer"
            )

    # --- Check 5: No duplicates ---
    seen_ids = {}
    for gi, group in enumerate(optimizer.param_groups):
        group_name = group.get('name', f'group_{gi}')
        for p in group['params']:
            pid = id(p)
            if pid in seen_ids:
                errors.append(
                    f"[CHECK 5] Param in group '{group_name}' also in '{seen_ids[pid]}'"
                )
            seen_ids[pid] = group_name

    # --- Check 6: Param count in expected range ---
    if mode in PARAM_RANGES:
        lo, hi = PARAM_RANGES[mode]
        if not (lo <= trainable_count <= hi):
            errors.append(
                f"[CHECK 6] Trainable params={trainable_count:,} outside expected "
                f"range [{lo:,}, {hi:,}] for mode={mode}"
            )

    # --- Log summary table ---
    group_stats = defaultdict(lambda: {'trainable': 0, 'frozen': 0})
    for name, param in all_params.items():
        group = _classify_param(name)
        if param.requires_grad:
            group_stats[group]['trainable'] += param.numel()
        else:
            group_stats[group]['frozen'] += param.numel()

    logger.info("=" * 70)
    logger.info("PREFLIGHT VALIDATION")
    logger.info("=" * 70)
    logger.info(f"{'Module Group':<25} {'Trainable':>12} {'Frozen':>12} {'Status':>10}")
    logger.info("-" * 70)
    for group in sorted(group_stats.keys()):
        stats = group_stats[group]
        status = "TRAIN" if stats['trainable'] > 0 else "FROZEN"
        if stats['trainable'] > 0 and stats['frozen'] > 0:
            status = "MIXED"
        logger.info(
            f"{group:<25} {stats['trainable']:>12,} {stats['frozen']:>12,} {status:>10}"
        )
    logger.info("-" * 70)
    logger.info(f"{'TOTAL':<25} {trainable_count:>12,} {frozen_count:>12,}")
    logger.info("=" * 70)

    # Log optimizer groups
    logger.info("Optimizer param groups:")
    for gi, group in enumerate(optimizer.param_groups):
        group_name = group.get('name', f'group_{gi}')
        n_params = sum(p.numel() for p in group['params'])
        logger.info(f"  [{gi}] {group_name}: {n_params:,} params, lr={group['lr']:.1e}")

    if errors:
        for err in errors:
            logger.error(err)
        raise RuntimeError(
            f"Preflight validation FAILED with {len(errors)} errors:\n"
            + "\n".join(errors)
        )

    logger.info("Preflight validation PASSED")


class DriftSentinel:
    """
    Detects parameter drift after training to catch frozen-by-accident bugs.

    Takes a snapshot of all trainable parameters at init time, then compares
    after epoch 1 to verify that training actually modified them.
    """

    def __init__(self, model: nn.Module):
        """Snapshot all trainable parameters."""
        self.initial_state = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.initial_state[name] = param.detach().cpu().clone()

        logger.info(f"DriftSentinel: Snapshotted {len(self.initial_state)} trainable params")

    def check(self, model: nn.Module) -> Dict[str, float]:
        """
        Compare current state to initial snapshot.

        Returns dict of {module_group: relative_change}.
        Logs WARNING if any trainable module < 1e-6 relative change.
        Raises RuntimeError if ALL trainable modules < 1e-6 (total training failure).
        """
        group_drift = defaultdict(lambda: {'l2_sum': 0.0, 'norm_sum': 0.0, 'n_params': 0})

        for name, initial_param in self.initial_state.items():
            # Find current param
            current_param = None
            for n, p in model.named_parameters():
                if n == name:
                    current_param = p
                    break

            if current_param is None:
                logger.warning(f"DriftSentinel: param '{name}' disappeared from model")
                continue

            current = current_param.detach().cpu()
            diff = (current - initial_param).float()
            l2 = diff.norm().item()
            base_norm = initial_param.float().norm().item()

            group = _classify_param(name)
            group_drift[group]['l2_sum'] += l2
            group_drift[group]['norm_sum'] += base_norm
            group_drift[group]['n_params'] += current.numel()

        # Compute relative change per group
        results = {}
        all_zero = True
        for group, stats in sorted(group_drift.items()):
            rel_change = stats['l2_sum'] / (stats['norm_sum'] + 1e-12)
            results[group] = rel_change
            if rel_change >= 1e-6:
                all_zero = False

        # Log drift report
        logger.info("=" * 60)
        logger.info("DRIFT SENTINEL REPORT")
        logger.info("=" * 60)
        logger.info(f"{'Module Group':<25} {'Rel. Change':>15} {'Status':>10}")
        logger.info("-" * 60)
        for group, rel_change in sorted(results.items()):
            status = "OK" if rel_change >= 1e-6 else "WARNING"
            logger.info(f"{group:<25} {rel_change:>15.6e} {status:>10}")
        logger.info("=" * 60)

        # Check for warnings
        zero_groups = [g for g, rc in results.items() if rc < 1e-6]
        if zero_groups:
            logger.warning(
                f"DriftSentinel: {len(zero_groups)} trainable groups with near-zero drift: "
                f"{zero_groups}"
            )

        if all_zero:
            raise RuntimeError(
                "DriftSentinel ABORT: ALL trainable modules have near-zero drift. "
                "Training is not updating parameters."
            )

        return results


def validate_checkpoint_load(
    missing_keys: List[str],
    unexpected_keys: List[str],
    expected_missing: Optional[List[str]] = None,
    expected_unexpected: Optional[List[str]] = None,
) -> None:
    """
    Assert that missing/unexpected keys from load_state_dict(strict=False)
    match expected patterns.

    Patterns support regex via re.fullmatch:
        'dann.*'           -> matches 'dann.weight', 'dann.bias', etc.
        'audio_adapters.*' -> matches any adapter key
        'exact_key_name'   -> matches only that exact key

    Args:
        missing_keys: Keys in the model but not in checkpoint.
        unexpected_keys: Keys in checkpoint but not in model.
        expected_missing: Regex patterns for acceptable missing keys.
        expected_unexpected: Regex patterns for acceptable unexpected keys.

    Raises:
        RuntimeError: If any key doesn't match expected patterns.
    """
    expected_missing = expected_missing or []
    expected_unexpected = expected_unexpected or []

    def matches_any_pattern(key: str, patterns: List[str]) -> bool:
        return any(re.fullmatch(p, key) for p in patterns)

    unexpected_surprise = [
        k for k in unexpected_keys if not matches_any_pattern(k, expected_unexpected)
    ]
    missing_surprise = [
        k for k in missing_keys if not matches_any_pattern(k, expected_missing)
    ]

    # Log expected keys
    if missing_keys:
        expected_m = [k for k in missing_keys if k not in missing_surprise]
        if expected_m:
            logger.info(f"Missing keys (expected, {len(expected_m)}): {expected_m[:5]}...")
    if unexpected_keys:
        expected_u = [k for k in unexpected_keys if k not in unexpected_surprise]
        if expected_u:
            logger.info(f"Unexpected keys (expected, {len(expected_u)}): {expected_u[:5]}...")

    if unexpected_surprise or missing_surprise:
        raise RuntimeError(
            f"Checkpoint load mismatch:\n"
            f"  Unexpected surprise keys ({len(unexpected_surprise)}): {unexpected_surprise[:10]}\n"
            f"  Missing surprise keys ({len(missing_surprise)}): {missing_surprise[:10]}"
        )

    logger.info("Checkpoint load validation PASSED")
