#!/usr/bin/env python3
"""Unprivileged analytical worker for the prospective Wave 57 phases."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

import _wave56_phase_worker as base
from geometria_proporcional.wave54_joint_set import target_set_indices
from geometria_proporcional.wave55_policy_bridge import action_metric_arrays
from geometria_proporcional.wave57_tail_guard import (
    HARM_MODEL_CONTRACT,
    apply_harm_guard,
    conditional_harm_shuffle,
    fit_harm_logistic,
    harm_labels,
    per_set_support,
    proposal_mask,
    score_harm_logistic,
    token_support,
)


HARD_ONLY = "hard_only"
WORKER_SOURCE = "experiments/geometria_proporcional/_wave57_phase_worker.py"


def _counts(data: dict[str, np.ndarray]) -> dict[str, int]:
    primary = np.asarray(data["primary"], dtype=bool)
    disagreement = np.asarray(data["disagreement"], dtype=bool) & primary[:, None]
    labels = harm_labels(data["gain"])
    return {
        "tokens": int(primary.sum()),
        "disagreement_rows": int(disagreement.sum()),
        "disagreement_tokens": int(disagreement.any(axis=1).sum()),
        "harm_tokens": int((disagreement & (labels == 1)).any(axis=1).sum()),
        "nonharm_tokens": int((disagreement & (labels == 0)).any(axis=1).sum()),
    }


def _global_failure(
    data: dict[str, np.ndarray], config: dict[str, Any], phase: str
) -> dict[str, Any] | None:
    counts = _counts(data)
    prefix = base.PHASE_TO_ROLE[phase]
    names = ["tokens", "disagreement_rows", "disagreement_tokens"]
    if phase == "fit":
        names.extend(("harm_tokens", "nonharm_tokens"))
    required = {
        name: int(config["minimums"][f"{prefix}_{name}"]) for name in names
    }
    failed = [name for name in names if counts[name] < required[name]]
    if not failed:
        return None
    return {
        "status": "NOT_EVALUABLE",
        "phase": phase,
        "counts": counts,
        "required": required,
        "failed": failed,
        "reason": "frozen token/row minimums were not met; redraw is forbidden",
        "split_disjointness": base._split_evidence(data),
    }


def _score_harm(state: dict[str, Any], data: dict[str, np.ndarray]) -> np.ndarray:
    scores = np.full(data["disagreement"].shape, np.nan, dtype=np.float64)
    rows = np.asarray(data["disagreement"], dtype=bool)
    scores[rows] = score_harm_logistic(state, data["design"][rows])
    return scores


def _harm_model_arrays(prefix: str, state: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        f"{prefix}__mean": np.asarray(state["mean"], dtype=np.float64),
        f"{prefix}__scale": np.asarray(state["scale"], dtype=np.float64),
        f"{prefix}__coef": np.asarray(state["coef"], dtype=np.float64),
        f"{prefix}__intercept": np.asarray(state["intercept"], dtype=np.float64),
        f"{prefix}__classes": np.asarray(state["classes"], dtype=np.int64),
        f"{prefix}__n_iter": np.asarray(state["n_iter"], dtype=np.int64),
    }


def _data_arrays(role: str, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    arrays = base.data_arrays(role, data)
    arrays[f"{role}__harm_label"] = harm_labels(data["gain"])
    arrays[f"{role}__set_index"] = target_set_indices(data["target"])
    return arrays


def run_fit(
    stage: Path,
    output: Path,
    config: dict[str, Any],
    utilities: np.ndarray,
    data: dict[str, np.ndarray],
) -> str:
    failure = _global_failure(data, config, "fit")
    if failure:
        base.write_json_atomic(output / "fit_not_evaluable.json", failure)
        return "FIT_NOT_EVALUABLE"
    rows = np.asarray(data["primary"], dtype=bool)[:, None] & data["disagreement"]
    labels = harm_labels(data["gain"])
    proposer = base.fit_ridge(data, 1.0, "all", data["gain"])
    advantage = base.fit_ridge(data, 100.0, ["advantage"], data["gain"])
    try:
        harm = fit_harm_logistic(
            data["design"][rows], labels[rows], data["weights"][rows], HARM_MODEL_CONTRACT
        )
    except (RuntimeError, FloatingPointError) as error:
        base.write_json_atomic(
            output / "fit_not_evaluable.json",
            {
                "status": "NOT_EVALUABLE",
                "phase": "fit",
                "reason": "predeclared_harm_model_fit_failure",
                "error_type": type(error).__name__,
                "message": str(error),
                "counts": _counts(data),
                "split_disjointness": base._split_evidence(data),
            },
        )
        return "FIT_NOT_EVALUABLE"

    arrays = _data_arrays("gate_fit", data)
    arrays.update(base.model_arrays("model__mean_proposer", proposer))
    arrays.update(base.model_arrays("model__advantage_only", advantage))
    arrays.update(_harm_model_arrays("model__harm_guard", harm))
    arrays["harm_label"] = labels
    shams: list[dict[str, Any]] = []
    mappings = []
    targets = []
    minimums = config["minimums"]
    for shuffle_id, seed in enumerate(config["shuffle_seeds"]):
        shuffled = conditional_harm_shuffle(
            labels[data["primary"]],
            data["disagreement"][data["primary"]],
            data["weights"][data["primary"]],
            int(seed),
        )
        target = labels.copy()
        target[data["primary"]] = shuffled["target"]
        diagnostic = shuffled["diagnostics"]
        evaluable = (
            float(diagnostic["permutable_fraction"] or 0.0)
            >= float(minimums["shuffle_permutable_fraction"])
            and float(diagnostic["hamming_global"] or 0.0)
            >= float(minimums["shuffle_hamming_global"])
            and float(diagnostic["hamming_weighted"] or 0.0)
            >= float(minimums["shuffle_hamming_weighted"])
        )
        try:
            model = fit_harm_logistic(
                data["design"][rows], target[rows], data["weights"][rows], HARM_MODEL_CONTRACT
            )
        except (RuntimeError, FloatingPointError) as error:
            model = None
            diagnostic = {**diagnostic, "fit_error": f"{type(error).__name__}: {error}"}
            evaluable = False
        status = "PASS" if evaluable else "NOT_EVALUABLE"
        shams.append(
            {
                "shuffle_id": shuffle_id,
                "seed": int(seed),
                "status": status,
                "diagnostics": diagnostic,
                "model": model,
            }
        )
        mappings.append(shuffled["mapping"])
        targets.append(target)
        if model is not None:
            arrays.update(_harm_model_arrays(f"model__sham__{shuffle_id}", model))
    arrays["sham_mapping"] = np.stack(mappings)
    arrays["sham_target"] = np.stack(targets)
    arrays["sham_id"] = np.arange(len(shams), dtype=np.int64)
    base.write_npz_atomic(output / "gate_fit_bundle.npz", {
        key: value for key, value in data.items() if isinstance(value, np.ndarray)
    })
    base.write_npz_atomic(output / "fit_arrays.npz", arrays)
    base.write_json_atomic(
        output / "feature_schema.json", {"features": base.retrospective.feature_schema()}
    )
    core = {
        "status": "FIT_COMPLETE",
        "counts": _counts(data),
        "models": {
            "mean_proposer": {"kind": "ridge", "alpha": 1.0, "model": proposer},
            "advantage_only": {"kind": "ridge", "alpha": 100.0, "model": advantage},
            "harm_guard": harm,
        },
        "shams": shams,
        "fit_population": "NEAR_RIVAL and cardinality >=2; disagreement rows; token weight 1/d_t",
        "split_disjointness": base._split_evidence(data),
    }
    base.write_json_atomic(output / "fit_core.json", core)
    base._write_freeze(
        output,
        "fit_freeze.json",
        "fit-complete-before-gate-select-label-access",
        ("fit_core.json", "fit_arrays.npz", "feature_schema.json", "gate_fit_bundle.npz"),
        {
            "harm_model_contract": HARM_MODEL_CONTRACT,
            "shuffle_seeds": config["shuffle_seeds"],
            "split_disjointness": base._split_evidence(data),
            "provenance": base.load_json(stage / "phase_request.json"),
        },
    )
    return "FIT_COMPLETE"


def _metrics(
    actions: np.ndarray, data: dict[str, np.ndarray], utilities: np.ndarray, config: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    values = action_metric_arrays(
        actions, data["target"], utilities, float(config["incompatible_regret_penalty"])
    )
    return values, base.retrospective.summarize(values, data["_selection_mask"])


def _choose_with_atol(rows: list[dict[str, Any]], atol: float, key) -> dict[str, Any]:
    feasible = [row for row in rows if row["evaluable"] and row["feasible"]]
    if not feasible:
        raise RuntimeError("selector has no feasible hard-only terminal")
    best = min(float(row["regret"]) for row in feasible)
    tied = [row for row in feasible if float(row["regret"]) <= best + atol]
    return min(tied, key=key)


def select_proposer(
    scores: np.ndarray,
    data: dict[str, np.ndarray],
    utilities: np.ndarray,
    config: dict[str, Any],
    token_mask: np.ndarray,
    minimum_tokens: int,
) -> dict[str, Any]:
    token_mask = np.asarray(token_mask, dtype=bool)
    data["_selection_mask"] = token_mask
    hard_metrics, hard_summary = _metrics(data["hard_actions"], data, utilities, config)
    del hard_metrics
    active = token_mask[:, None] & data["disagreement"]
    values = scores[active]
    if len(values) == 0 or np.any(~np.isfinite(values)):
        raise RuntimeError("proposer selector has no finite active scores")
    rows = []
    for cell, q in enumerate(config["proposer_quantiles"]):
        threshold = float(np.quantile(values, float(q), method="higher"))
        proposed = proposal_mask(scores, data["disagreement"], threshold)
        actions = np.where(proposed, data["posterior_actions"], data["hard_actions"])
        _, summary = _metrics(actions, data, utilities, config)
        support = token_support(proposed, token_mask)
        evaluable = support["tokens"] >= int(minimum_tokens)
        feasible = evaluable and (
            summary["accuracy"] >= hard_summary["accuracy"] - float(config["selection"]["accuracy_noninferiority_margin"])
            and summary["compatible"] >= hard_summary["compatible"] - float(config["selection"]["compatible_noninferiority_margin"])
        )
        rows.append({
            "cell_index": cell,
            "q": float(q),
            "threshold": threshold,
            "evaluable": evaluable,
            "feasible": bool(feasible),
            "support": support,
            **summary,
        })
    rows.append({
        "cell_index": len(rows), "q": HARD_ONLY, "threshold": HARD_ONLY,
        "evaluable": True, "feasible": True, "support": {"rows": 0, "tokens": 0},
        **hard_summary,
    })

    def tie(row: dict[str, Any]) -> tuple[Any, ...]:
        if row["threshold"] == HARD_ONLY:
            threshold, quantile = float("inf"), float("inf")
        else:
            threshold, quantile = float(row["threshold"]), float(row["q"])
        return (
            int(row["support"]["rows"]), int(row["support"]["tokens"]),
            -threshold, -quantile, int(row["cell_index"]),
        )

    selected = _choose_with_atol(rows, float(config["selection"]["tie_atol"]), tie)
    return {"selected": selected, "grid": rows}


def select_guard(
    probabilities: np.ndarray,
    proposals: np.ndarray,
    data: dict[str, np.ndarray],
    utilities: np.ndarray,
    config: dict[str, Any],
    token_mask: np.ndarray,
    minimum_tokens: int,
) -> dict[str, Any]:
    token_mask = np.asarray(token_mask, dtype=bool)
    data["_selection_mask"] = token_mask
    hard_metrics, hard_summary = _metrics(data["hard_actions"], data, utilities, config)
    del hard_metrics
    eligible = token_mask[:, None] & proposals
    values = probabilities[eligible]
    if len(values) == 0 or np.any(~np.isfinite(values)):
        raise RuntimeError("guard selector has no finite proposed probabilities")
    rows = []
    for cell, q in enumerate(config["guard_acceptance_quantiles"]):
        threshold = float(np.quantile(values, float(q), method="higher"))
        guarded = apply_harm_guard(
            proposals, probabilities, data["hard_actions"], data["posterior_actions"], threshold
        )
        _, summary = _metrics(guarded["actions"], data, utilities, config)
        support = token_support(guarded["authorized"], token_mask)
        evaluable = support["tokens"] >= int(minimum_tokens)
        feasible = evaluable and (
            summary["accuracy"] >= hard_summary["accuracy"] - float(config["selection"]["accuracy_noninferiority_margin"])
            and summary["compatible"] >= hard_summary["compatible"] - float(config["selection"]["compatible_noninferiority_margin"])
            and summary["worst_regret"] <= hard_summary["worst_regret"] + float(config["selection"]["worst_regret_noninferiority_margin"])
        )
        rows.append({
            "cell_index": cell, "q": float(q), "threshold": threshold,
            "evaluable": evaluable, "feasible": bool(feasible), "support": support,
            **summary,
        })
    rows.append({
        "cell_index": len(rows), "q": HARD_ONLY, "threshold": HARD_ONLY,
        "evaluable": True, "feasible": True, "support": {"rows": 0, "tokens": 0},
        **hard_summary,
    })

    def tie(row: dict[str, Any]) -> tuple[Any, ...]:
        if row["threshold"] == HARD_ONLY:
            threshold, quantile = float("inf"), float("inf")
        else:
            threshold, quantile = float(row["threshold"]), float(row["q"])
        return (
            int(row["support"]["rows"]), int(row["support"]["tokens"]),
            threshold, quantile, int(row["cell_index"]),
        )

    selected = _choose_with_atol(rows, float(config["selection"]["tie_atol"]), tie)
    return {"selected": selected, "grid": rows}


def _selection_sequence(
    mean_scores: np.ndarray,
    harm_scores: np.ndarray,
    sham_scores: list[np.ndarray | None],
    sham_statuses: list[str],
    data: dict[str, np.ndarray],
    utilities: np.ndarray,
    config: dict[str, Any],
    token_mask: np.ndarray,
    proposal_minimum: int,
    authorized_minimum: int,
) -> dict[str, Any]:
    proposer = select_proposer(
        mean_scores, data, utilities, config, token_mask, proposal_minimum
    )
    selected = proposer["selected"]
    if selected["threshold"] == HARD_ONLY or selected["support"]["tokens"] < proposal_minimum:
        return {"status": "NOT_EVALUABLE", "reason": "proposer_identity_or_low_support", "proposer": proposer}
    proposals = proposal_mask(mean_scores, data["disagreement"], selected["threshold"])
    guard = select_guard(
        harm_scores, proposals, data, utilities, config, token_mask, authorized_minimum
    )
    shams = []
    for shuffle_id, (scores, status) in enumerate(zip(sham_scores, sham_statuses, strict=True)):
        if status != "PASS" or scores is None:
            shams.append({"shuffle_id": shuffle_id, "status": "NOT_EVALUABLE", "selection": None})
        else:
            shams.append({
                "shuffle_id": shuffle_id,
                "status": "PASS",
                "selection": select_guard(
                    scores, proposals, data, utilities, config, token_mask, authorized_minimum
                ),
            })
    return {"status": "PASS", "proposer": proposer, "guard": guard, "shams": shams}


def _shard_assignment(tokens: np.ndarray) -> np.ndarray:
    import hashlib

    return np.asarray([
        hashlib.sha256((str(token) + "wave57-shard").encode()).digest()[-1] & 1
        for token in tokens.astype(str)
    ], dtype=np.int8)


def run_select(
    stage: Path,
    output: Path,
    config: dict[str, Any],
    utilities: np.ndarray,
    data: dict[str, np.ndarray],
) -> str:
    failure = _global_failure(data, config, "select")
    if failure:
        base.write_json_atomic(output / "selection_not_evaluable.json", failure)
        return "SELECT_NOT_EVALUABLE"
    fit_core = base.load_json(stage / "previous/fit_core.json")
    models = fit_core["models"]
    mean_scores = base.score_model(models["mean_proposer"]["model"], data, "all")
    advantage_scores = base.score_model(models["advantage_only"]["model"], data, ["advantage"])
    harm_scores = _score_harm(models["harm_guard"], data)
    sham_scores = [
        _score_harm(row["model"], data) if row["status"] == "PASS" and row["model"] else None
        for row in fit_core["shams"]
    ]
    statuses = [str(row["status"]) for row in fit_core["shams"]]
    primary = np.asarray(data["primary"], dtype=bool)
    sequence = _selection_sequence(
        mean_scores, harm_scores, sham_scores, statuses, data, utilities, config, primary,
        int(config["minimums"]["gate_select_proposal_tokens"]),
        int(config["minimums"]["gate_select_authorized_tokens"]),
    )
    if sequence["status"] != "PASS":
        base.write_json_atomic(
            output / "selection_not_evaluable.json",
            {**sequence, "phase": "select", "counts": _counts(data), "split_disjointness": base._split_evidence(data)},
        )
        return "SELECT_NOT_EVALUABLE"
    advantage = select_proposer(
        advantage_scores, data, utilities, config, primary,
        int(config["minimums"]["gate_select_proposal_tokens"]),
    )
    assignment = _shard_assignment(data["pair_token"])
    shards = {}
    for shard in (0, 1):
        local = primary & (assignment == shard)
        local_counts = {
            "tokens": int(local.sum()),
            "disagreement_rows": int((local[:, None] & data["disagreement"]).sum()),
            "disagreement_tokens": int((local & data["disagreement"].any(axis=1)).sum()),
        }
        required = {
            "tokens": int(config["minimums"]["gate_select_shard_tokens"]),
            "disagreement_rows": int(config["minimums"]["gate_select_shard_disagreement_rows"]),
            "disagreement_tokens": int(config["minimums"]["gate_select_shard_disagreement_tokens"]),
        }
        if any(local_counts[key] < required[key] for key in required):
            shards[str(shard)] = {"status": "NOT_EVALUABLE", "counts": local_counts, "required": required}
        else:
            shards[str(shard)] = _selection_sequence(
                mean_scores, harm_scores, sham_scores, statuses, data, utilities, config, local,
                int(config["minimums"]["gate_select_shard_proposal_tokens"]),
                int(config["minimums"]["gate_select_shard_authorized_tokens"]),
            )

    arrays = _data_arrays("gate_select", data)
    arrays["score__mean_proposer"] = mean_scores
    arrays["score__harm_guard"] = harm_scores
    arrays["score__advantage_only"] = advantage_scores
    arrays["shard_assignment"] = assignment
    arrays["score__sham"] = np.stack([
        scores if scores is not None else np.full(mean_scores.shape, np.nan) for scores in sham_scores
    ])
    proposer_selected = sequence["proposer"]["selected"]
    proposals = proposal_mask(
        mean_scores, data["disagreement"], proposer_selected["threshold"]
    )
    proposer_actions = np.where(
        proposals, data["posterior_actions"], data["hard_actions"]
    )
    arrays["selection__proposer__proposals"] = proposals
    arrays["selection__proposer__actions"] = proposer_actions
    threshold, hard_only = base._threshold_arrays(proposer_selected["threshold"])
    arrays["selection__proposer__threshold"] = threshold
    arrays["selection__proposer__hard_only"] = hard_only
    for metric, values in action_metric_arrays(
        proposer_actions,
        data["target"],
        utilities,
        float(config["incompatible_regret_penalty"]),
    ).items():
        arrays[f"selection__proposer__metric__{metric}"] = values

    guard_selected = sequence["guard"]["selected"]
    guarded = apply_harm_guard(
        proposals,
        harm_scores,
        data["hard_actions"],
        data["posterior_actions"],
        guard_selected["threshold"],
    )
    arrays["selection__guard__authorized"] = guarded["authorized"]
    arrays["selection__guard__actions"] = guarded["actions"]
    threshold, hard_only = base._threshold_arrays(guard_selected["threshold"])
    arrays["selection__guard__threshold"] = threshold
    arrays["selection__guard__hard_only"] = hard_only
    for metric, values in action_metric_arrays(
        guarded["actions"],
        data["target"],
        utilities,
        float(config["incompatible_regret_penalty"]),
    ).items():
        arrays[f"selection__guard__metric__{metric}"] = values

    sham_authorized = []
    sham_actions = []
    sham_thresholds = []
    sham_hard_only = []
    sham_metrics: dict[str, list[np.ndarray]] = {}
    for row, scores in zip(sequence["shams"], sham_scores, strict=True):
        if row["status"] == "PASS" and scores is not None:
            chosen = row["selection"]["selected"]
            result = apply_harm_guard(
                proposals,
                scores,
                data["hard_actions"],
                data["posterior_actions"],
                chosen["threshold"],
            )
            local_threshold, local_hard = base._threshold_arrays(chosen["threshold"])
        else:
            result = apply_harm_guard(
                proposals,
                harm_scores,
                data["hard_actions"],
                data["posterior_actions"],
                HARD_ONLY,
            )
            local_threshold, local_hard = base._threshold_arrays(HARD_ONLY)
        sham_authorized.append(result["authorized"])
        sham_actions.append(result["actions"])
        sham_thresholds.append(local_threshold)
        sham_hard_only.append(local_hard)
        for metric, values in action_metric_arrays(
            result["actions"],
            data["target"],
            utilities,
            float(config["incompatible_regret_penalty"]),
        ).items():
            sham_metrics.setdefault(metric, []).append(values)
    arrays["selection__sham__authorized"] = np.stack(sham_authorized)
    arrays["selection__sham__actions"] = np.stack(sham_actions)
    arrays["selection__sham__threshold"] = np.stack(sham_thresholds)
    arrays["selection__sham__hard_only"] = np.stack(sham_hard_only)
    for metric, values in sham_metrics.items():
        arrays[f"selection__sham__metric__{metric}"] = np.stack(values)

    advantage_selected = advantage["selected"]
    advantage_proposals = proposal_mask(
        advantage_scores, data["disagreement"], advantage_selected["threshold"]
    )
    advantage_actions = np.where(
        advantage_proposals, data["posterior_actions"], data["hard_actions"]
    )
    arrays["selection__advantage__proposals"] = advantage_proposals
    arrays["selection__advantage__actions"] = advantage_actions
    base.write_npz_atomic(output / "gate_select_bundle.npz", {
        key: value for key, value in data.items() if isinstance(value, np.ndarray)
    })
    base.write_npz_atomic(output / "selection_arrays.npz", arrays)
    core = {
        "status": "SELECT_COMPLETE",
        "counts": _counts(data),
        "sequence": sequence,
        "advantage_only": advantage,
        "shards": shards,
        "split_disjointness": base._split_evidence(data),
    }
    base.write_json_atomic(output / "selection_core.json", core)
    freeze = {
        "proposer": sequence["proposer"]["selected"],
        "guard": sequence["guard"]["selected"],
        "shams": [
            {
                "shuffle_id": row["shuffle_id"],
                "status": row["status"],
                "selected": row["selection"]["selected"] if row["selection"] else None,
            }
            for row in sequence["shams"]
        ],
        "advantage_only": advantage["selected"],
        "shards": shards,
    }
    base._write_freeze(
        output,
        "selection_freeze.json",
        "selection-complete-before-sealed-monitor-label-access",
        ("selection_core.json", "selection_arrays.npz", "gate_select_bundle.npz"),
        {"selected": freeze, "split_disjointness": base._split_evidence(data), "provenance": base.load_json(stage / "phase_request.json")},
    )
    return "SELECT_COMPLETE"


def _arm(actions, authorized, data, utilities, config):
    return base._arm(actions, authorized, data, utilities, config)


def _ci(left, right, metric, indices, bootstrap):
    return base.retrospective.paired_delta_ci(
        left["metrics"][metric][indices], right["metrics"][metric][indices], bootstrap
    )


def _calibration(probability: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    p = probability[mask]
    y = labels[mask].astype(np.float64)
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    order = np.argsort(p, kind="stable")
    bins = []
    for index, ids in enumerate(np.array_split(order, 10)):
        if len(ids):
            bins.append({"decile": index, "rows": len(ids), "mean_probability": float(p[ids].mean()), "harm_rate": float(y[ids].mean())})
    return {
        "rows": len(p),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(np.mean(-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))),
        "deciles": bins,
    }


def run_adjudicate(stage: Path, output: Path, config: dict[str, Any], utilities: np.ndarray, data: dict[str, np.ndarray]) -> str:
    failure = _global_failure(data, config, "adjudicate")
    if failure:
        base.write_json_atomic(output / "monitor_not_evaluable.json", failure)
        return "MONITOR_NOT_EVALUABLE"
    fit_core = base.load_json(stage / "previous/fit_core.json")
    select_core = base.load_json(stage / "previous/selection_core.json")
    selected = base.load_json(stage / "previous/selection_freeze.json")["selected"]
    models = fit_core["models"]
    mean_scores = base.score_model(models["mean_proposer"]["model"], data, "all")
    advantage_scores = base.score_model(models["advantage_only"]["model"], data, ["advantage"])
    harm_scores = _score_harm(models["harm_guard"], data)
    proposals = proposal_mask(mean_scores, data["disagreement"], selected["proposer"]["threshold"])
    guarded = apply_harm_guard(proposals, harm_scores, data["hard_actions"], data["posterior_actions"], selected["guard"]["threshold"])
    zero = np.zeros(data["hard_actions"].shape, dtype=bool)
    hard = _arm(data["hard_actions"], zero, data, utilities, config)
    pure = _arm(data["posterior_actions"], data["disagreement"], data, utilities, config)
    proposer = _arm(np.where(proposals, data["posterior_actions"], data["hard_actions"]), proposals, data, utilities, config)
    main = _arm(guarded["actions"], guarded["authorized"], data, utilities, config)
    advantage_mask = proposal_mask(advantage_scores, data["disagreement"], selected["advantage_only"]["threshold"])
    advantage = _arm(np.where(advantage_mask, data["posterior_actions"], data["hard_actions"]), advantage_mask, data, utilities, config)
    oracle_mask = data["disagreement"] & (data["gain"] > 1e-12)
    oracle = _arm(np.where(oracle_mask, data["posterior_actions"], data["hard_actions"]), oracle_mask, data, utilities, config)
    arms = {
        "hard_set_policy": hard,
        "pure_joint_full": pure,
        "mean_proposer_only": proposer,
        "mean_plus_harm_guard": main,
        "advantage_only_value_gate": advantage,
        "oracle_positive_gain": oracle,
    }

    sham_arms = []
    sham_statuses = []
    sham_scores = []
    for fit_row, selected_row in zip(fit_core["shams"], selected["shams"], strict=True):
        status = "PASS" if fit_row["status"] == "PASS" and selected_row["status"] == "PASS" else "NOT_EVALUABLE"
        sham_statuses.append(status)
        if status == "PASS":
            scores = _score_harm(fit_row["model"], data)
            result = apply_harm_guard(proposals, scores, data["hard_actions"], data["posterior_actions"], selected_row["selected"]["threshold"])
            arm = _arm(result["actions"], result["authorized"], data, utilities, config)
        else:
            scores = np.full(mean_scores.shape, np.nan)
            arm = hard
        sham_scores.append(scores)
        sham_arms.append(arm)
    average_metrics = {
        metric: np.mean(np.stack([arm["metrics"][metric] for arm in sham_arms]), axis=0)
        for metric in sham_arms[0]["metrics"]
    }
    sham_average = {"metrics": average_metrics, "summary": base.retrospective.summarize(average_metrics, data["primary"])}

    arrays = _data_arrays("sealed_monitor", data)
    for source_name, prefix in (("fit_arrays.npz", "gate_fit_archive"), ("selection_arrays.npz", "gate_select_archive")):
        with np.load(stage / "previous" / source_name, allow_pickle=False) as prior:
            for key in prior.files:
                arrays[f"{prefix}__{key}"] = prior[key]
    arrays["score__mean_proposer"] = mean_scores
    arrays["score__harm_guard"] = harm_scores
    arrays["score__advantage_only"] = advantage_scores
    arrays["proposal_mask"] = proposals
    arrays["sham_score"] = np.stack(sham_scores)
    arrays["sham_id"] = np.arange(len(sham_arms), dtype=np.int64)
    for name, arm in arms.items():
        base._store_arm(arrays, name, arm)
    for index, arm in enumerate(sham_arms):
        arrays[f"sham__actions__{index}"] = arm["actions"]
        arrays[f"sham__authorized__{index}"] = arm["override"]
        for metric, values in arm["metrics"].items():
            arrays[f"sham__metric__{metric}__{index}"] = values
    for metric, values in average_metrics.items():
        arrays[f"sham__average_metric__{metric}"] = values

    indices = np.flatnonzero(data["primary"])
    tokens = data["pair_token"][indices].astype(str)
    if tokens.tolist() != sorted(tokens.tolist()):
        raise RuntimeError("monitor tokens are not canonical")
    bootstrap = base.retrospective.paired_bootstrap_indices(len(indices), config)
    arrays["bootstrap_indices"] = bootstrap
    references = {"hard": hard, "proposer": proposer, "shuffled": sham_average}
    contrasts = {
        f"main_minus_{name}": {
            metric: _ci(main, reference, metric, indices, bootstrap)
            for metric in ("accuracy", "compatible", "regret", "worst_regret")
        }
        for name, reference in references.items()
    }
    criteria = config["diagnostic_criteria"]
    hard_c = contrasts["main_minus_hard"]
    proposer_c = contrasts["main_minus_proposer"]
    shuffled_c = contrasts["main_minus_shuffled"]
    shams_evaluable = all(status == "PASS" for status in sham_statuses)

    full_signs = {
        metric: base._sign(float(main["summary"][metric] - hard["summary"][metric]), float(config["selection"]["tie_atol"]))
        for metric in ("regret", "accuracy", "worst_regret")
    }
    shard_results = {}
    stable = True
    for shard, row in sorted(selected["shards"].items()):
        if row.get("status") != "PASS":
            stable = False
            shard_results[shard] = {"status": "NOT_EVALUABLE"}
            continue
        shard_proposer = row["proposer"]["selected"]
        shard_guard = row["guard"]["selected"]
        shard_proposals = proposal_mask(mean_scores, data["disagreement"], shard_proposer["threshold"])
        shard_result = apply_harm_guard(shard_proposals, harm_scores, data["hard_actions"], data["posterior_actions"], shard_guard["threshold"])
        shard_arm = _arm(shard_result["actions"], shard_result["authorized"], data, utilities, config)
        signs = {
            metric: base._sign(float(shard_arm["summary"][metric] - hard["summary"][metric]), float(config["selection"]["tie_atol"]))
            for metric in ("regret", "accuracy", "worst_regret")
        }
        nonidentity = shard_proposer["threshold"] != HARD_ONLY and shard_guard["threshold"] != HARD_ONLY
        same = signs == full_signs
        stable &= nonidentity and same
        shard_results[shard] = {"status": "PASS", "proposer": shard_proposer, "guard": shard_guard, "monitor_signs": signs, "nonidentity": nonidentity, "same_signs": same}
        base._store_arm(arrays, f"main_shard_{shard}", shard_arm)

    conditions: dict[str, Any] = {
        "diagnostic_condition_1": bool(hard_c["regret"]["mean_diff"] <= -float(criteria["regret_reduction_vs_hard_min"]) and hard_c["regret"]["ci95_high"] < float(criteria["regret_vs_hard_ci95_upper_below"])),
        "diagnostic_condition_2": bool(hard_c["accuracy"]["ci95_low"] >= float(criteria["accuracy_vs_hard_ci95_lower_at_least"]) and hard_c["compatible"]["ci95_low"] >= float(criteria["compatibility_vs_hard_ci95_lower_at_least"])),
        "diagnostic_condition_3": bool(hard_c["worst_regret"]["mean_diff"] <= float(criteria["worst_regret_vs_hard_mean_at_most"]) and hard_c["worst_regret"]["ci95_high"] <= float(criteria["worst_regret_vs_hard_ci95_upper_at_most"])),
        "diagnostic_condition_4": bool(proposer_c["accuracy"]["ci95_low"] > float(criteria["accuracy_vs_proposer_ci95_lower_above"]) and proposer_c["worst_regret"]["ci95_high"] < float(criteria["worst_regret_vs_proposer_ci95_upper_below"]) and proposer_c["regret"]["ci95_high"] <= float(criteria["regret_vs_proposer_ci95_upper_at_most"])),
        "diagnostic_condition_6_without_replay": bool(stable),
    }
    conditions["diagnostic_condition_5"] = (
        bool(shuffled_c["regret"]["ci95_high"] < float(criteria["regret_vs_shuffled_ci95_upper_below"]) and shuffled_c["worst_regret"]["ci95_high"] < float(criteria["worst_regret_vs_shuffled_ci95_upper_below"]))
        if shams_evaluable else "NOT_EVALUABLE"
    )

    set_index = target_set_indices(data["target"])
    support = per_set_support(
        set_index,
        data["primary"],
        config["absent_support"]["set_indices"],
        int(config["minimums"]["absent_support_tokens_per_set"]),
    )
    for target_index in config["absent_support"]["set_indices"]:
        mask = data["primary"] & (set_index == int(target_index))
        row = support[str(target_index)]
        if row["status"] == "EVALUABLE":
            local_indices = np.flatnonzero(mask)
            local_bootstrap = base.retrospective.paired_bootstrap_indices(len(local_indices), config)
            arrays[f"absent_set__{target_index}__bootstrap_indices"] = local_bootstrap
            row["status"] = "EVALUABLE"
            row["summaries"] = {name: base.retrospective.summarize(arm["metrics"], mask) for name, arm in arms.items()}
            row["contrasts"] = {
                f"main_minus_{name}": {
                    metric: base.retrospective.paired_delta_ci(main["metrics"][metric][local_indices], reference["metrics"][metric][local_indices], local_bootstrap)
                    for metric in ("accuracy", "compatible", "regret", "worst_regret")
                }
                for name, reference in references.items()
            }
        else:
            row.update({"summaries": None, "contrasts": None})

    labels = harm_labels(data["gain"])
    calibration_mask = data["primary"][:, None] & data["disagreement"]
    arm_summaries = {
        name: {"status": "PASS", "summary": arm["summary"], "override_diagnostics": arm["override_diagnostics"]}
        for name, arm in arms.items()
    }
    arm_summaries["mean_plus_shuffled_harm_guard"] = {
        "status": "PASS" if shams_evaluable else "NOT_EVALUABLE",
        "summary": sham_average["summary"],
        "replicate_statuses": sham_statuses,
    }
    core = {
        "status": "COMPLETE",
        "counts": _counts(data),
        "estimand": {
            "unit": "pair_token",
            "accuracy": "mean over 24 policies within token",
            "compatible": "mean over 24 policies within token",
            "regret": "mean over 24 policies within token",
            "worst_regret": "max over 24 policies within token then mean over tokens",
            "bootstrap_scope": config["bootstrap"]["scope"],
            "sham_average": "mean of five already-evaluated token-wise metric arrays",
        },
        "arms": arm_summaries,
        "contrasts": contrasts,
        "selector_stability": {"full_signs": full_signs, "shards": shard_results, "stable": stable},
        "harm_calibration": _calibration(harm_scores, labels, calibration_mask),
        "absent_support_by_set": support,
        "diagnostic_pattern": {
            "conditions": conditions,
            "all_observed_conditions_without_replay": bool(all(value is True for value in conditions.values())) if all(isinstance(value, bool) for value in conditions.values()) else None,
            "aggregate_with_replay": None,
            "decision_authority": "user",
        },
        "diagnostic_criteria": criteria,
        "split_disjointness": base._split_evidence(data),
        "claim_scope": "fresh realization of same synthetic law; fixed 24-policy catalogue; monitor-only conditional bootstrap; no GO/NO-GO",
    }
    base.write_npz_atomic(output / "sealed_monitor_bundle.npz", {key: value for key, value in data.items() if isinstance(value, np.ndarray)})
    base.write_npz_atomic(output / "result_arrays.npz", arrays)
    base.write_json_atomic(output / "analysis_core.json", core)
    base.write_json_atomic(output / "REPORT_WAVE57.json", core)
    return "COMPLETE"


def execute_phase(stage: Path, output: Path, phase: str) -> str:
    base.validate_stage(stage, phase)
    config, utilities, data = base.load_inputs(stage, phase)
    if output.exists():
        if any(output.iterdir()):
            raise RuntimeError("worker output directory must be empty")
    else:
        output.mkdir(parents=True)
    if phase == "fit":
        return run_fit(stage, output, config, utilities, data)
    if phase == "select":
        return run_select(stage, output, config, utilities, data)
    if phase == "adjudicate":
        return run_adjudicate(stage, output, config, utilities, data)
    raise ValueError(phase)


def _module_receipts(request: dict[str, Any]) -> list[dict[str, str]]:
    receipts = base.local_module_receipts(request)
    path = Path(__file__).resolve(strict=True)
    expected = request["execution_sources"].get(WORKER_SOURCE)
    actual = base.sha256_file(path)
    if expected != actual:
        raise RuntimeError("staged Wave 57 worker hash mismatch")
    receipts.append({
        "module": "_wave57_phase_worker",
        "module_file": str(path),
        "runtime_relative_path": path.name,
        "source": WORKER_SOURCE,
        "sha256": actual,
    })
    return sorted(receipts, key=lambda row: row["module"])


def main() -> None:
    args = base.parse_args()
    if not base.STAGED_RUNTIME:
        raise RuntimeError("Wave 57 analytical worker requires isolated staged runtime")
    stage = args.stage.resolve(strict=True)
    if Path.cwd().resolve() != stage:
        raise RuntimeError("worker cwd must be isolated stage")
    if os.geteuid() != 65534 or os.getegid() != 65534:
        raise RuntimeError("worker must run as nobody/nogroup")
    request = base.load_json(stage / "phase_request.json")
    phase = str(request["phase"])
    security = base.process_security_state()
    _module_receipts(request)
    files = base.validate_stage(stage, phase)
    probes = base.verify_forbidden_probes(args.forbidden_probe)
    output = args.output.resolve()
    if output.parent != stage.parent or output.name != "worker-output":
        raise RuntimeError("worker output must be isolated staging sibling")
    status = execute_phase(stage, output, phase)
    modules = _module_receipts(request)
    base.write_json_atomic(
        output / "access_receipt.json",
        {
            "phase": phase,
            "status": status,
            "effective_uid": os.geteuid(),
            "effective_gid": os.getegid(),
            "process_security": security,
            "stage_inventory": files,
            "stage_hashes": {name: base.sha256_file(stage / name) for name in files},
            "sealed_probes": probes,
            "benchmark_root_received": False,
            "local_import_runtime": str(base.RUNTIME_ROOT),
            "local_module_receipts": modules,
            "output_inventory": base.inventory(output),
        },
    )
    print(json.dumps({"phase": phase, "status": status}, sort_keys=True))


if __name__ == "__main__":
    main()
