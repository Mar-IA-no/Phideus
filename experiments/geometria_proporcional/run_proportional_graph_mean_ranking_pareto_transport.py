#!/usr/bin/env python3
"""Evaluate utility-free Pareto transport for frozen mean-ranking policies on CPU."""

from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, os, platform, resource, subprocess, sys, time
from pathlib import Path
from typing import Any
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/geometria_proporcional")]
import run_proportional_graph_conditional_risk_gate as conditional  # noqa: E402
import run_proportional_graph_frozen_adapters as frozen  # noqa: E402
import run_proportional_graph_mean_ranking_attribution as attribution  # noqa: E402
import run_proportional_graph_mean_ranking_power_selection_audit as power  # noqa: E402

DEFAULT_CONFIG = (
    ROOT
    / "experiments/geometria_proporcional/configs/proportional_graph_mean_ranking_pareto_transport_v1.json"
)
SOURCE_FILES = (
    "experiments/geometria_proporcional/PLAN_PROPORTIONAL_MEAN_RANKING_PARETO_TRANSPORT_CPU.md",
    "experiments/geometria_proporcional/configs/proportional_graph_mean_ranking_pareto_transport_v1.json",
    "experiments/geometria_proporcional/run_proportional_graph_mean_ranking_pareto_transport.py",
    "experiments/geometria_proporcional/run_proportional_graph_mean_ranking_attribution.py",
    "experiments/geometria_proporcional/run_proportional_graph_mean_ranking_power_selection_audit.py",
    "experiments/geometria_proporcional/run_proportional_graph_conditional_risk_gate.py",
    "experiments/geometria_proporcional/run_proportional_graph_frozen_adapters.py",
)
MAIN = ("reduced_mean", "public_base_mean", "topology_mean")
CONTROL = "topology_permuted_mean"
POLICY_IDS = (
    "identity",
    "budget_0.01",
    "budget_0.02",
    "budget_0.05",
    "budget_0.10",
    "budget_0.20",
    "budget_0.40",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_config(path: Path) -> dict[str, Any]:
    c = json.loads(path.read_text())
    expected = {
        "schema_version",
        "source_power_selection_audit",
        "source_power_manifest_sha256",
        "source_mean_ranking_attribution",
        "source_attribution_manifest_sha256",
        "arms",
        "proposal_regimes",
        "primary_proposal",
        "budget_fractions",
        "control_replicates",
        "bootstrap_replicates",
        "numerical_zero_tolerance",
        "execution",
    }
    if (
        set(c) != expected
        or c["schema_version"] != "proportional-graph-mean-ranking-pareto-transport-v1"
    ):
        raise ValueError("invalid Pareto schema")
    if c["arms"] != [
        "raw_generic",
        "raw_typed",
        "closure_generic",
        "closure_typed",
    ] or c["proposal_regimes"] != ["fixed_alpha_0.25", "r368_common"]:
        raise ValueError("factorial changed")
    if c["primary_proposal"] != "fixed_alpha_0.25" or c["budget_fractions"] != [
        0.01,
        0.02,
        0.05,
        0.10,
        0.20,
        0.40,
    ]:
        raise ValueError("policy set changed")
    if (
        c["control_replicates"] != 16
        or c["bootstrap_replicates"] != 2000
        or c["numerical_zero_tolerance"] != 1e-12
    ):
        raise ValueError("audit contract changed")
    if c["execution"] != {"max_seconds": 300, "max_rss_gib": 4.0}:
        raise ValueError("execution contract changed")
    return c


def verify_manifest(root: Path, expected: str, label: str) -> None:
    if sha(root / "manifest.json") != expected:
        raise AssertionError(f"{label} manifest mismatch")
    m = json.loads((root / "manifest.json").read_text())
    for rel, h in m["deterministic_files"].items():
        if sha(root / rel) != h:
            raise AssertionError(f"{label} mismatch: {rel}")


def verify_sources(
    c: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    pr = ROOT / c["source_power_selection_audit"]
    ar = ROOT / c["source_mean_ranking_attribution"]
    verify_manifest(pr, c["source_power_manifest_sha256"], "R370")
    verify_manifest(ar, c["source_attribution_manifest_sha256"], "R369")
    pc = power.load_config(pr / "resolved_config.json")
    power.verify_source(pc)
    ac = attribution.load_config(ar / "resolved_config.json")
    r368, recon, _ = attribution.verify_sources(ac)
    return ac, r368, recon


def require_clean(dev: bool) -> None:
    if (
        not dev
        and subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    ):
        raise RuntimeError("official Pareto diagnostic requires clean worktree")


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def pareto_mask(means: np.ndarray, tol: float) -> np.ndarray:
    x = np.asarray(means, float)
    keep = np.ones(len(x), bool)
    for i in range(len(x)):
        weak = np.all(x <= x[i] + tol, axis=1)
        strict = np.any(x < x[i] - tol, axis=1)
        if np.any(weak & strict):
            keep[i] = False
    return keep


def objective(quotient: np.ndarray, action: np.ndarray) -> np.ndarray:
    d = conditional.selected_delta(quotient, action)
    return np.stack((d[::2], d[1::2]))


def front_record(
    values: np.ndarray, bootstrap: np.ndarray, tol: float
) -> tuple[dict[str, Any], np.ndarray]:
    means = values.mean(axis=2)
    mask = pareto_mask(means, tol)
    freq = np.zeros(len(values))
    for idx in bootstrap:
        freq += pareto_mask(values[:, :, idx].mean(axis=2), tol)
    freq /= len(bootstrap)
    return {
        "front_policy_ids": [POLICY_IDS[i] for i in np.flatnonzero(mask)],
        "front_size": int(mask.sum()),
        "coordinates": {
            POLICY_IDS[i]: {
                "iid_delta": float(means[i, 0]),
                "grouped_delta": float(means[i, 1]),
                "bootstrap_front_frequency": float(freq[i]),
            }
            for i in range(len(values))
        },
    }, mask


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    u = np.sum(a | b)
    return float(np.sum(a & b) / u) if u else 1.0


def transport_record(
    sel: np.ndarray,
    adj: np.ndarray,
    sel_values: np.ndarray,
    adj_values: np.ndarray,
) -> dict[str, Any]:
    sel_means = sel_values.mean(axis=2)
    adj_means = adj_values.mean(axis=2)

    def changed(mask: np.ndarray) -> list[dict[str, Any]]:
        return [
            {
                "policy_id": POLICY_IDS[index],
                "policy_selection": {
                    "iid_delta": float(sel_means[index, 0]),
                    "grouped_delta": float(sel_means[index, 1]),
                },
                "adjudication": {
                    "iid_delta": float(adj_means[index, 0]),
                    "grouped_delta": float(adj_means[index, 1]),
                },
            }
            for index in np.flatnonzero(mask)
        ]

    return {
        "jaccard": jaccard(sel, adj),
        "retention": float(np.sum(sel & adj) / np.sum(sel)),
        "oracle_recall": float(np.sum(sel & adj) / np.sum(adj)),
        "lost": changed(sel & ~adj),
        "added": changed(adj & ~sel),
    }


def covers(left: np.ndarray, right: np.ndarray, tol: float) -> bool:
    return all(np.any(np.all(left <= point + tol, axis=1)) for point in right)


def compare_sets(left: np.ndarray, right: np.ndarray, tol: float) -> str:
    lr = covers(left, right, tol)
    rl = covers(right, left, tol)
    if lr and rl:
        return "EQUIVALENT"
    if lr:
        return "TOPOLOGY_DOMINATES"
    if rl:
        return "CONTROL_DOMINATES"
    return "INCOMPARABLE"


def family_values(
    c: dict[str, Any],
    pack: dict[str, np.ndarray],
    quotient: np.ndarray,
    regime: str,
    arm: str,
    role: str,
    family: str,
    replicate: int | None = None,
) -> np.ndarray:
    actions = [np.zeros(quotient.shape[-1], dtype=np.int64)]
    for frac in c["budget_fractions"]:
        key = (
            f"{regime}|{arm}|budget={frac:.2f}|action|{family}"
            if role == "policy_selection"
            else f"{regime}|{arm}|budget={frac:.2f}|ranked|action|{family}"
        )
        value = pack[key]
        actions.append(value if replicate is None else value[replicate])
    return np.stack([objective(quotient, a) for a in actions])


def audit_cohort(
    c: dict[str, Any],
    ac: dict[str, Any],
    r368: dict[str, Any],
    recon: dict[str, Any],
    name: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = ROOT / c["source_mean_ranking_attribution"]
    co = attribution.load_cohort(ac, r368, recon, name)
    packs = {
        "policy_selection": read_npz(root / f"cohort_{name.lower()}_selection.npz"),
        "adjudication": read_npz(root / f"cohort_{name.lower()}_adjudication.npz"),
    }
    boots = {
        "policy_selection": co["selection_bootstrap"],
        "adjudication": co["adjudication_bootstrap"],
    }
    out = {"cohort": name, "proposals": {}}
    saved = {f"{r}|bootstrap": b for r, b in boots.items()}
    for regime in c["proposal_regimes"]:
        rr = {"arms": {}}
        for ai, arm in enumerate(c["arms"]):
            records = {}
            arrays = {}
            for role in ("policy_selection", "adjudication"):
                q = co["states"][role]["data"]["quotient_rmse"][ai]
                records[role] = {"families": {}, CONTROL: []}
                arrays[role] = {}
                for fam in MAIN:
                    v = family_values(c, packs[role], q, regime, arm, role, fam)
                    rec, mask = front_record(
                        v, boots[role], c["numerical_zero_tolerance"]
                    )
                    records[role]["families"][fam] = rec
                    arrays[role][fam] = (v, mask)
                    saved[f"{regime}|{arm}|{role}|{fam}|objectives"] = v
                    saved[f"{regime}|{arm}|{role}|{fam}|front_mask"] = mask
                for rep in range(c["control_replicates"]):
                    v = family_values(
                        c, packs[role], q, regime, arm, role, CONTROL, rep
                    )
                    rec, mask = front_record(
                        v, boots[role], c["numerical_zero_tolerance"]
                    )
                    records[role][CONTROL].append({"replicate": rep, **rec})
                    arrays[role].setdefault(CONTROL, []).append((v, mask))
                    saved[f"{regime}|{arm}|{role}|{CONTROL}={rep}|objectives"] = v
                    saved[f"{regime}|{arm}|{role}|{CONTROL}={rep}|front_mask"] = mask
            comparisons = {}
            for fam in MAIN:
                sel_values, sel_mask = arrays["policy_selection"][fam]
                adj_values, adj_mask = arrays["adjudication"][fam]
                comparisons[fam] = transport_record(
                    sel_mask, adj_mask, sel_values, adj_values
                )
            comparisons[CONTROL] = []
            for rep in range(c["control_replicates"]):
                sel_values, sel_mask = arrays["policy_selection"][CONTROL][rep]
                adj_values, adj_mask = arrays["adjudication"][CONTROL][rep]
                comparisons[CONTROL].append(
                    {
                        "replicate": rep,
                        **transport_record(sel_mask, adj_mask, sel_values, adj_values),
                    }
                )
            top_sel = arrays["policy_selection"]["topology_mean"][1]
            top_adj_values = arrays["adjudication"]["topology_mean"][0]
            top_oracle = arrays["adjudication"]["topology_mean"][1]
            dominance = {}
            for fam in ("reduced_mean", "public_base_mean"):
                sv, sm = arrays["adjudication"][fam]
                dominance[fam] = {
                    "selected_fronts": compare_sets(
                        top_adj_values[top_sel].mean(axis=2),
                        sv[arrays["policy_selection"][fam][1]].mean(axis=2),
                        c["numerical_zero_tolerance"],
                    ),
                    "oracle_fronts": compare_sets(
                        top_adj_values[top_oracle].mean(axis=2),
                        sv[sm].mean(axis=2),
                        c["numerical_zero_tolerance"],
                    ),
                }
            dominance[CONTROL] = []
            for rep, (sv, sm) in enumerate(arrays["adjudication"][CONTROL]):
                csel = arrays["policy_selection"][CONTROL][rep][1]
                dominance[CONTROL].append(
                    {
                        "replicate": rep,
                        "selected_fronts": compare_sets(
                            top_adj_values[top_sel].mean(axis=2),
                            sv[csel].mean(axis=2),
                            c["numerical_zero_tolerance"],
                        ),
                        "oracle_fronts": compare_sets(
                            top_adj_values[top_oracle].mean(axis=2),
                            sv[sm].mean(axis=2),
                            c["numerical_zero_tolerance"],
                        ),
                    }
                )
            rr["arms"][arm] = {
                "roles": records,
                "transport": comparisons,
                "dominance": dominance,
            }
        out["proposals"][regime] = rr
    return out, saved


def build_analysis(c: dict[str, Any], reports: dict[str, Any]) -> dict[str, Any]:
    result = {
        "status": "OPENED_PARETO_TRANSPORT_DIAGNOSTIC",
        "authority": "utility-free post hoc set diagnostic; no policy selection, promotion, or GO/NO-GO",
        "proposals": {},
    }
    for regime in c["proposal_regimes"]:
        statuses = (
            "TOPOLOGY_DOMINATES",
            "CONTROL_DOMINATES",
            "EQUIVALENT",
            "INCOMPARABLE",
        )
        rr = {
            "arms": {},
            "aggregate": {
                "topology_retention": [],
                "topology_selection_front_size": [],
                "topology_cross_cohort_jaccard": [],
                "control_transport_jaccard": [],
                "control_transport_retention": [],
                "control_transport_oracle_recall": [],
                "main_dominance": {
                    fam: {
                        role: {k: 0 for k in statuses}
                        for role in ("selected_fronts", "oracle_fronts")
                    }
                    for fam in ("reduced_mean", "public_base_mean")
                },
                "permuted_dominance": {
                    role: {k: 0 for k in statuses}
                    for role in ("selected_fronts", "oracle_fronts")
                },
            },
        }
        for arm in c["arms"]:
            rows = {n: reports[n]["proposals"][regime]["arms"][arm] for n in ("A", "B")}
            am = {}
            for n, row in rows.items():
                t = row["transport"]["topology_mean"]
                size = row["roles"]["policy_selection"]["families"]["topology_mean"][
                    "front_size"
                ]
                rr["aggregate"]["topology_retention"].append(t["retention"])
                rr["aggregate"]["topology_selection_front_size"].append(size)
                for fam in ("reduced_mean", "public_base_mean"):
                    for role in ("selected_fronts", "oracle_fronts"):
                        rr["aggregate"]["main_dominance"][fam][role][
                            row["dominance"][fam][role]
                        ] += 1
                for x in row["dominance"][CONTROL]:
                    for role in ("selected_fronts", "oracle_fronts"):
                        rr["aggregate"]["permuted_dominance"][role][x[role]] += 1
                for x in row["transport"][CONTROL]:
                    for metric in ("jaccard", "retention", "oracle_recall"):
                        rr["aggregate"][f"control_transport_{metric}"].append(x[metric])
                am[n] = {
                    "topology_transport": t,
                    "topology_selection_front": row["roles"]["policy_selection"][
                        "families"
                    ]["topology_mean"]["front_policy_ids"],
                    "dominance": row["dominance"],
                }
            sa = set(am["A"]["topology_selection_front"])
            sb = set(am["B"]["topology_selection_front"])
            jac = len(sa & sb) / len(sa | sb)
            rr["aggregate"]["topology_cross_cohort_jaccard"].append(jac)
            am["selection_front_cross_cohort_jaccard"] = jac
            rr["arms"][arm] = am
        for k in (
            "topology_retention",
            "topology_selection_front_size",
            "topology_cross_cohort_jaccard",
            "control_transport_jaccard",
            "control_transport_retention",
            "control_transport_oracle_recall",
        ):
            x = np.asarray(rr["aggregate"][k], float)
            rr["aggregate"][k] = {
                "mean": float(x.mean()),
                "min": float(x.min()),
                "max": float(x.max()),
                "values": x.tolist(),
            }
        result["proposals"][regime] = rr
    return result


def write_replay(out: Path, c: dict[str, Any], head: str) -> None:
    checks = "\n".join(
        f"printf '%s  %s\\n' '{sha(ROOT/r)}' \"$repo/{r}\" | sha256sum -c -"
        for r in SOURCE_FILES
    )
    for key, pathkey in (
        ("source_power_manifest_sha256", "source_power_selection_audit"),
        ("source_attribution_manifest_sha256", "source_mean_ranking_attribution"),
    ):
        checks += f"\nprintf '%s  %s\\n' '{c[key]}' \"$repo/{c[pathkey]}/manifest.json\" | sha256sum -c -"
    text = f"""#!/bin/sh\nset -eu\nrepo=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)\n[ "$(git -C "$repo" rev-parse HEAD)" = "{head}" ] || exit 2\n[ -z "$(git -C "$repo" status --porcelain)" ] || exit 2\n{checks}\n: "${{OUTPUT_DIR:?set OUTPUT_DIR}}"\nexec env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$repo/venv/bin/python" "$repo/experiments/geometria_proporcional/run_proportional_graph_mean_ranking_pareto_transport.py" --config "$repo/experiments/geometria_proporcional/configs/proportional_graph_mean_ranking_pareto_transport_v1.json" --output "$OUTPUT_DIR"\n"""
    (out / "replay.sh").write_text(text)
    (out / "replay.sh").chmod(0o755)


def run(c: dict[str, Any], out: Path, dev: bool) -> None:
    require_clean(dev)
    if out.exists():
        raise FileExistsError(out)
    if (
        not dev
        and (ROOT / "data/geometria_proporcional").resolve()
        not in out.resolve().parents
    ):
        raise ValueError("official path invalid")
    started = time.monotonic()
    ac, r368, recon = verify_sources(c)
    out.mkdir(parents=True)
    reports = {}
    for n in ("A", "B"):
        report, pack = audit_cohort(c, ac, r368, recon, n)
        reports[n] = report
        write_json(out / f"cohort_{n.lower()}_pareto.json", report)
        frozen.save_npz(out / f"cohort_{n.lower()}_objectives.npz", pack)
    write_json(out / "analysis.json", build_analysis(c, reports))
    write_json(
        out / "environment.json",
        {
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "scipy": importlib.metadata.version("scipy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "threads": 1,
            "cuda_visible_devices": "",
            "refit": False,
            "new_views": 0,
            "new_solves": 0,
            "gpu_queried": False,
        },
    )
    write_json(out / "resolved_config.json", c)
    write_replay(out, c, git_head())
    runtime = {
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
    }
    if (
        runtime["elapsed_seconds"] > c["execution"]["max_seconds"]
        or runtime["max_rss_gib"] > c["execution"]["max_rss_gib"]
    ):
        raise RuntimeError("resource budget exceeded")
    write_json(out / "runtime_observation.json", runtime)
    files = sorted(
        p
        for p in out.iterdir()
        if p.is_file() and p.name not in {"manifest.json", "runtime_observation.json"}
    )
    write_json(
        out / "manifest.json",
        {
            "schema_version": c["schema_version"],
            "git_head": git_head(),
            "source_power_manifest_sha256": c["source_power_manifest_sha256"],
            "source_attribution_manifest_sha256": c[
                "source_attribution_manifest_sha256"
            ],
            "source_hashes": {r: sha(ROOT / r) for r in SOURCE_FILES},
            "deterministic_files": {p.name: sha(p) for p in files},
            "runtime_exclusions": ["runtime_observation.json"],
        },
    )
    print(json.dumps(runtime, sort_keys=True))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--development", action="store_true")
    a = p.parse_args()
    run(load_config(a.config.resolve()), a.output.resolve(), a.development)


if __name__ == "__main__":
    main()
