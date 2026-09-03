#!/usr/bin/env python3
"""Run the CPU preflight for the local proportional-coherence graph contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from geometria_proporcional.proportional_graph_contract import (  # noqa: E402
    ProportionalGraphConfig,
    cycle_residual_rms,
    exact_path_closure,
    generate_graph_views,
    public_schema_hash,
    result_arrays,
    score_solver,
    solve_huber_irls,
    solve_oracle_weights,
    solve_weighted_least_squares,
    validate_graph_view,
)


DEFAULT_OUTPUT = REPO_ROOT / "data" / "geometria_proporcional" / "proportional_graph_preflight_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state() -> dict[str, object]:
    def git(*args: str) -> str:
        process = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False
        )
        return process.stdout.strip() if process.returncode == 0 else "UNAVAILABLE"

    status = git("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status and status != "UNAVAILABLE"),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _save_view(output: Path, view, config: ProportionalGraphConfig) -> dict[str, object]:
    view_id = view.private.view_id
    public_path = output / "public" / f"{view_id}.npz"
    private_path = output / "private" / f"{view_id}.npz"
    metadata_path = output / "private" / f"{view_id}.json"
    public_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(public_path, **view.public.arrays())
    np.savez_compressed(private_path, **view.private.arrays())
    metadata_path.write_text(
        json.dumps(view.private.metadata(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    raw_results: dict[str, object] = {}
    solvers = {
        "unweighted_wls": score_solver(solve_weighted_least_squares(view.public), view.private),
        "huber_irls": score_solver(
            solve_huber_irls(
                view.public,
                delta=config.huber_delta,
                max_iterations=config.irls_iterations,
                damping=config.irls_damping,
                weight_floor=config.weight_floor,
            ),
            view.private,
        ),
        "oracle_weights": solve_oracle_weights(view, config.weight_floor),
    }
    raw_dir = output / "evaluation" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, result in solvers.items():
        raw_path = raw_dir / f"{view_id}.{name}.npz"
        np.savez_compressed(raw_path, **result_arrays(result))
        raw_results[name] = {
            "quotient_rmse": result.quotient_rmse,
            "relation_rmse": result.relation_rmse,
            "weighted_residual_rmse": result.weighted_residual_rmse,
            "laplacian_rank": result.laplacian_rank,
            "laplacian_condition": result.laplacian_condition,
            "converged": result.converged,
            "iterations": result.iterations,
        }

    clean_cycle = cycle_residual_rms(
        view.public.edge_index,
        view.private.clean_log_ratio,
        view.public.n_nodes,
        view.public.edge_valid,
    )
    observed_cycle = cycle_residual_rms(
        view.public.edge_index,
        view.public.observed_log_ratio,
        view.public.n_nodes,
        view.public.edge_valid,
    )
    observed_path = exact_path_closure(view.public)
    return {
        "view_id": view_id,
        "master_id": view.private.master_id,
        "lineage_id": view.private.lineage_id,
        "split": view.private.split,
        "mechanism": view.private.corruption_mechanism,
        "n_nodes": view.public.n_nodes,
        "n_edges": len(view.public.edge_index),
        "n_paths": len(view.public.path_index),
        "n_corrupted": int(view.private.causal_corruption_mask.sum()),
        "clean_cycle_rms": clean_cycle,
        "observed_cycle_rms": observed_cycle,
        "observed_path_closure_rms": (
            float(np.sqrt(np.mean(observed_path**2))) if len(observed_path) else None
        ),
        "solvers": raw_results,
    }


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        key = f"{row['split']}|{row['mechanism']}"
        groups.setdefault(key, []).append(row)
    output: dict[str, object] = {}
    for key, members in sorted(groups.items()):
        solver_names = members[0]["solvers"].keys()
        output[key] = {
            "views": len(members),
            "masters": len({row["lineage_id"] for row in members}),
            "mean_nodes": float(np.mean([row["n_nodes"] for row in members])),
            "mean_edges": float(np.mean([row["n_edges"] for row in members])),
            "mean_paths": float(np.mean([row["n_paths"] for row in members])),
            "max_clean_cycle_rms": float(max(row["clean_cycle_rms"] for row in members)),
            "solvers": {
                solver: {
                    **{
                        metric: float(
                            np.mean(
                                [
                                    row["solvers"][solver][metric]
                                    for row in members
                                    if row["solvers"][solver]["converged"]
                                ]
                            )
                        )
                        if any(row["solvers"][solver]["converged"] for row in members)
                        else float("nan")
                        for metric in ("quotient_rmse", "relation_rmse", "weighted_residual_rmse")
                    },
                    "failure_rate": float(
                        np.mean([not row["solvers"][solver]["converged"] for row in members])
                    ),
                }
                for solver in solver_names
            },
        }
    return output


def _write_bootstrap_indices(output: Path, rows: list[dict[str, object]], config: ProportionalGraphConfig) -> None:
    test_masters = sorted({row["master_id"] for row in rows if row["split"] == "test"})
    rng = np.random.default_rng(config.seed + 1)
    indices = rng.integers(
        0,
        len(test_masters),
        size=(config.bootstrap_replicates, len(test_masters)),
    )
    np.savez_compressed(
        output / "bootstrap_indices.npz",
        master_id=np.asarray(test_masters),
        indices=indices,
    )


def _write_report(output: Path, config: ProportionalGraphConfig, summary: dict[str, object]) -> None:
    lines = [
        "# Preflight CPU — núcleo local de coherencia proporcional",
        "",
        "> Este corte valida contrato, generación, orientación, checkers y baselines clásicos. "
        "No evalúa todavía el mixer neuronal ni autoriza GO/NO-GO.",
        "",
        f"- masters: `{config.masters}`",
        f"- schema público: `{public_schema_hash()}`",
        f"- seed: `{config.seed}`",
        "",
        "| Slice | vistas | masters | nodos | aristas | paths | WLS quotient RMSE | IRLS quotient RMSE | IRLS fail | Oracle quotient RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, values in summary.items():
        solvers = values["solvers"]
        lines.append(
            f"| {key} | {values['views']} | {values['masters']} | {values['mean_nodes']:.2f} | "
            f"{values['mean_edges']:.2f} | {values['mean_paths']:.2f} | "
            f"{solvers['unweighted_wls']['quotient_rmse']:.4f} | "
            f"{solvers['huber_irls']['quotient_rmse']:.4f} | "
            f"{solvers['huber_irls']['failure_rate']:.3f} | "
            f"{solvers['oracle_weights']['quotient_rmse']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Lectura",
            "",
            "Las cifras son diagnósticas. El oracle usa la máscara causal privada y sólo mide "
            "solvabilidad; WLS e IRLS son referencias deployables. Una diferencia entre ellas no "
            "se atribuye a ninguna arquitectura neuronal.",
        ]
    )
    (output / "PREFLIGHT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output: Path, config: ProportionalGraphConfig, force: bool = False) -> dict[str, object]:
    if output.exists() and any(output.iterdir()):
        if not force:
            raise RuntimeError(f"output directory is not empty: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol_config.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    target_authority = {
        "schema_version": config.schema_version,
        "tracks": {
            "quotient_reconstruction": {
                "authority": "synthetic latent state modulo global additive gauge",
                "output": "centered node potentials",
                "scoring": "per-master quotient RMSE",
            },
            "causal_provenance": {
                "authority": "edges altered by the simulator",
                "output": "per-edge probability",
                "scoring": "AP and Brier; not observational falsity",
            },
            "nonidentifiable": {
                "authority": "certified equivalence class or posterior",
                "output": "set, posterior, or abstention",
                "scoring": "coverage or proper scoring; no binary accuracy",
                "status": "reserved_not_generated_in_preflight_v1",
            },
        },
    }
    (output / "target_authority_table.json").write_text(
        json.dumps(target_authority, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    views = generate_graph_views(config)
    rows = []
    for view in views:
        validate_graph_view(view)
        rows.append(_save_view(output, view, config))
    (output / "evaluation" / "per_view.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = _summary(rows)
    (output / "evaluation" / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_bootstrap_indices(output, rows, config)
    _write_report(output, config, summary)
    replay = output / "replay.sh"
    replay.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(REPO_ROOT))}\n"
        f"venv/bin/python {shlex.quote(str(Path(__file__).relative_to(REPO_ROOT)))} "
        f"--output {shlex.quote(str(output))} "
        f"--config {shlex.quote(str(output / 'protocol_config.json'))} --force\n",
        encoding="utf-8",
    )
    replay.chmod(0o755)
    source_paths = [
        Path(__file__).resolve(),
        REPO_ROOT / "src/geometria_proporcional/proportional_graph_contract.py",
        REPO_ROOT / "tests/test_proportional_graph_contract.py",
        REPO_ROOT / "experiments/geometria_proporcional/configs/proportional_graph_preflight_v1.json",
    ]
    files = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {
        "schema_version": config.schema_version,
        "public_schema_sha256": public_schema_hash(),
        "config": config.to_dict(),
        "git": _git_state(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "source_files": {
            str(path.relative_to(REPO_ROOT)): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in source_paths
        },
        "counts": {
            "masters": config.masters,
            "views": len(views),
            "public_files": len(list((output / "public").glob("*.npz"))),
            "private_array_files": len(list((output / "private").glob("*.npz"))),
        },
        "files": {
            str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in files
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--masters", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.config:
        config = ProportionalGraphConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    else:
        kwargs = {}
        if args.masters is not None:
            kwargs["masters"] = args.masters
        if args.seed is not None:
            kwargs["seed"] = args.seed
        config = ProportionalGraphConfig(**kwargs)
    manifest = run(args.output.resolve(), config, force=args.force)
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
