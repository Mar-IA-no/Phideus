#!/usr/bin/env python3
"""Execute the audited proportional MAPPING-FEASIBILITY gate on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments/geometria_proporcional"
DEFAULT_CONFIG = HERE / "configs/proportional_mapping_feasibility_v1.json"
DEFAULT_OUTPUT = ROOT / "data/geometria_proporcional/proportional_mapping_feasibility_v1"
SCRIPTS = {
    "prepare": HERE / "prepare_proportional_mapping_feasibility.py",
    "build": HERE / "build_proportional_mapping_candidate.py",
    "evaluate": HERE / "evaluate_proportional_mapping_feasibility.py",
    "check": HERE / "check_proportional_mapping_feasibility.py",
}
FIXED = {
    "gpu_used_or_queried": False,
    "architecture_promoted": False,
    "scientific_decision": None,
    "decision_authority": "user",
}


class BudgetExceeded(RuntimeError):
    """A watchdog, address-space limit, or resource signal closed the run."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def scientific_files(run: Path, exclusions: set[str]) -> dict[str, Any]:
    files = {}
    for path in sorted(p for p in run.rglob("*") if p.is_file()):
        relative = path.relative_to(run).as_posix()
        if relative in exclusions:
            continue
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return files


def limit_address_space(limit: int) -> None:
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def run_command(command: list[str], deadline: float, address_limit: int, env: dict[str, str]) -> dict[str, Any]:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("global mapping-feasibility watchdog expired")
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=remaining,
        preexec_fn=lambda: limit_address_space(address_limit),
    )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        message = (
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
        budget_markers = ("MemoryError", "Cannot allocate memory", "failed to map segment", "out of memory", "std::bad_alloc")
        if completed.returncode < 0 or any(marker.lower() in message.lower() for marker in budget_markers):
            raise BudgetExceeded(message)
        raise RuntimeError(message)
    return {"argv": [Path(command[0]).name, Path(command[1]).name, *command[2:]], "wall_seconds": elapsed, "returncode": completed.returncode}


def commands_for(run: Path, config: Path) -> list[tuple[str, list[str]]]:
    python = sys.executable
    return [
        ("prepare", [python, str(SCRIPTS["prepare"]), "--config", str(config), "--output", str(run)]),
        ("build", [python, str(SCRIPTS["build"]), "--public", str(run / "prepared/public"), "--output", str(run)]),
        (
            "evaluate",
            [
                python,
                str(SCRIPTS["evaluate"]),
                "--public",
                str(run / "prepared/public"),
                "--private",
                str(run / "prepared/private_dev"),
                "--candidate",
                str(run / "mapping_candidate.json"),
                "--output",
                str(run),
            ],
        ),
        ("check_pre", [python, str(SCRIPTS["check"]), "--config", str(config), "--run", str(run), "--phase", "pre"]),
    ]


def compare_cores(run_a: Path, run_b: Path) -> dict[str, Any]:
    a = json.loads((run_a / "core_manifest.json").read_text())
    b = json.loads((run_b / "core_manifest.json").read_text())
    if a != b:
        return {"status": "FAIL", "reason": "core_manifest_mismatch", "core_a_sha256": sha256_file(run_a / "core_manifest.json"), "core_b_sha256": sha256_file(run_b / "core_manifest.json")}
    mismatches = []
    for relative in a["files"]:
        if (run_a / relative).read_bytes() != (run_b / relative).read_bytes():
            mismatches.append(relative)
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "core_a_sha256": sha256_file(run_a / "core_manifest.json"),
        "core_b_sha256": sha256_file(run_b / "core_manifest.json"),
        "files_compared": len(a["files"]),
        "mismatches": mismatches,
        **FIXED,
    }


def current_usage(started: float, budget: dict[str, Any]) -> tuple[float, int]:
    wall = time.monotonic() - started
    rss = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024)
    if wall >= float(budget["wall_seconds_exclusive"]) or rss >= int(budget["rss_bytes_exclusive"]):
        raise BudgetExceeded(f"terminal budget exceeded: wall={wall}, rss={rss}")
    return wall, rss


def invalidate_outputs(output: Path, artifact_status: str, error: str | None, replay_status: str) -> None:
    for path in (output / "manifest.json",):
        path.unlink(missing_ok=True)
    for label in ("run_a", "run_b"):
        run = output / label
        if not run.is_dir():
            continue
        for relative in ("scientific_manifest.json", "REPORT_MAPPING_FEASIBILITY.md"):
            (run / relative).unlink(missing_ok=True)
        write_json(
            run / "adjudication.json",
            {
                "schema_version": "proportional-mapping-adjudication-v1",
                "technical_status": {
                    "source_status": "FAIL" if artifact_status == "FAIL" else "PASS",
                    "artifact_status": artifact_status,
                    "checker_status": "FAIL",
                    "replay_status": replay_status,
                },
                "mapping_decision": None,
                "predicates": [],
                "terminal_error": error,
                **FIXED,
            },
        )


def execute(config_path: Path, output: Path, development: bool) -> None:
    config = json.loads(config_path.read_text())
    budget = config["budget"]
    if output.exists():
        raise FileExistsError(output)
    if not development:
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
        if dirty:
            raise RuntimeError("official mapping-feasibility run requires a clean worktree")
    output.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + float(budget["wall_seconds_exclusive"])
    address_limit = int(budget["address_space_bytes_exclusive"])
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    phases: dict[str, list[dict[str, Any]]] = {"run_a": [], "run_b": [], "global": []}
    artifact_status = "PASS"
    error: str | None = None
    try:
        for label in ("run_a", "run_b"):
            run = output / label
            for phase, command in commands_for(run, config_path):
                receipt = run_command(command, deadline, address_limit, env)
                receipt["phase"] = phase
                phases[label].append(receipt)
                if phase == "build":
                    (run / "mapping_candidate.json").chmod(0o444)
        test_command = [sys.executable, "-m", "pytest", "-q", "tests/test_proportional_mapping_feasibility.py"]
        test_env = dict(env)
        test_env["MAPPING_FEASIBILITY_RUN_A"] = str(output / "run_a")
        test_env["MAPPING_FEASIBILITY_RUN_B"] = str(output / "run_b")
        test_receipt = run_command(test_command, deadline, address_limit, test_env)
        test_receipt["phase"] = "tests"
        phases["global"].append(test_receipt)
        for label in ("run_a", "run_b"):
            run = output / label
            mutations = json.loads((run / "mutation_results.json").read_text())
            if mutations.get("status") != "PASS" or set(mutations.get("candidate_corruptions", {}).values()) != {"REJECTED"}:
                raise RuntimeError(f"{label} actual mutation suite incomplete")
            files = scientific_files(
                run,
                {"runtime.json", "core_manifest.json", "adjudication.json", "REPORT_MAPPING_FEASIBILITY.md", "scientific_manifest.json", "replay_evidence.json"},
            )
            write_json(run / "core_manifest.json", {"schema_version": "proportional-mapping-core-manifest-v1", "files": files, **FIXED})
        replay = compare_cores(output / "run_a", output / "run_b")
        if replay["status"] != "PASS":
            raise RuntimeError(f"scientific replay mismatch: {replay}")
        write_json(output / "replay_evidence.json", {"schema_version": "proportional-mapping-replay-evidence-v1", **replay})
        wall_before_final, rss_before_final = current_usage(started, budget)
        for label in ("run_a", "run_b"):
            run = output / label
            shutil.copyfile(output / "replay_evidence.json", run / "replay_evidence.json")
            write_json(
                run / "terminal_status.json",
                {
                    "schema_version": "proportional-mapping-terminal-status-v1",
                    "artifact_status": "PASS",
                    "wall_seconds_before_final": wall_before_final,
                    "max_ru_maxrss_bytes_before_final": rss_before_final,
                    "limits": budget,
                    "core_manifest_sha256": sha256_file(run / "core_manifest.json"),
                    "replay_evidence_sha256": sha256_file(run / "replay_evidence.json"),
                    **FIXED,
                },
            )
            command = [sys.executable, str(SCRIPTS["check"]), "--config", str(config_path), "--run", str(run), "--phase", "final", "--replay-evidence", str(run / "replay_evidence.json"), "--terminal-status", str(run / "terminal_status.json")]
            receipt = run_command(command, deadline, address_limit, env)
            receipt["phase"] = "check_final"
            phases[label].append(receipt)
        for relative in ("adjudication.json", "REPORT_MAPPING_FEASIBILITY.md"):
            if (output / "run_a" / relative).read_bytes() != (output / "run_b" / relative).read_bytes():
                raise RuntimeError(f"final replay mismatch: {relative}")
        for label in ("run_a", "run_b"):
            adjudication = json.loads((output / label / "adjudication.json").read_text())
            if any(
                adjudication.get("technical_status", {}).get(key) != "PASS"
                for key in ("source_status", "artifact_status", "checker_status", "replay_status")
            ):
                raise RuntimeError(f"{label} final checker did not authorize semantic adjudication")
        for label in ("run_a", "run_b"):
            run = output / label
            files = scientific_files(run, {"runtime.json", "scientific_manifest.json"})
            write_json(run / "scientific_manifest.json", {"schema_version": "proportional-mapping-scientific-manifest-v1", "files": files, **FIXED})
        if (output / "run_a/scientific_manifest.json").read_bytes() != (output / "run_b/scientific_manifest.json").read_bytes():
            raise RuntimeError("scientific manifest replay mismatch")
        current_usage(started, budget)
    except (TimeoutError, subprocess.TimeoutExpired, MemoryError, BudgetExceeded) as exc:
        artifact_status = "BUDGET_EXCEEDED"
        error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # preserve a terminal technical failure without a semantic leaf
        artifact_status = "FAIL"
        error = f"{type(exc).__name__}: {exc}"
    wall = time.monotonic() - started
    max_rss_bytes = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024)
    if wall >= float(budget["wall_seconds_exclusive"]) or max_rss_bytes >= int(budget["rss_bytes_exclusive"]):
        artifact_status = "BUDGET_EXCEEDED"
    runtime = {
        "schema_version": "proportional-mapping-runtime-v1",
        "artifact_status": artifact_status,
        "wall_seconds": wall,
        "max_ru_maxrss_bytes": max_rss_bytes,
        "limits": budget,
        "phases": phases,
        "error": error,
        **FIXED,
    }
    write_json(output / "runtime.json", runtime)
    for label in ("run_a", "run_b"):
        run = output / label
        if run.is_dir():
            write_json(run / "runtime.json", runtime)
    if artifact_status != "PASS":
        replay_status = "PASS" if (output / "replay_evidence.json").is_file() and json.loads((output / "replay_evidence.json").read_text()).get("status") == "PASS" else "NOT_RUN"
        invalidate_outputs(output, artifact_status, error, replay_status)
        raise RuntimeError(f"mapping-feasibility terminated with {artifact_status}: {error}")
    adjudication = json.loads((output / "run_a/adjudication.json").read_text())
    write_json(
        output / "manifest.json",
        {
            "schema_version": "proportional-mapping-package-manifest-v1",
            "mapping_decision": adjudication["mapping_decision"],
            "run_a_manifest_sha256": sha256_file(output / "run_a/scientific_manifest.json"),
            "run_b_manifest_sha256": sha256_file(output / "run_b/scientific_manifest.json"),
            "replay_evidence_sha256": sha256_file(output / "replay_evidence.json"),
            "execution_sources": {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in [Path(__file__), config_path, *SCRIPTS.values()]},
            **FIXED,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    execute(args.config.resolve(strict=True), args.output.resolve(), args.development)


if __name__ == "__main__":
    main()
