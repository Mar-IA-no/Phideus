#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def norm_path(path):
    return str(path).replace("\\", "/").strip("./")


def under_root(path, root):
    p = norm_path(path)
    r = norm_path(root)
    return p == r or p.startswith(r + "/")


def main():
    parser = argparse.ArgumentParser(description="Validate documentation changes against local policies.")
    parser.add_argument("--changed", action="append", default=[], help="Changed path (repeatable)")
    parser.add_argument("--new-doc", action="append", default=[], help="New/renamed doc path (repeatable)")
    parser.add_argument("--front", default="", help="Detected front")
    parser.add_argument("--collab-mode", default="off", choices=["on", "off"], help="Current collab mode")
    parser.add_argument("--legacy-allowed", action="store_true", help="Allow legacy updates")
    parser.add_argument(
        "--policy",
        default="tools/skills/phideus-doc-maintainer/references/global_update_policy.yaml",
        help="Path to global policy yaml",
    )
    parser.add_argument("--repo-root", default=".", help="Repo root")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    policy = load_yaml(repo_root / args.policy)
    defaults = policy.get("defaults", {})

    changed = [norm_path(p) for p in args.changed]
    new_docs = [norm_path(p) for p in args.new_doc]

    errors = []
    warnings = []

    forbidden_paths = [norm_path(p) for p in defaults.get("forbidden_paths", [])]
    tier_a_docs = [norm_path(p) for p in defaults.get("tier_a_docs", [])]
    legacy_roots = [norm_path(p) for p in defaults.get("legacy_roots", [])]
    collab_root = norm_path(defaults.get("collab_root", "COLLAB"))

    for path in changed:
        if path in forbidden_paths:
            errors.append(f"forbidden_path: {path}")

        if args.collab_mode == "off" and under_root(path, collab_root):
            errors.append(f"collab_off_path: {path}")

        if not args.legacy_allowed:
            for legacy_root in legacy_roots:
                if under_root(path, legacy_root):
                    front_is_legacy = args.front in {"legacy", "uoemd"}
                    if not front_is_legacy:
                        errors.append(f"legacy_path_without_allowance: {path}")
                    break

        abs_path = repo_root / path
        if not abs_path.exists():
            warnings.append(f"path_not_found: {path}")

    if new_docs and "Documents/INDICE_DOCUMENTACION.md" not in changed:
        warnings.append("indice_not_updated_for_new_docs")

    if "README.md" in changed and "Documents/Proyecto_Estado_Actual.md" not in changed:
        warnings.append("readme_changed_without_estado_actual")

    tier_a_touched = [p for p in changed if p in tier_a_docs]
    if tier_a_touched and not any(p == "Documents/bitacora_desarrollo.md" for p in changed):
        warnings.append("tier_a_changed_without_bitacora_update")

    result = {
        "errors": errors,
        "warnings": warnings,
        "policy_checks_passed": len(errors) == 0,
    }

    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0 if len(errors) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
