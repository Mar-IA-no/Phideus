#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_path(path):
    return str(path).replace("\\", "/").strip("./")


def infer_custom_front(front_id, repo_root):
    token = front_id.lower()

    doc_roots = []
    docs_root = repo_root / "Documents"
    if docs_root.exists():
        for child in docs_root.iterdir():
            if child.is_dir() and token in child.name.lower():
                doc_roots.append(str(child.relative_to(repo_root)).replace("\\", "/"))

    code_roots = []
    for code_parent in [repo_root / "experiments", repo_root / "src"]:
        if code_parent.exists():
            for child in code_parent.iterdir():
                if child.is_dir() and token in child.name.lower():
                    code_roots.append(str(child.relative_to(repo_root)).replace("\\", "/"))

    primary_docs = []
    secondary_docs = []

    for root in doc_roots:
        root_path = repo_root / root
        for candidate in sorted(root_path.glob("*.md")):
            rel = str(candidate.relative_to(repo_root)).replace("\\", "/")
            if "roadmap" in candidate.name.lower() or "plan" in candidate.name.lower():
                primary_docs.append(rel)
            elif any(x in candidate.name.lower() for x in ["resultado", "informe", "estado"]):
                secondary_docs.append(rel)

    return {
        "aliases": [front_id],
        "tags": [front_id],
        "doc_roots": doc_roots,
        "code_roots": code_roots,
        "primary_docs": sorted(set(primary_docs))[:8],
        "secondary_docs": sorted(set(secondary_docs))[:10],
        "legacy_flag": False,
    }


def extract_pattern_docs(doc_roots, repo_root):
    docs = []
    patterns = ["*ROADMAP*.md", "*Plan*.md", "*PLAN*.md", "*RESULT*.md", "*INFORME*.md", "*Estado*.md"]

    for root in doc_roots:
        root_path = repo_root / root
        if not root_path.exists() or not root_path.is_dir():
            continue
        for pattern in patterns:
            for candidate in root_path.rglob(pattern):
                if candidate.is_file():
                    rel = str(candidate.relative_to(repo_root)).replace("\\", "/")
                    docs.append(rel)

    return sorted(set(docs))


def apply_legacy_policy(paths, front_id, front_rec, defaults, explicit_legacy_request):
    excluded = []
    filtered = []

    legacy_roots = defaults.get("legacy_roots", [])
    exclude_legacy = bool(defaults.get("legacy_excluded_by_default", True))

    legacy_front = front_rec.get("legacy_flag", False) or front_id in {"legacy", "uoemd"}

    for path in paths:
        normalized = normalize_path(path)
        is_legacy_path = any(
            normalized == lr or normalized.startswith(lr + "/")
            for lr in map(normalize_path, legacy_roots)
        )

        if exclude_legacy and is_legacy_path and not legacy_front and not explicit_legacy_request:
            excluded.append({"path": normalized, "reason": "legacy_excluded_by_default"})
        else:
            filtered.append(normalized)

    return sorted(set(filtered)), excluded


def main():
    parser = argparse.ArgumentParser(description="Select documentation targets by front and event type.")
    parser.add_argument("--front", required=True, help="front id")
    parser.add_argument("--event-type", required=True, help="event type from policy")
    parser.add_argument("--collab-mode", default="off", choices=["on", "off"], help="collab mode")
    parser.add_argument(
        "--registry",
        default="tools/skills/phideus-doc-maintainer/references/front_registry.yaml",
        help="front registry yaml",
    )
    parser.add_argument(
        "--policy",
        default="tools/skills/phideus-doc-maintainer/references/global_update_policy.yaml",
        help="global update policy yaml",
    )
    parser.add_argument("--repo-root", default=".", help="repo root")

    parser.add_argument("--experimental-advance", action="store_true")
    parser.add_argument("--technical-decision", action="store_true")
    parser.add_argument("--status-changed", action="store_true")
    parser.add_argument("--focus-changed", action="store_true")
    parser.add_argument("--doc-structure-changed", action="store_true")
    parser.add_argument("--trunk-milestone", action="store_true")
    parser.add_argument("--explicit-legacy-request", action="store_true")

    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    registry = load_yaml(repo_root / args.registry).get("fronts", {})
    policy = load_yaml(repo_root / args.policy)

    defaults = policy.get("defaults", {})
    events = policy.get("event_types", {})

    if args.event_type not in events:
        raise SystemExit(f"Unknown event type: {args.event_type}")

    event_cfg = events[args.event_type]
    front_id = args.front.strip()
    front_rec = registry.get(front_id)

    if front_rec is None:
        front_rec = infer_custom_front(front_id, repo_root)

    front_docs = []

    front_levels = event_cfg.get("front_levels", [])
    if "primary" in front_levels:
        front_docs.extend(front_rec.get("primary_docs", []))
    if "secondary" in front_levels:
        front_docs.extend(front_rec.get("secondary_docs", []))

    pattern_docs = extract_pattern_docs(front_rec.get("doc_roots", []), repo_root)
    front_docs.extend(pattern_docs)

    flags = {
        "experimental_advance": bool(args.experimental_advance),
        "technical_decision": bool(args.technical_decision),
        "status_changed": bool(args.status_changed),
        "focus_changed": bool(args.focus_changed),
        "doc_structure_changed": bool(args.doc_structure_changed),
        "trunk_milestone": bool(args.trunk_milestone),
    }

    for default_true in event_cfg.get("default_true_flags", []):
        if default_true in flags:
            flags[default_true] = True

    global_docs = []
    for rule in event_cfg.get("global_rules", []):
        doc = rule.get("doc")
        required = rule.get("require_any", [])
        if doc and any(flags.get(k, False) for k in required):
            global_docs.append(doc)

    front_docs, excluded_front = apply_legacy_policy(
        front_docs,
        front_id,
        front_rec,
        defaults,
        explicit_legacy_request=args.explicit_legacy_request,
    )
    global_docs, excluded_global = apply_legacy_policy(
        global_docs,
        front_id,
        front_rec,
        defaults,
        explicit_legacy_request=args.explicit_legacy_request,
    )

    output = {
        "front_id": front_id,
        "event_type": args.event_type,
        "collab_mode": args.collab_mode,
        "flags": flags,
        "front_docs": front_docs,
        "global_docs": global_docs,
        "excluded_by_policy": excluded_front + excluded_global,
    }

    print(json.dumps(output, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
