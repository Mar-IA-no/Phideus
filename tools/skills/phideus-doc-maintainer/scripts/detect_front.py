#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path

import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def norm(text):
    return str(text).lower().strip()


def normalize_path(path):
    return str(path).replace("\\", "/").strip("./")


def tokenize_hints(raw_hints):
    tokens = []
    for item in raw_hints:
        for part in str(item).split(","):
            p = norm(part)
            if p:
                tokens.append(p)
    return sorted(set(tokens))


def starts_with_root(path, root):
    p = normalize_path(path)
    r = normalize_path(root)
    return p == r or p.startswith(r + "/")


def read_text_if_exists(path):
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return ""


def score_front(front_id, front_rec, hints, paths, weights, repo_root):
    score = 0
    evidence = []

    alias_tag_space = [norm(x) for x in front_rec.get("aliases", []) + front_rec.get("tags", [])]

    matched_hints = [h for h in hints if any(h in at or at in h for at in alias_tag_space)]
    if matched_hints:
        score += int(weights["hint_match"])
        evidence.append(
            f"+{weights['hint_match']} hint_match ({', '.join(sorted(set(matched_hints))[:4])})"
        )

    doc_roots = front_rec.get("doc_roots", [])
    code_roots = front_rec.get("code_roots", [])

    matched_doc_roots = set()
    matched_code_roots = set()

    for path in paths:
        for root in doc_roots:
            if starts_with_root(path, root):
                matched_doc_roots.add(root)
        for root in code_roots:
            if starts_with_root(path, root):
                matched_code_roots.add(root)

    if matched_doc_roots:
        add = int(weights["path_doc_root"]) * len(matched_doc_roots)
        score += add
        evidence.append(f"+{add} path_doc_root ({', '.join(sorted(matched_doc_roots))})")

    if matched_code_roots:
        add = int(weights["path_code_root"]) * len(matched_code_roots)
        score += add
        evidence.append(f"+{add} path_code_root ({', '.join(sorted(matched_code_roots))})")

    if hints:
        docs_for_match = front_rec.get("primary_docs", [])[:2]
        for rel_doc in docs_for_match:
            abs_doc = repo_root / rel_doc
            content = read_text_if_exists(abs_doc)
            if not content:
                continue
            if any(h in content for h in hints):
                score += int(weights["roadmap_state_match"])
                evidence.append(f"+{weights['roadmap_state_match']} roadmap_state_match ({rel_doc})")
                break

    return score, evidence


def detect_from_fallback(registry, fallback_file, repo_root):
    content = read_text_if_exists(repo_root / fallback_file)
    if not content:
        return None, []

    best_front = None
    best_count = 0

    for front_id, front_rec in registry.items():
        terms = set([norm(front_id)] + [norm(x) for x in front_rec.get("aliases", []) + front_rec.get("tags", [])])
        count = sum(content.count(t) for t in terms if t and len(t) >= 3)
        if count > best_count:
            best_count = count
            best_front = front_id

    if best_front:
        return best_front, [f"fallback_focus_match ({fallback_file})"]

    return None, []


def main():
    parser = argparse.ArgumentParser(description="Detect active documentation front for Phideus.")
    parser.add_argument("--front", default="auto", help="auto|known_front|custom_front")
    parser.add_argument("--hints", nargs="*", default=[], help="Hint tokens from user request")
    parser.add_argument("--paths", nargs="*", default=[], help="Relevant changed/read paths")
    parser.add_argument(
        "--registry",
        default="tools/skills/phideus-doc-maintainer/references/front_registry.yaml",
        help="Path to front registry YAML",
    )
    parser.add_argument(
        "--rules",
        default="tools/skills/phideus-doc-maintainer/references/detection_rules.yaml",
        help="Path to detection rules YAML",
    )
    parser.add_argument("--repo-root", default=".", help="Repo root")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    registry_data = load_yaml(repo_root / args.registry)
    rules_data = load_yaml(repo_root / args.rules)

    registry = registry_data.get("fronts", {})
    weights = rules_data.get("weights", {})
    ambiguity_threshold = int(rules_data.get("ambiguity_threshold", 2))
    fallback_file = rules_data.get("fallback", {}).get("file", "Documents/Proyecto_Estado_Actual.md")

    hints = tokenize_hints(args.hints)
    paths = [normalize_path(p) for p in args.paths]

    result = {
        "front_id": None,
        "confidence": None,
        "score": 0,
        "runner_up": None,
        "evidence": [],
        "scores": {},
    }

    requested_front = norm(args.front)

    if requested_front != "auto":
        if requested_front in registry:
            result["front_id"] = requested_front
            result["confidence"] = "high"
            result["score"] = int(weights.get("override", 4))
            result["evidence"] = [f"+{weights.get('override', 4)} manual_override"]
        else:
            result["front_id"] = requested_front
            result["confidence"] = "high"
            result["score"] = int(weights.get("override", 4))
            result["evidence"] = [f"+{weights.get('override', 4)} manual_override_custom"]

        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0

    scored = []
    for front_id, front_rec in registry.items():
        score, evidence = score_front(front_id, front_rec, hints, paths, weights, repo_root)
        scored.append((front_id, score, evidence))
        result["scores"][front_id] = score

    scored.sort(key=lambda x: x[1], reverse=True)
    top_front, top_score, top_evidence = scored[0]
    runner = scored[1] if len(scored) > 1 else None

    if top_score == 0:
        fallback_front, fallback_evidence = detect_from_fallback(registry, fallback_file, repo_root)
        if fallback_front:
            result["front_id"] = fallback_front
            result["confidence"] = "low"
            result["score"] = 0
            result["evidence"] = fallback_evidence
            print(json.dumps(result, indent=2, ensure_ascii=True))
            return 0

    if runner is not None and (top_score - runner[1]) < ambiguity_threshold:
        result["front_id"] = top_front
        result["confidence"] = "ambiguous"
        result["score"] = top_score
        result["runner_up"] = {"front_id": runner[0], "score": runner[1]}
        result["evidence"] = top_evidence
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 2

    if top_score >= 6:
        confidence = "high"
    elif top_score >= 3:
        confidence = "medium"
    else:
        confidence = "low"

    result["front_id"] = top_front
    result["confidence"] = confidence
    result["score"] = top_score
    result["evidence"] = top_evidence
    if runner is not None:
        result["runner_up"] = {"front_id": runner[0], "score": runner[1]}

    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
