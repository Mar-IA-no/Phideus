#!/usr/bin/env python3
"""Validate Phideus' LLM wiki and optionally rebuild its machine catalog."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
WIKI = REPO / "Documents" / "05_WIKI"
EXEMPT = {"README.md", "SCHEMA.md", "index.md", "log.md"}
REQUIRED = {
    "schema_version",
    "id",
    "kind",
    "page_status",
    "front_status",
    "updated",
    "verified_at",
    "valid_at",
    "recorded_at",
    "evidence_commit",
    "source_paths",
    "depends_on",
    "tangents",
}
FRONT_REQUIRED = {
    "architecture_status",
    "experiment_status",
    "evidence_status",
    "decision_status",
}
LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
SOURCE_ID_RE = re.compile(r"\bSRC-[A-Z0-9-]+\b")
SHA_RE = re.compile(r"[0-9a-f]{40}")
WORK_UNIT_RANGE_RE = re.compile(r"\bP\d+\s*[–-]\s*P\d+\b")
SOURCE_REQUIRED = {
    "id",
    "title",
    "path",
    "authority",
    "evidence_regime",
    "mutable",
    "fronts",
}


def load_frontmatter(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return {}, text
    parts = text.split("---\n", 2)
    if len(parts) != 3:
        return {}, text
    data = yaml.safe_load(parts[1]) or {}
    return data, parts[2]


def title_from(body: str) -> str:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def validate_link(source: Path, target: str) -> bool:
    target = target.strip().split()[0]
    if target.startswith(("http://", "https://", "mailto:", "#")):
        return True
    target = target.split("#", 1)[0]
    if not target:
        return True
    return (source.parent / target).resolve().exists()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-catalog", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    pages: list[dict] = []
    ids: dict[str, Path] = {}
    index_text = (WIKI / "index.md").read_text(encoding="utf-8")

    for path in sorted(WIKI.rglob("*.md")):
        rel = path.relative_to(WIKI)
        meta, body = load_frontmatter(path)
        if rel.as_posix() not in EXEMPT:
            missing = REQUIRED - set(meta)
            if missing:
                errors.append(f"{rel}: faltan campos {sorted(missing)}")
            if meta.get("schema_version") != 1:
                errors.append(f"{rel}: schema_version debe ser 1")
            if not SHA_RE.fullmatch(str(meta.get("evidence_commit", ""))):
                errors.append(f"{rel}: evidence_commit no es un SHA completo")
            if meta.get("kind") == "front":
                missing_front = FRONT_REQUIRED - set(meta)
                if missing_front:
                    errors.append(
                        f"{rel}: faltan estados ortogonales {sorted(missing_front)}"
                    )
            page_id = meta.get("id")
            if page_id in ids:
                errors.append(f"ID duplicado {page_id}: {ids[page_id]} y {rel}")
            elif page_id:
                ids[page_id] = rel
            if rel.as_posix() not in index_text:
                errors.append(f"{rel}: página sustantiva ausente de index.md")
            for source_path in meta.get("source_paths", []):
                if not (REPO / source_path).exists():
                    errors.append(f"{rel}: source_path inexistente: {source_path}")
            pages.append(
                {
                    "id": page_id,
                    "path": rel.as_posix(),
                    "title": title_from(body),
                    "kind": meta.get("kind"),
                    "page_status": meta.get("page_status"),
                    "front_status": meta.get("front_status"),
                    "architecture_status": meta.get("architecture_status"),
                    "experiment_status": meta.get("experiment_status"),
                    "evidence_status": meta.get("evidence_status"),
                    "decision_status": meta.get("decision_status"),
                    "updated": str(meta.get("updated", "")),
                    "verified_at": str(meta.get("verified_at", "")),
                    "valid_at": str(meta.get("valid_at", "")),
                    "recorded_at": str(meta.get("recorded_at", "")),
                    "source_paths": meta.get("source_paths", []),
                    "depends_on": meta.get("depends_on", []),
                    "tangents": meta.get("tangents", []),
                }
            )

        for target in LINK_RE.findall(path.read_text(encoding="utf-8")):
            if not validate_link(path, target):
                errors.append(f"{rel}: enlace roto: {target}")
        for compact_range in WORK_UNIT_RANGE_RE.findall(body):
            errors.append(
                f"{rel}: rango compacto ambiguo {compact_range}; enumerar unidades ejecutadas"
            )

    registry = yaml.safe_load((WIKI / "sources.yaml").read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1:
        errors.append("sources.yaml: schema_version debe ser 1")
    if not SHA_RE.fullmatch(str(registry.get("evidence_commit", ""))):
        errors.append("sources.yaml: evidence_commit no es un SHA completo")
    source_ids: set[str] = set()
    for source in registry.get("sources", []):
        missing_source = SOURCE_REQUIRED - set(source)
        if missing_source:
            errors.append(
                f"sources.yaml: faltan campos {sorted(missing_source)} en {source.get('id')}"
            )
        sid = source.get("id")
        if sid in source_ids:
            errors.append(f"sources.yaml: ID duplicado {sid}")
        source_ids.add(sid)
        source_path = source.get("path")
        if not source_path or not (REPO / source_path).exists():
            errors.append(f"sources.yaml: path inexistente para {sid}: {source_path}")

    for page in pages:
        for relation in page["depends_on"] + page["tangents"]:
            if relation not in ids:
                errors.append(
                    f"{page['path']}: relación a ID de página inexistente: {relation}"
                )

    for path in sorted(WIKI.rglob("*.md")):
        rel = path.relative_to(WIKI)
        for source_id in SOURCE_ID_RE.findall(path.read_text(encoding="utf-8")):
            if source_id not in source_ids:
                errors.append(f"{rel}: source ID no registrado: {source_id}")

    catalog = {
        "schema_version": 1,
        "generated_from": "Documents/05_WIKI",
        "evidence_commit": registry.get("evidence_commit"),
        "pages": pages,
    }
    if args.write_catalog:
        (WIKI / "catalog.json").write_text(
            json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"FAIL: {len(errors)} error(es)")
        return 1

    print(f"PASS: {len(pages)} páginas, {len(source_ids)} fuentes, IDs y enlaces válidos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
