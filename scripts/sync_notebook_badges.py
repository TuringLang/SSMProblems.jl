#!/usr/bin/env python3
"""Normalize example notebook top-cell badges.

Usage:
  python scripts/sync_notebook_badges.py <notebook.ipynb> [more.ipynb ...]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = "TuringLang/SSMProblems.jl"


def source_to_text(source: object) -> str:
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    if isinstance(source, str):
        return source
    return ""


def text_to_source(text: str) -> list[str]:
    lines = text.splitlines()
    if not lines:
        return []
    return [f"{line}\n" for line in lines[:-1]] + [lines[-1]]


def title_from_slug(slug: str) -> str:
    return " ".join(word.capitalize() for word in slug.replace("_", "-").split("-"))


def is_badge_line(line: str) -> bool:
    return (
        "colab.research.google.com" in line
        or "View%20Source-GitHub" in line
        or "Example%20Page-Docs" in line
    )


def notebook_urls(path: Path) -> tuple[str, str, str]:
    parts = path.parts
    try:
        pkg_idx = parts.index("GeneralisedFilters")
    except ValueError:
        try:
            pkg_idx = parts.index("SSMProblems")
        except ValueError as exc:
            raise ValueError(f"notebook path does not include known package dir: {path}") from exc

    pkg = parts[pkg_idx]
    rel_path = "/".join(parts[pkg_idx:])

    if len(parts) < pkg_idx + 4 or parts[pkg_idx + 1] != "examples":
        raise ValueError(f"notebook path is not under {pkg}/examples/: {path}")

    slug = parts[pkg_idx + 2]
    colab = f"https://colab.research.google.com/github/{REPO}/blob/main/{rel_path}"
    source = f"https://github.com/{REPO}/blob/main/{rel_path}"
    docs = f"https://turinglang.org/SSMProblems.jl/{pkg}/dev/examples/{slug}/"
    return colab, source, docs


def build_badge_line(colab: str, source: str, docs: str) -> str:
    return (
        f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab}) "
        f"[![View Source](https://img.shields.io/badge/View%20Source-GitHub-181717?logo=github)]({source}) "
        f"[![Example Page](https://img.shields.io/badge/Example%20Page-Docs-0A7F2E)]({docs})"
    )


def normalize_notebook(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    data = json.loads(original)
    cells = data.get("cells", [])
    if not isinstance(cells, list):
        raise ValueError(f"notebook cells are invalid in {path}")

    colab, source, docs = notebook_urls(path)
    badges = build_badge_line(colab, source, docs)

    if not cells or cells[0].get("cell_type") != "markdown":
        slug = path.parent.name
        title = f"# {title_from_slug(slug)}"
        top_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": text_to_source(f"{title}\n\n{badges}"),
        }
        cells.insert(0, top_cell)
        data["cells"] = cells
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return True

    first = cells[0]
    source_text = source_to_text(first.get("source", []))
    lines = source_text.splitlines()

    title = None
    for idx, line in enumerate(lines):
        if re.match(r"^#\s+", line):
            title = line.strip()
            lines = lines[idx + 1 :]
            break

    if title is None:
        title = f"# {title_from_slug(path.parent.name)}"

    body = [line for line in lines if not is_badge_line(line)]
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()

    new_lines = [title, "", badges]
    if body:
        new_lines += [""] + body

    new_source = text_to_source("\n".join(new_lines))
    if first.get("source") == new_source:
        return False

    first["source"] = new_source
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return True


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: sync_notebook_badges.py <notebook.ipynb> [more.ipynb ...]", file=sys.stderr)
        return 2

    changed: list[str] = []
    failed = False

    for raw in argv[1:]:
        path = Path(raw)
        if not path.exists():
            continue

        try:
            if normalize_notebook(path):
                changed.append(str(path))
        except Exception as exc:
            print(f"failed to sync badges for {path}: {exc}", file=sys.stderr)
            failed = True

    if changed:
        print("synchronized notebook badges:")
        for filename in changed:
            print(f"  - {filename}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
