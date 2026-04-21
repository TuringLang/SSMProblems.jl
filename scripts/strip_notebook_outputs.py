#!/usr/bin/env python3
"""Strip execution state from Jupyter notebooks.

Usage:
  python scripts/strip_notebook_outputs.py <notebook.ipynb> [more.ipynb ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def strip_notebook(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    data = json.loads(original)
    changed = False

    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True

        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True

    metadata = data.get("metadata")
    if isinstance(metadata, dict) and "widgets" in metadata:
        del metadata["widgets"]
        changed = True

    if not changed:
        return False

    path.write_text(
        json.dumps(data, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return True


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: strip_notebook_outputs.py <notebook.ipynb> [more.ipynb ...]", file=sys.stderr)
        return 2

    failed = False
    changed_files: list[str] = []

    for raw in argv[1:]:
        path = Path(raw)
        if not path.exists():
            continue

        try:
            if strip_notebook(path):
                changed_files.append(str(path))
        except Exception as exc:  # pragma: no cover - defensive for hook usage
            print(f"failed to strip {path}: {exc}", file=sys.stderr)
            failed = True

    if changed_files:
        print("stripped notebook outputs:")
        for filename in changed_files:
            print(f"  - {filename}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
