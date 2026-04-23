#!/usr/bin/env python3
"""Check that the training stack dependencies are importable."""

from __future__ import annotations

import importlib
import sys
from typing import List, Tuple

MODULES = [
    "torch",
    "transformers",
    "trl",
    "unsloth",
    "matplotlib",
    "httpx",
]


def check_module(name: str) -> Tuple[bool, str]:
    try:
        importlib.import_module(name)
        return True, "imported"
    except Exception as exc:  # pragma: no cover - exercised in user environments
        return False, f"{exc.__class__.__name__}: {exc}"


def main() -> int:
    print("AdaptiveSRE dependency import check")
    print("=" * 40)

    failures: List[str] = []
    for module_name in MODULES:
        ok, detail = check_module(module_name)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {module_name}: {detail}")
        if not ok:
            failures.append(module_name)

    print("=" * 40)
    if failures:
        print("FAIL: missing or broken imports -> " + ", ".join(failures))
        return 1

    print("PASS: all requested modules imported successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
