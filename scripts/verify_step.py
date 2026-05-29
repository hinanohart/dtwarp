#!/usr/bin/env python3
"""Local verification helper: run the same gates CI runs (pytest + ruff + mypy + honest-marketing).

Usage: python scripts/verify_step.py   (run from an activated venv)
Exit code 0 iff every gate passes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

GATES = [
    (["python", "-m", "pytest", "-q"], "pytest"),
    (["ruff", "check", "src", "tests"], "ruff"),
    (["mypy", "src/dtwarp"], "mypy --strict"),
    (["python", "scripts/check_honest_marketing.py"], "honest-marketing"),
]


def main() -> int:
    failed: list[str] = []
    for cmd, name in GATES:
        print(f"\n=== {name} ===", flush=True)
        rc = subprocess.run(cmd, cwd=REPO).returncode
        if rc != 0:
            failed.append(name)
    print("\n" + ("ALL GATES PASSED" if not failed else f"FAILED GATES: {', '.join(failed)}"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
