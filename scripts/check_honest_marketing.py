#!/usr/bin/env python3
"""Honest-marketing gate (CI-enforced).

Scans the marketing surface (README.md) for unsupported performance claims and for
hand-written benchmark numbers that must instead come from the eval results JSON. Implemented
in Python (not shell grep) to avoid BRE/ERE pitfalls, and self-tests against a negative fixture
so the check is proven to actually fire.

Exit code 0 = clean, 1 = violation. The scan deliberately targets only README.md (the marketing
surface); source code may use words like "permanently" in a purely technical sense.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"
RESULTS_DIR = REPO / "src" / "dtwarp" / "eval" / "results"

# Unsupported performance / absolutist marketing claims (case-insensitive).
BANNED = re.compile(
    r"\b(outperform(s|ed|ing)?|state[- ]of[- ]the[- ]art|sota|superior|"
    r"fastest|best[- ]in[- ]class|world[- ]class|unmatched|guarantee[sd]?\s+(better|improv))\b",
    re.IGNORECASE,
)

# A hard-written benchmark number is a percentage or an "Nx" speed-up sitting in prose.
NUMBER_CLAIM = re.compile(
    r"(\d+(\.\d+)?\s*%|\b\d+(\.\d+)?\s*x\s+(faster|better|higher|lower))", re.IGNORECASE
)

# Allowed numeric tokens that are config examples / versions, not claims.
ALLOWED_NUMBER_CONTEXT = re.compile(r"(gamma|blend|version|0\.1\.0|python|>=|\bv?\d+\.\d+)", re.IGNORECASE)


def scan_text(text: str, results_exist: bool) -> list[str]:
    violations: list[str] = []
    for i, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith("<!--"):
            continue  # placeholder / comment line
        m = BANNED.search(line)
        if m:
            violations.append(f"README.md:{i}: banned performance claim {m.group(0)!r}: {line.strip()}")
        if not results_exist:
            n = NUMBER_CLAIM.search(line)
            if n and not ALLOWED_NUMBER_CONTEXT.search(line):
                violations.append(
                    f"README.md:{i}: hand-written benchmark number {n.group(0)!r} but no eval "
                    f"results JSON exists yet: {line.strip()}"
                )
    return violations


def _self_test() -> None:
    """Negative fixture: the scanner MUST fire on a known-bad string."""
    bad = "dtwarp outperforms the baseline and is 42% better than ACT."
    hits = scan_text(bad, results_exist=False)
    if not hits:
        print("FATAL: honest-marketing self-test failed — scanner did not catch the negative fixture.")
        sys.exit(2)


def main() -> int:
    _self_test()
    if not README.exists():
        print("README.md not found")
        return 1
    results_exist = RESULTS_DIR.exists() and any(RESULTS_DIR.glob("*.json"))
    violations = scan_text(README.read_text(encoding="utf-8"), results_exist)
    if violations:
        print("Honest-marketing check FAILED:")
        for v in violations:
            print("  -", v)
        return 1
    print(f"Honest-marketing check passed (results_json_present={results_exist}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
