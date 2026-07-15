#!/usr/bin/env python3
"""Fail CI when repository content could contain clinical data or secrets."""

from __future__ import annotations

import sys
from pathlib import Path

from ai_clinician.repository_safety import scan


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    findings = scan(root)
    if findings:
        print("Repository safety check failed:", file=sys.stderr)
        for finding in findings:
            print(f"- {finding}", file=sys.stderr)
        return 1
    print("Repository safety check passed: no forbidden tracked artifacts detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
