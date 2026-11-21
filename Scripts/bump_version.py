"""Raises the version in version.py by one step.

Every digit runs from 0 to 9 and carries to the left: 0.0.9 -> 0.1.0, 0.9.9 -> 1.0.0.
Run from the pre-commit hook; can also be run by hand: `python Scripts/bump_version.py`.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

VERSION_FILE = Path(__file__).resolve().parents[1] / "version.py"
PATTERN = re.compile(r'__version__ = "(\d+)\.(\d+)\.(\d+)"')


def next_version(version: str) -> str:
    major, minor, patch = (int(part) for part in version.split("."))
    patch += 1
    if patch > 9:
        patch, minor = 0, minor + 1
    if minor > 9:
        minor, major = 0, major + 1
    return f"{major}.{minor}.{patch}"


def bump(path: Path = VERSION_FILE) -> str:
    text = path.read_text(encoding="utf-8")
    match = PATTERN.search(text)
    if not match:
        raise SystemExit(f"No __version__ = \"x.y.z\" found in {path}")
    old = ".".join(match.groups())
    new = next_version(old)
    path.write_text(text.replace(match.group(0), f'__version__ = "{new}"'), encoding="utf-8")
    return new


if __name__ == "__main__":
    print(bump())
    sys.exit(0)
