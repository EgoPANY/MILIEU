#!/usr/bin/env python3
"""Verify SHA256SUMS and MANIFEST row counts for data/manuscript/."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "data" / "manuscript"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    manifest_path = MANUSCRIPT / "MANIFEST.json"
    sums_path = MANUSCRIPT / "SHA256SUMS"
    if not manifest_path.is_file() or not sums_path.is_file():
        print("missing MANIFEST.json or SHA256SUMS", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = 0
    for entry in manifest.get("files", []):
        rel = entry["file"]
        path = ROOT / rel
        if not path.is_file():
            print(f"missing: {rel}")
            errors += 1
            continue
        got = sha256_file(path)
        if got != entry["sha256"]:
            print(f"sha256 mismatch: {rel}")
            errors += 1
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        path = MANUSCRIPT / name.strip()
        if sha256_file(path) != digest:
            print(f"SHA256SUMS mismatch: {name}")
            errors += 1
    if errors:
        print(f"FAILED ({errors} issues)")
        return 1
    print("OK:", len(manifest.get("files", [])), "files,", manifest.get("row_counts", {}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
