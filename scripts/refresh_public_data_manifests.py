#!/usr/bin/env python3
"""Refresh SHA256SUMS / MANIFEST.json for tracked public data paths."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PUBLIC_MANUSCRIPT_CSV = [
    "data/manuscript/sc_gene_ood_adamson_jurkat_hepg2_multiseed_metrics.csv",
    "data/manuscript/spatial_benchmark_radar_metrics_hippocampus_perturbmap_16seed_multiseed.csv",
    "data/manuscript/spatial_benchmark_mmd_hippocampus_perturbmap_16seed_multiseed.csv",
    "data/manuscript/spatial_hippocampus_vcc_stk39_fasn_clu_ridge_mean_train_multiseed.csv",
    "data/manuscript/figure_significance_ours_vs_comparator_panel_summary.csv",
    "data/manuscript/figure_significance_paired_seed_multiseed_tests.csv",
]

GLOBS = [
    "data/manuscript/README.md",
    "data/figure_tables/**/*.csv",
    "data/figure_tables/README.md",
    "data/figure_tables/MANIFEST.json",
    "data/demo/sc/README.json",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    files: list[Path] = [ROOT / rel for rel in PUBLIC_MANUSCRIPT_CSV]
    for pat in GLOBS:
        files.extend(sorted(ROOT.glob(pat)))
    files = [p for p in files if p.is_file() and "SHA256SUMS" not in p.name]

    manifest_files = []
    row_counts: dict[str, int] = {}
    sum_lines: list[str] = []
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        digest = sha256_file(p)
        if p.suffix == ".csv":
            with p.open("r", encoding="utf-8", errors="replace") as f:
                n = sum(1 for _ in f) - 1
            row_counts[rel] = max(n, 0)
        manifest_files.append({"file": rel, "sha256": digest})
        if rel.startswith("data/manuscript/") and p.suffix == ".csv":
            sum_lines.append(f"{digest}  {p.name}")

    ms_dir = ROOT / "data/manuscript"
    (ms_dir / "SHA256SUMS").write_text("\n".join(sorted(sum_lines)) + "\n")
    meta = {
        "exported_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": "public review release",
        "row_counts": {k: v for k, v in row_counts.items() if k.startswith("data/manuscript/")},
        "files": [e for e in manifest_files if e["file"].startswith("data/manuscript/")],
    }
    (ms_dir / "MANIFEST.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Updated manifests for {len(files)} files")


if __name__ == "__main__":
    main()
