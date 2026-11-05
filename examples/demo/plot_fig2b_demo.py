#!/usr/bin/env python3
"""Replay one main-text bar panel from shipped figure_tables CSV (no MILIEU training code)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Plot CSV (default: data/figure_tables/main_fig2/Fig2B_ed5_essential_multiseed_plot_data.csv)",
    )
    ap.add_argument("--metric", default="skd_d16", help="metric_key column filter")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    csv_path = args.csv or (
        root / "data/figure_tables/main_fig2/Fig2B_ed5_essential_multiseed_plot_data.csv"
    )
    df = pd.read_csv(csv_path)
    sub = df[(df["metric_key"] == args.metric) & (df["seed"] == args.seed)].copy()
    if sub.empty:
        raise SystemExit(f"No rows for metric={args.metric!r} seed={args.seed}")

    sub = sub.sort_values("value", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(sub["method_label"], sub["value"], color="#4C72B0")
    ax.set_ylabel(sub["metric_label"].iloc[0])
    ax.set_title(f"{sub['dataset_label'].iloc[0]} · seed {args.seed}")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()

    out = args.out or (root / "data/demo/sc/fig2b_essential_skd_seed0_demo.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
