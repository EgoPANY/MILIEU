#!/usr/bin/env python3
"""Public review demo: infer-only on shipped te_cond.npz + demo_cloud_ft.pt (no training)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "data" / "demo" / "sc"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sc_infer.model import DualForkVAE  # noqa: E402
from sc_infer.predict import predict_dual_cloud  # noqa: E402


def _pearson_rows(pred: np.ndarray, true: np.ndarray) -> float:
    corrs = []
    for i in range(len(true)):
        a, b = pred[i], true[i]
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            continue
        corrs.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(corrs)) if corrs else float("nan")


def main() -> int:
    pack_path = DEMO / "te_cond.npz"
    ckpt_path = DEMO / "demo_cloud_ft.pt"
    for p in (pack_path, ckpt_path):
        if not p.is_file():
            print(f"Missing {p}", file=sys.stderr)
            return 1

    pack = dict(np.load(pack_path, allow_pickle=True))
    X = np.asarray(pack["X"], np.float32)
    t = np.asarray(pack["t"], np.int64)
    U = np.asarray(pack["U"], np.float32) if "U" in pack else None
    Um = np.asarray(pack["Umask"]).astype(bool) if "Umask" in pack else None

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pool_dim = int(blob.get("pool_dim") or (U.shape[-1] if U is not None else 0))
    model = DualForkVAE(
        blob["V16"],
        blob["Vh"],
        pool_dim=pool_dim,
        mid_fork=True,
        deep_panel=False,
        n_pert=int(blob["state"]["emb16.weight"].shape[0]) if "emb16.weight" in blob["state"] else 0,
    )
    model.load_state_dict(blob["state"], strict=False)
    model.eval()

    sx_m, sx_s = blob.get("sx_mean"), blob.get("sx_scale")
    if sx_m is not None and sx_s is not None:
        X = (X - np.asarray(sx_m, np.float32)) / np.asarray(sx_s, np.float32).clip(1e-6)
    if U is not None and blob.get("pool_mean") is not None:
        pm = np.asarray(blob["pool_mean"], np.float32)
        ps = np.asarray(blob.get("pool_std", blob.get("pool_scale", 1.0)), np.float32)
        U = (U - pm) / np.clip(ps, 1e-6, None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = predict_dual_cloud(model, X, t, U, Um, n_samples=8, cloud_head="d16", device=device)
    pred = out["d16_cmean"].astype(np.float32)

    out_npy = DEMO / "demo_pred.npy"
    np.save(out_npy, pred)

    score = {"pred_shape": list(pred.shape), "device": device}
    if "D" in pack:
        true = np.asarray(pack["D"], np.float32)
        score["mse_mean"] = float(np.mean((pred - true) ** 2))
        score["pearson_delta_mean"] = _pearson_rows(pred, true)

    score_path = DEMO / "demo_score.json"
    score_path.write_text(json.dumps(score, indent=2) + "\n")
    print(json.dumps(score, indent=2))
    print(f"wrote {out_npy} and {score_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
