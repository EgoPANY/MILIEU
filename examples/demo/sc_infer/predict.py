"""Inference helpers for DualFork cell-cloud (deterministic + sampled cloud).

Inference readouts (deterministic decode and optional cloud sampling):
  - deterministic μ decode → d16 / dh
  - optional Monte-Carlo cloud sampling (n_cloud) with ``cloud_head`` selecting
    which head's samples form the cell cloud (product default: ``d16``)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .model import DualForkVAE


@torch.no_grad()
def predict_dual_det(
    model: DualForkVAE,
    X: np.ndarray,
    t_idx: np.ndarray,
    U: np.ndarray | None,
    Umask: np.ndarray | None,
    device: torch.device | str = "cpu",
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    """Deterministic (μ) decode for all rows → ``d16``, ``dh``, ``dfuse``."""
    model.eval()
    device = torch.device(device)
    model = model.to(device)
    outs_d16, outs_dh, outs_fuse = [], [], []
    n = len(X)
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        xb = torch.tensor(X[s:e], device=device)
        tb = torch.tensor(t_idx[s:e], device=device, dtype=torch.long)
        Ub = Umb = None
        if U is not None:
            Ub = torch.tensor(U[s:e], device=device)
            Umb = torch.tensor(Umask[s:e], device=device, dtype=torch.bool)
        _, d16, dh, dfuse, *_ = model(xb, tb, Ub, Umb, sample=False)
        outs_d16.append(d16.cpu().numpy())
        outs_dh.append(dh.cpu().numpy())
        outs_fuse.append(dfuse.cpu().numpy())
    return {
        "d16": np.concatenate(outs_d16, 0),
        "dh": np.concatenate(outs_dh, 0),
        "dfuse": np.concatenate(outs_fuse, 0),
    }


@torch.no_grad()
def predict_dual_cloud(
    model: DualForkVAE,
    X: np.ndarray,
    t_idx: np.ndarray,
    U: np.ndarray | None,
    Umask: np.ndarray | None,
    *,
    n_samples: int = 16,
    cloud_head: str = "d16",
    device: torch.device | str = "cpu",
    batch_size: int = 128,
) -> dict[str, Any]:
    """Sample a cell cloud and also return condition means of d16 / dh.

    Returns
    -------
    cloud : (n_cond, n_samples, G) from the selected head
    d16_cmean / dh_cmean / dfuse_cmean : (n_cond, G)
    """
    model.eval()
    device = torch.device(device)
    model = model.to(device)
    clouds, d16m, dhm, fusem = [], [], [], []
    n = len(X)
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        xb = torch.tensor(X[s:e], device=device)
        tb = torch.tensor(t_idx[s:e], device=device, dtype=torch.long)
        Ub = Umb = None
        if U is not None:
            Ub = torch.tensor(U[s:e], device=device)
            Umb = torch.tensor(Umask[s:e], device=device, dtype=torch.bool)
        samples = []
        d16_acc = dh_acc = fuse_acc = None
        for _ in range(n_samples):
            # Force sampling even in eval: temporarily flip training flag for reparam.
            was_training = model.training
            model.train()
            _, d16, dh, dfuse, *_ = model(xb, tb, Ub, Umb, sample=True)
            model.train(was_training)
            head = {"d16": d16, "dh": dh, "dfuse": dfuse}[cloud_head]
            samples.append(head.unsqueeze(1))
            d16_acc = d16 if d16_acc is None else d16_acc + d16
            dh_acc = dh if dh_acc is None else dh_acc + dh
            fuse_acc = dfuse if fuse_acc is None else fuse_acc + dfuse
        clouds.append(torch.cat(samples, dim=1).cpu().numpy())
        d16m.append((d16_acc / n_samples).cpu().numpy())
        dhm.append((dh_acc / n_samples).cpu().numpy())
        fusem.append((fuse_acc / n_samples).cpu().numpy())
    return {
        "cloud": np.concatenate(clouds, 0),
        "d16_cmean": np.concatenate(d16m, 0),
        "dh_cmean": np.concatenate(dhm, 0),
        "dfuse_cmean": np.concatenate(fusem, 0),
        "n_samples": int(n_samples),
        "cloud_head": cloud_head,
    }
