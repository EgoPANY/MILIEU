"""DualForkVAE — inference stub for the public review demo (cloud fine-tune checkpoint).

Heads: low-dimensional PCA fork (Δ16) and high-dimensional PCA fork (Δh).
Checkpoints for this demo use the cell-level fine-tuned format (``demo_cloud_ft.pt``).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def build_enc_to_h(d_in, hidden=512, layers=3):
    """Build the shared MLP encoder: d_in → hidden × ``layers`` (ReLU + Dropout).

    Locked product uses ``d_in=1280`` (ESM-style condition embedding).
    """
    mods, prev = [], d_in
    for _ in range(layers):
        mods += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(0.1)]
        prev = hidden
    return nn.Sequential(*mods)


class DualForkVAE(nn.Module):
    """Shared CrossAttn trunk + mid-fork dual heads (Δ16 / Δh).

    Parameters
    ----------
    V16, Vh :
        PCA bases registered as buffers; decoded as ``z @ V``.
        Shapes ``(k16, G)`` and ``(kh, G)``; ``G`` is dataset-bound (not shared across species).
    pool_dim :
        Width of the cell-context pool embedding. Zero disables context cross-attention.
    mid_fork :
        Product default ``True``: separate towers before μ/σ heads.
    n_pert / gene_prior :
        Optional perturbation embeddings / logfc prior (present in some ckpts).
    """

    def __init__(
        self, V16, Vh, pool_dim=0, hidden=512, layers=3, n_heads=4, mid_fork=True,
        deep_panel=False, n_pert=0, gene_prior=None,
    ):
        super().__init__()
        # Condition encoder (fixed 1280-d input in the locked recipe).
        self.enc = build_enc_to_h(1280, hidden=hidden, layers=layers)
        self.register_buffer("V16", torch.tensor(V16, dtype=torch.float32))
        self.register_buffer("Vh", torch.tensor(Vh, dtype=torch.float32))
        k16, kh = V16.shape[0], Vh.shape[0]
        self.mid_fork = mid_fork
        self.n_pert = int(n_pert)
        self.use_pool = pool_dim > 0
        if self.use_pool:
            # MultiheadAttention requires embed_dim divisible by n_heads.
            d_attn = int(((pool_dim + n_heads - 1) // n_heads) * n_heads)
            self.kv_proj = nn.Identity() if d_attn == pool_dim else nn.Linear(pool_dim, d_attn)
            self.q_proj = nn.Linear(1280, d_attn)
            self.attn = nn.MultiheadAttention(d_attn, n_heads, batch_first=True)
            self.fuse = nn.Linear(d_attn, hidden)
            nn.init.zeros_(self.fuse.weight)
            nn.init.zeros_(self.fuse.bias)
            self.alpha_attn = nn.Parameter(torch.tensor(0.1))
        if mid_fork:
            if deep_panel:
                self.tower16 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1))
                self.tower_h = nn.Sequential(
                    nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
                )
            else:
                self.tower16 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1))
                self.tower_h = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1))
        else:
            self.tower16 = nn.Identity()
            self.tower_h = nn.Identity()
        if self.n_pert > 0:
            self.emb16 = nn.Embedding(self.n_pert, hidden)
            self.emb_h = nn.Embedding(self.n_pert, hidden)
            nn.init.zeros_(self.emb16.weight)
            nn.init.zeros_(self.emb_h.weight)
        if gene_prior is not None:
            self.register_buffer("gene_prior", torch.as_tensor(gene_prior, dtype=torch.float32))
            self.prior_scale = nn.Parameter(torch.zeros(()))
        self.mu16 = nn.Linear(hidden, k16)
        self.logvar16 = nn.Linear(hidden, k16)
        self.mu_h = nn.Linear(hidden, kh)
        self.logvar_h = nn.Linear(hidden, kh)
        # Learnable fuse weight between Δ16 and Δh (reported, not product primary).
        self.fuse_logit = nn.Parameter(torch.tensor(0.0))

    def encode_feat(self, x, U=None, Umask=None):
        """Encode condition ``x`` and optionally fuse a cell pool ``U`` via Cross-Attn."""
        h = self.enc(x)
        if self.use_pool and U is not None:
            Up = self.kv_proj(U)
            q = self.q_proj(x).unsqueeze(1)
            # ``Umask`` True = valid token; MHA expects True = *pad*, so we invert.
            c, _ = self.attn(q, Up, Up, key_padding_mask=~Umask, need_weights=False)
            h = h + self.alpha_attn * self.fuse(c.squeeze(1))
        return h

    def forward(self, x, t_idx, U=None, Umask=None, sample=True, stopgrad_panel=False):
        """Return (skd16, d16, dh, dfuse, mu16, lv16, muh, lvh, fuse_w).

        ``skd16`` is the on-target gene entry of ``d16`` selected by ``t_idx``.
        """
        h = self.encode_feat(x, U, Umask)
        h16 = self.tower16(h)
        hp = h.detach() if stopgrad_panel else h
        hh = self.tower_h(hp)
        if self.n_pert > 0:
            h16 = h16 + self.emb16(t_idx)
            hh = hh + self.emb_h(t_idx)
        mu16, lv16 = self.mu16(h16), self.logvar16(h16).clamp(-8, 8)
        muh, lvh = self.mu_h(hh), self.logvar_h(hh).clamp(-8, 8)
        if sample and self.training:
            z16 = mu16 + torch.randn_like(mu16) * torch.exp(0.5 * lv16)
            zh = muh + torch.randn_like(muh) * torch.exp(0.5 * lvh)
        else:
            z16, zh = mu16, muh
        d16 = z16 @ self.V16
        dh = zh @ self.Vh
        if hasattr(self, "gene_prior"):
            dh = dh + self.prior_scale * self.gene_prior
        w = torch.sigmoid(self.fuse_logit)
        dfuse = w * d16 + (1.0 - w) * dh
        b = torch.arange(x.size(0), device=x.device)
        skd16 = d16[b, t_idx]
        return skd16, d16, dh, dfuse, mu16, lv16, muh, lvh, float(w.detach())
