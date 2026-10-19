"""Stress-Invariant Feature Extractor Phi_theta with multi-scale regime conditioning (FiLM).

    tokens  = [PatchTST(price window), MLP(liq), Conv1d(LOB window)]          -> fuse -> h in R^d
    FiLM    : h <- h * (1 + tanh(g(c))) + b(c), c = multi-scale context prior  (volatility term structure, depth, spread,
              cross-asset correlation, IV skew). The context modulates *sensitivity*, it is not a shortcut: the invariance
              penalty is applied to the classifier on top of the modulated representation, so stress-specific spurious
              directions are still penalised. Zero-initialised so training starts from the unconditioned encoder.
"""
from __future__ import annotations
from typing import Any, Dict
import torch
import torch.nn as nn

from warn2trade.models.encoders import PatchTSEncoder, LOBConvEncoder


class FiLM(nn.Module):
    def __init__(self, ctx_dim: int, d: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(ctx_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 2 * d))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        g, b = self.net(ctx).chunk(2, dim=-1)
        return h * (1.0 + torch.tanh(g)) + b


class InvariantEncoder(nn.Module):
    def __init__(self, dims: Dict[str, int], cfg: Dict[str, Any]):
        super().__init__()
        d = int(cfg.get("d_model", 128))
        self.d_model = d
        drop = float(cfg.get("dropout", 0.1))
        self.price_enc = PatchTSEncoder(dims["price"], d, int(cfg.get("patch_len", 8)), int(cfg.get("stride", 4)),
                                        int(cfg.get("price_layers", 2)), int(cfg.get("n_heads", 4)), drop)
        self.liq_enc = nn.Sequential(nn.Linear(dims["liq"], d), nn.GELU())
        self.use_lob = bool(cfg.get("use_lob", True)) and dims.get("lob", 0) > 0
        if self.use_lob:
            self.lob_enc = LOBConvEncoder(dims["lob"], d, dropout=drop)
        n_tok = 2 + int(self.use_lob)
        self.fuse = nn.Sequential(nn.Linear(n_tok * d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, d))
        self.use_ctx = bool(cfg.get("use_ctx", True)) and dims.get("ctx", 0) > 0
        if self.use_ctx:
            self.film = FiLM(dims["ctx"], d, int(cfg.get("film_hidden", 64)), drop)
        self.norm = nn.LayerNorm(d)
        self.register_buffer("in_gate", torch.ones(dims["price"]))      # stress-extrapolated channel gate (models/gate.py); 1 = off
        Fp = dims["price"]
        # v4 adaptive suppression: g_c(s_t) = clip((a_c + b_c log s_t) / beta0_c, 0, 1) for flagged channels, applied at inference only
        self.register_buffer("ad_a", torch.zeros(Fp)); self.register_buffer("ad_b", torch.zeros(Fp))
        self.register_buffer("ad_beta0", torch.ones(Fp)); self.register_buffer("ad_mask", torch.zeros(Fp, dtype=torch.bool))
        self.register_buffer("ad_mu", torch.zeros(Fp))
        self.adaptive = bool(cfg.get("adaptive_gate", False))
        self.aug_p = float(cfg.get("aug_channel_p", 0.0))

    def set_gate(self, g) -> None:
        self.in_gate.copy_(torch.as_tensor(g, dtype=self.in_gate.dtype))

    def set_adaptive(self, info) -> None:
        f = lambda k, dt=torch.float32: torch.as_tensor(info[k], dtype=dt)
        self.ad_a.copy_(f("intercept")); self.ad_b.copy_(f("trend")); self.ad_beta0.copy_(f("beta_calm"))
        self.ad_mask.copy_(f("mask", torch.bool)); self.ad_mu.copy_(f("mu"))

    def adaptive_gate(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        s = batch.get("stress_score")
        ls = torch.log(s.clamp_min(1e-3)) if s is not None else torch.zeros(batch["price"].shape[0], device=self.ad_a.device)
        beta0 = torch.where(self.ad_beta0.abs() > 1e-6, self.ad_beta0, torch.ones_like(self.ad_beta0))
        g = ((self.ad_a + self.ad_b * ls[:, None]) / beta0).clamp(0.0, 1.0)            # (B, F)
        return torch.where(self.ad_mask, g, torch.ones_like(g))

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["price"] * self.in_gate
        mu = self.ad_mu.view(1, 1, -1)
        if self.training and self.aug_p > 0:
            B, _, F = x.shape
            g = torch.where(torch.rand(B, 1, F, device=x.device) < self.aug_p, torch.rand(B, 1, F, device=x.device), torch.ones(B, 1, F, device=x.device))
            x = mu + g * (x - mu)
        elif (not self.training) and self.adaptive and bool(self.ad_mask.any()):
            x = mu + self.adaptive_gate(batch)[:, None, :] * (x - mu)
        toks = [self.price_enc(x), self.liq_enc(batch["liq"])]
        if self.use_lob:
            toks.append(self.lob_enc(batch["lob"]))
        h = self.fuse(torch.cat(toks, -1))
        if self.use_ctx:
            h = self.film(h, batch["ctx"])
        return self.norm(h)
