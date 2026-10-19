"""StressGuard = InvariantEncoder -> {anomaly logit, severity quantiles} -> cost-sensitive dynamic decision boundary -> hedge ratio.

forward(batch) returns
    z        (B, d)   invariant representation
    logit, p (B,)     crash-within-lead probability
    sev_q    (B, Q)   monotone severity quantiles  E[|dP_{t+tau}| | crash]  (scale = cfg.sev_scale * cumsum(softplus))
    sev_hat  (B,)     the quantile used by the decision rule (cfg.decision.sev_q_index), or the sigma-plug-in if no head
    p_raw    (B,)     sigmoid(logit) as trained (cost-weighted, env-balanced, neg-subsampled -> NOT a calibrated probability)
    p        (B,)     Platt-calibrated probability sigmoid(cal_a * logit + cal_b), (cal_a, cal_b) fitted on the validation fold.
                      The Bayes rule needs a calibrated p (Prop. 1); cost re-weighting shifts the raw posterior (Elkan 2001).
    thr      (B,)     tau*_t = c_FP gamma_t / (c_FP gamma_t + c_FN sev_hat_t)      (dynamic)   or the val-fitted static threshold
    h        (B,)     hedge ratio in [0, h_max]; h > 0 iff p > thr
`batch["gamma"]` (round-trip friction in log-return units) must be attached by `attach_gamma(batch, cost_model)` before forward.
"""
from __future__ import annotations
import math
from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from warn2trade.backtest import CostModel
from warn2trade.utils.registry import MODELS
from .encoder import InvariantEncoder
from .losses import bayes_threshold, hedge_ratio


def attach_gamma(batch: Dict[str, torch.Tensor], cost_model: CostModel) -> Dict[str, torch.Tensor]:
    batch["gamma"] = cost_model.round_trip_estimate(batch["liq"][:, 0], batch["sigma"], batch["liq"][:, 2])
    return batch


@MODELS.register("stressguard")
class StressGuard(nn.Module):
    def __init__(self, dims: Dict[str, int], cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.enc = InvariantEncoder(dims, cfg.get("encoder", {}))
        d = self.enc.d_model
        self.head_cls = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        taus = [float(t) for t in cfg.get("sev_taus", [0.25, 0.5, 0.75, 0.9])]
        self.register_buffer("taus", torch.tensor(taus))
        self.head_sev = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, len(taus)))
        self.sev_scale = float(cfg.get("sev_scale", 0.01))
        self.use_sev_head = bool(cfg.get("use_sev_head", True))
        self.tau = int(cfg.get("tau", 10))
        dec = cfg.get("decision", {})
        self.dynamic = bool(dec.get("dynamic", True))
        self.c_fn, self.c_fp = float(dec.get("c_fn", 5.0)), float(dec.get("c_fp", 1.0))
        self.h_max = float(dec.get("h_max", 1.0))
        self.sev_q_index = int(dec.get("sev_q_index", 1))
        self.sev_plugin_k = float(dec.get("sev_plugin_k", 2.0))
        self.static_temp = float(dec.get("static_temp", 0.05))
        self.register_buffer("static_thr", torch.tensor(0.5))
        self.register_buffer("cal_a", torch.tensor(1.0))
        self.register_buffer("cal_b", torch.tensor(0.0))
        self.register_buffer("cal_c", torch.tensor(0.0))      # coefficient on log stress score (regime-aware calibration; 0 = plain Platt)
        self.ab = cfg.get("ablation", {})

    def set_calibration(self, a: float, b: float, c: float = 0.0) -> None:
        if a > 0:
            self.cal_a.fill_(float(a)); self.cal_b.fill_(float(b)); self.cal_c.fill_(float(c))

    @staticmethod
    def stress_cov(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        s = batch.get("stress_score")
        return torch.log(s.clamp_min(1e-3)) if s is not None else torch.zeros_like(batch["sigma"])

    def parameter_groups(self) -> Dict[str, list]:
        return {"encoder": list(self.enc.parameters()), "heads": list(self.head_cls.parameters()) + list(self.head_sev.parameters())}

    def set_costs(self, c_fn: float, c_fp: float = 1.0) -> None:
        self.c_fn, self.c_fp = float(c_fn), float(c_fp)

    def severity(self, z: torch.Tensor, batch: Dict[str, torch.Tensor]):
        q = self.sev_scale * torch.cumsum(F.softplus(self.head_sev(z)), dim=-1)
        if self.use_sev_head:
            return q, q[:, self.sev_q_index]
        return q, self.sev_plugin_k * batch["sigma"] * math.sqrt(self.tau)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        z = self.enc(batch)
        logit = self.head_cls(z).squeeze(-1)
        p_raw = torch.sigmoid(logit)
        p = torch.sigmoid((self.cal_a * logit + self.cal_b + self.cal_c * self.stress_cov(batch)).clamp(-60.0, 60.0))
        q, sev_hat = self.severity(z, batch)
        gamma = batch["gamma"]
        if self.dynamic:
            thr = bayes_threshold(sev_hat, gamma, self.c_fn, self.c_fp)
            h = hedge_ratio(p, sev_hat, gamma, self.c_fn, self.c_fp, self.h_max)
        else:
            thr = self.static_thr.expand_as(p)
            if self.training:
                h = self.h_max * torch.sigmoid((p - thr) / self.static_temp)
            else:
                h = self.h_max * (p >= thr).float()
        return {"z": z, "logit": logit, "p_raw": p_raw, "p": p, "sev_q": q, "sev_hat": sev_hat, "thr": thr, "h": h}
