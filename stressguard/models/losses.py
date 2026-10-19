"""StressGuard objective.

    L(theta, w) = (1/|E|) sum_e [ L_econ^e(w . Phi_theta) + lambda_IRM || grad_{w|w=1} L_econ^e(w . Phi_theta) ||^2 ]
                + lambda_VREx Var_e[L_econ^e]  (alternative invariance penalty)
                + lambda_sev  L_pinball(severity quantiles | y = 1)
                + lambda_CVaR CVaR_alpha( - PnL_overlay(h) )

Asymmetric economic loss (per sample, see paper eq. L_econ):
    linear surrogate  : c_FN y (1 - p) s  +  c_FP (1 - y) p gamma             (exactly the brief's formula; non-proper)
    log surrogate     : c_FN y s (-log p)  +  c_FP (1 - y) gamma (-log(1 - p))  (strictly proper; default for training)
Both are minimised by the same Bayes decision rule  d = 1[p >= tau*],  tau*(c) = c_FP gamma(c) / (c_FP gamma(c) + c_FN s_hat(c)).
s = |dP_{t+tau}| forward crash severity (log-return units), gamma = round-trip hedge friction from the cost model (same units).
"""
from __future__ import annotations
from typing import Callable, Optional
import torch
import torch.nn.functional as F


def econ_weights(y: torch.Tensor, sev: torch.Tensor, gamma: torch.Tensor, c_fn: float = 1.0, c_fp: float = 1.0,
                 use_sev_weight: bool = True, use_gamma_weight: bool = True, sev_floor: float = 1e-4) -> torch.Tensor:
    s = sev.clamp_min(sev_floor) if use_sev_weight else torch.ones_like(sev)
    g = gamma.clamp_min(1e-6) if use_gamma_weight else torch.ones_like(gamma)
    return torch.where(y > 0.5, c_fn * s, c_fp * g)


def cost_sensitive_loss(logit: torch.Tensor, y: torch.Tensor, sev: torch.Tensor, gamma: torch.Tensor, c_fn: float = 1.0, c_fp: float = 1.0,
                        surrogate: str = "log", use_sev_weight: bool = True, use_gamma_weight: bool = True, sev_floor: float = 1e-4,
                        normalize: bool = True, norm: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Per-sample asymmetric economic loss (B,). c_fn = c_fp = 1 with both weights off recovers plain BCE (the symmetric ablation).
    `norm` (a detached scalar) overrides the batch-mean normaliser so several calls (per environment, IRM closure) share one scale."""
    w = econ_weights(y, sev, gamma, c_fn, c_fp, use_sev_weight, use_gamma_weight, sev_floor)
    if norm is not None:
        w = w / norm
    elif normalize:
        w = w / w.mean().detach().clamp_min(1e-12)
    if surrogate == "linear":
        p = torch.sigmoid(logit)
        base = y * (1.0 - p) + (1.0 - y) * p
    else:
        base = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    return w * base


def irm_penalty(logit: torch.Tensor, loss_fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """IRMv1 (Arjovsky et al. 2019): squared gradient of the environment risk w.r.t. a dummy classifier scale w = 1.0."""
    if not torch.is_grad_enabled():
        return logit.new_zeros(())
    scale = torch.ones(1, device=logit.device, requires_grad=True)
    loss = loss_fn(logit * scale)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return (grad ** 2).sum()


def vrex_penalty(env_losses: torch.Tensor) -> torch.Tensor:
    """V-REx (Krueger et al. 2021): variance of the environment risks."""
    return env_losses.var(unbiased=False) if env_losses.numel() > 1 else env_losses.new_zeros(())


def severity_pinball(q: torch.Tensor, sev: torch.Tensor, taus: torch.Tensor, y: torch.Tensor, pos_only: bool = True) -> torch.Tensor:
    """Quantile regression of the crash severity. q (B, Q) monotone quantiles, sev (B,), taus (Q,)."""
    u = sev.unsqueeze(-1) - q
    loss = torch.maximum(taus * u, (taus - 1.0) * u).mean(-1)
    if pos_only:
        m = y > 0.5
        return loss[m].mean() if m.any() else loss.sum() * 0.0
    return loss.mean()


def cvar_loss(pnl: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    """Rockafellar-Uryasev CVaR_alpha of the loss (-pnl), nu fixed at the empirical VaR."""
    loss = -pnl
    nu = torch.quantile(loss.detach(), 1.0 - alpha)
    return nu + torch.clamp(loss - nu, min=0.0).mean() / alpha


def bayes_threshold(sev_hat: torch.Tensor, gamma: torch.Tensor, c_fn: float, c_fp: float) -> torch.Tensor:
    """tau*(c) = c_FP gamma / (c_FP gamma + c_FN s_hat): the cost-sensitive decision boundary, re-computed every bar."""
    num = c_fp * gamma.clamp_min(1e-8)
    return num / (num + c_fn * sev_hat.clamp_min(1e-8))


def hedge_ratio(p: torch.Tensor, sev_hat: torch.Tensor, gamma: torch.Tensor, c_fn: float, c_fp: float, h_max: float = 1.0) -> torch.Tensor:
    """Utility-optimal hedge size: h = h_max (p c_FN s_hat - (1-p) c_FP gamma)_+ / (c_FN s_hat).
    h > 0  <=>  p > tau*(c); h -> h_max as p -> 1. Linear in the expected net edge, so it is continuous and differentiable."""
    edge = p * c_fn * sev_hat - (1.0 - p) * c_fp * gamma
    return (h_max * torch.relu(edge) / (c_fn * sev_hat.clamp_min(1e-8))).clamp(0.0, h_max)
