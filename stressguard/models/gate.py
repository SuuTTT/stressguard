"""Stress-Extrapolated Channel Gating (SECG).

Invariance across the *seen* environments is the wrong target when the test regime is more severe than anything in training:
a shortcut whose predictive coefficient decays with stress still looks useful in every training environment, and IRMv1 on a deep
encoder does not remove it (Rosenfeld et al. 2021; Gulrajani & Lopez-Paz 2021). SECG estimates, per input channel c, how its
association with the crash label drifts with stress and extrapolates that drift beyond the training range:

    1. environments e = stress-score quantile bins fitted on TRAIN bars; s_e = mean stress score of e
    2. beta_{e,c}: standardised logistic coefficient of channel c (window summary features) on the label within e
    3. weighted least squares  beta_{e,c} ~ a_c + b_c s_e   (weights = positives in e)
    4. reliability r_c = clip( (a_c + b_c s*) / beta_calm_c , 0, 1 ) for channels that are significant in calm markets AND whose
       trend is significant and decays toward zero; all other channels keep r_c = 1 (v2: log stress, univariate, tested)
    5. gate g_c = r_c ** power multiplies channel c of the price window before the encoder.
Everything uses training bars only; the gate is frozen before training the network.
"""
from __future__ import annotations
from typing import Any, Dict, Sequence
import numpy as np
from sklearn.linear_model import LogisticRegression


def _summaries(pf: np.ndarray, tt: np.ndarray, L: int) -> np.ndarray:
    """(n, N, F) window summaries per channel: last value and mean of the last L bars (trailing)."""
    last = pf[tt]
    lo = np.maximum(tt - L + 1, 0)
    cs = np.cumsum(np.nan_to_num(pf), axis=0)
    mean = (cs[tt] - np.where(lo[:, None, None] > 0, cs[np.maximum(lo - 1, 0)], 0.0)) / (tt - lo + 1)[:, None, None]
    return np.stack([last, mean], -1)                        # (n, N, F, 2)


def _uni_logit(x: np.ndarray, y: np.ndarray, C: float = 10.0):
    """Univariate logistic slope of standardised x on y, with its Fisher standard error."""
    lr = LogisticRegression(C=C, max_iter=500).fit(x[:, None], y)
    b = float(lr.coef_[0, 0])
    p = lr.predict_proba(x[:, None])[:, 1]
    info = float((p * (1 - p) * x ** 2).sum())
    return b, (1.0 / np.sqrt(info) if info > 0 else np.inf)


def channel_reliability(price_feat: np.ndarray, labels: np.ndarray, stress_score: np.ndarray, train_idx: np.ndarray,
                        quantiles: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 0.9, 0.95), extrap: float = 0.5, min_calm: float = 2.0,
                        power: float = 1.0, L: int = 4, C: float = 10.0, z_trend: float = 2.0) -> Dict[str, Any]:
    """v2 (significance-tested). Per channel c and environment e (quantile bins of log stress score on TRAIN bars):
    univariate logistic slope beta_{e,c} (+ s.e.) of the channel's trailing L-bar mean on the label; weighted LS of beta on the
    environment's mean log stress; a channel is gated only if (i) it is significant in calm markets (|beta_0|/se >= min_calm),
    (ii) its trend is significant (|b|/se_b >= z_trend) and (iii) the trend decays toward zero. Then
    r_c = clip(beta_c(s*) / beta_0, 0, 1) at s* = s_top + extrap (s_top - s_0) (log units); otherwise r_c = 1."""
    T, N, F = price_feat.shape
    tr = np.asarray(train_idx)
    ls = np.log(np.clip(np.nan_to_num(stress_score, nan=1e-3), 1e-3, None))
    edges = np.nanquantile(ls[tr], list(quantiles))
    env = np.digitize(ls[tr], edges)
    S = _summaries(price_feat, tr, L)[..., 1]                 # (n, N, F) trailing mean
    mu, sd = np.nanmean(S.reshape(-1, F), 0), np.nanstd(S.reshape(-1, F), 0) + 1e-8
    B, SE, s_e = [], [], []
    for e in range(len(quantiles) + 1):
        m = env == e
        X = ((S[m] - mu) / sd).reshape(-1, F)
        y = labels[tr[m]].reshape(-1)
        ok = np.isfinite(X).all(1)
        X, y = X[ok], y[ok]
        if y.sum() < 5 or y.sum() == len(y):
            continue
        fits = [_uni_logit(X[:, c], y, C) for c in range(F)]
        B.append([f[0] for f in fits]); SE.append([f[1] for f in fits]); s_e.append(float(ls[tr[m]].mean()))
    B, SE, s_e = np.array(B), np.array(SE), np.array(s_e)
    rel = np.ones(F)
    info: Dict[str, Any] = {"betas": B.tolist(), "se": SE.tolist(), "s_e": s_e.tolist()}
    if len(B) < 3:
        info.update(gate=rel, reliability=rel, note="fewer than 3 usable environments")
        return info
    s_star = s_e[-1] + extrap * (s_e[-1] - s_e[0])
    trend, trend_se, beta_star, icpt, mask = np.zeros(F), np.zeros(F), np.zeros(F), np.zeros(F), np.zeros(F, bool)
    for c in range(F):
        w = 1.0 / np.clip(SE[:, c], 1e-6, None) ** 2
        A = np.stack([np.ones_like(s_e), s_e], 1)
        W = np.diag(w)
        cov = np.linalg.pinv(A.T @ W @ A)
        coef = cov @ A.T @ W @ B[:, c]
        trend[c], trend_se[c], icpt[c] = coef[1], np.sqrt(max(cov[1, 1], 0.0)), coef[0]
        beta_star[c] = coef[0] + coef[1] * s_star
        b0, se0 = B[0, c], SE[0, c]
        sig_calm = abs(b0) / se0 >= min_calm
        sig_trend = abs(trend[c]) / max(trend_se[c], 1e-12) >= z_trend
        decays = np.sign(trend[c]) == -np.sign(b0)
        if sig_calm and sig_trend and decays:
            rel[c] = float(np.clip(beta_star[c] / b0, 0.0, 1.0))
            mask[c] = True
    mu = np.nanmean(price_feat[tr].reshape(-1, F), 0)
    info.update(gate=rel ** power, reliability=rel, s_star=float(s_star), trend=trend.tolist(), trend_se=trend_se.tolist(), beta_star=beta_star.tolist(),
                intercept=icpt.tolist(), beta_calm=B[0].tolist(), mask=mask.tolist(), mu=np.nan_to_num(mu).tolist())
    return info
