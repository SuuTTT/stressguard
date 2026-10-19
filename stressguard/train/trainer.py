"""Environment-batched training of StressGuard with an annealed invariance penalty and worst-environment model selection.

Per mini-batch (env-balanced sampler guarantees Stress samples):
    L = mean_e L_econ^e + lam_IRM(ep) mean_e IRMv1^e [+ lam_VREx Var_e L_econ^e] + lam_sev L_pinball + lam_CVaR CVaR_alpha(-hedged return)
lam_IRM(ep) = lam_irm_warm for the first `irm_warmup` epochs, then lam_irm (Arjovsky et al.: total /= lam when lam > 1).
Model selection: `select` in {worst_env_econ (default, = minimax objective), mean_econ, stress_auc}.
"""
from __future__ import annotations
import copy
import time
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from warn2trade.backtest import CostModel
from warn2trade.backtest import metrics as M
from warn2trade.train.trainer import resolve_device, to_device
from warn2trade.utils.logger import get_logger
from ..models import losses as Lz
from ..models.stressguard import attach_gamma
from ..backtest import overlay_pnl, OverlayBacktester
from ..baselines.base import calibrate_scores

log = get_logger("stressguard")


class StressTrainer:
    def __init__(self, cfg: Dict[str, Any], model: torch.nn.Module, loaders: Dict[str, DataLoader], cost_model: CostModel,
                 device: torch.device, bars_per_year: int):
        self.cfg, self.model, self.loaders, self.cost, self.device, self.bpy = cfg, model.to(device), loaders, cost_model, device, bars_per_year
        self.mcfg, self.tcfg = cfg["model"], cfg["train"]
        self.lcfg = self.tcfg.get("loss", {})
        self.tau, self.latency = int(self.mcfg["tau"]), int(cfg["data"]["latency"])
        rq = cfg.get("regime", {}).get("irm_quantiles")
        self.n_env = (len(rq) + 1) if rq else int(cfg.get("regime", {}).get("n_env", 2))

    # ------------------------------------------------------------------ helpers
    def _prep(self, b: Dict[str, torch.Tensor], cost_model: Optional[CostModel] = None) -> Dict[str, torch.Tensor]:
        return attach_gamma(to_device(b, self.device), cost_model or self.cost)

    def _econ_kwargs(self) -> Dict[str, Any]:
        lc = self.lcfg
        return dict(c_fn=float(lc.get("c_fn", self.model.c_fn)), c_fp=float(lc.get("c_fp", self.model.c_fp)), surrogate=lc.get("surrogate", "log"),
                    use_sev_weight=bool(lc.get("use_sev_weight", True)), use_gamma_weight=bool(lc.get("use_gamma_weight", True)))

    def _lam_irm(self, epoch: int) -> float:
        if self.lcfg.get("invariance", "irm") != "irm":
            return 0.0
        return float(self.lcfg.get("lam_irm_warm", 1.0)) if epoch < int(self.lcfg.get("irm_warmup", 2)) else float(self.lcfg.get("lam_irm", 10.0))

    # ------------------------------------------------------------------ objective
    def compute_losses(self, out: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], lam_irm: float) -> Dict[str, torch.Tensor]:
        lc, kw = self.lcfg, self._econ_kwargs()
        y, sev, gam, logit = b["y"], b["sev"], b["gamma"], out["logit"]
        env = b["irm_env"].long() if "irm_env" in b else b["env"]
        wkw = {k: kw[k] for k in ("c_fn", "c_fp", "use_sev_weight", "use_gamma_weight")}
        if lc.get("label_balance_env", False):
            # per-environment class balancing: every environment contributes equal positive and negative mass, so a regime-dependent
            # crash base rate is not mistaken for a spurious mechanism by the invariance penalty
            lb = torch.ones_like(y)
            for e in torch.unique(env):
                m = env == e
                npos, nneg = float(y[m].sum()), float((1 - y[m]).sum())
                if npos > 0 and nneg > 0:
                    lb = torch.where(m & (y > 0.5), torch.full_like(y, 0.5 * (npos + nneg) / npos), lb)
                    lb = torch.where(m & (y < 0.5), torch.full_like(y, 0.5 * (npos + nneg) / nneg), lb)
            gam = gam * torch.where(y < 0.5, lb, torch.ones_like(lb)) if kw["use_gamma_weight"] else gam
            sev = sev * torch.where(y > 0.5, lb, torch.ones_like(lb)) if kw["use_sev_weight"] else sev
            if not kw["use_gamma_weight"] and not kw["use_sev_weight"]:
                kw = dict(kw, use_sev_weight=True, use_gamma_weight=True, c_fn=kw["c_fn"], c_fp=kw["c_fp"])
                sev, gam = torch.where(y > 0.5, lb, torch.ones_like(lb)), torch.where(y < 0.5, lb, torch.ones_like(lb))
                wkw = {k: kw[k] for k in ("c_fn", "c_fp", "use_sev_weight", "use_gamma_weight")}
        norm = Lz.econ_weights(y, sev, gam, **wkw).mean().detach().clamp_min(1e-12)      # one normaliser for risk and IRM penalty
        per = Lz.cost_sensitive_loss(logit, y, sev, gam, norm=norm, **kw)
        inv, min_n = lc.get("invariance", "irm"), int(lc.get("min_env_samples", 8))
        L_env, P_env = [], []
        for e in range(self.n_env):
            m = env == e
            if int(m.sum()) < min_n:
                continue
            L_env.append(per[m].mean())
            if inv == "irm" and lam_irm > 0:
                ym, sm, gm = y[m], sev[m], gam[m]
                P_env.append(Lz.irm_penalty(logit[m], lambda lg: Lz.cost_sensitive_loss(lg, ym, sm, gm, norm=norm, **kw).mean()))
        if not L_env:
            L_env = [per.mean()]
        L: Dict[str, torch.Tensor] = {"econ": torch.stack(L_env).mean()}
        total = L["econ"]
        if P_env:
            L["irm"] = torch.stack(P_env).mean()
            total = total + lam_irm * L["irm"]
        lam_vrex = float(lc.get("lam_vrex", 0.0))
        if inv == "vrex" and len(L_env) > 1 and lam_vrex > 0:
            L["vrex"] = Lz.vrex_penalty(torch.stack(L_env))
            total = total + lam_vrex * L["vrex"]
        lam_sev = float(lc.get("lam_sev", 1.0))
        if lam_sev > 0 and getattr(self.model, "use_sev_head", True):
            L["sev"] = Lz.severity_pinball(out["sev_q"], sev, self.model.taus, y, bool(lc.get("sev_pos_only", True)))
            total = total + lam_sev * L["sev"]
        lam_cvar = float(lc.get("lam_cvar", 0.0))
        if lam_cvar > 0:
            r = overlay_pnl(out["h"], b, self.cost, self.latency, self.tau)
            L["cvar"] = Lz.cvar_loss(r["hedged"] * float(lc.get("pnl_scale", 100.0)), float(lc.get("alpha", 0.05)))
            total = total + lam_cvar * L["cvar"]
        if lam_irm > 1.0:
            total = total / lam_irm
        L["total"] = total
        return L

    # ------------------------------------------------------------------ training loop
    def fit(self) -> Dict[str, Any]:
        opt = torch.optim.AdamW(self.model.parameters(), lr=float(self.tcfg.get("lr", 5e-4)), weight_decay=float(self.tcfg.get("weight_decay", 1e-4)))
        best, best_state, bad, history = -float("inf"), None, 0, []
        for ep in range(int(self.tcfg.get("epochs", 12))):
            lam = self._lam_irm(ep)
            self.model.train()
            t0, agg, n = time.time(), {}, 0
            for b in self.loaders["train"]:
                b = self._prep(b)
                out = self.model(b)
                L = self.compute_losses(out, b, lam)
                opt.zero_grad(set_to_none=True)
                L["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.tcfg.get("grad_clip", 1.0)))
                opt.step()
                for k, v in L.items():
                    agg[k] = agg.get(k, 0.0) + float(v.detach())
                n += 1
            val = self.validate()
            rec = {"epoch": ep, "lam_irm": lam, "sec": round(time.time() - t0, 1), **{k: v / max(n, 1) for k, v in agg.items()},
                   "val_score": val["score"], "val_auc_normal": val["auc_normal"], "val_auc_stress": val["auc_stress"], "cal_a": float(self.model.cal_a), "cal_b": float(self.model.cal_b),
                   **{f"val_econ_env{e}": v for e, v in val["econ_by_env"].items()}}
            history.append(rec)
            log.info(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
            if val["score"] > best:
                best, best_state, bad = val["score"], copy.deepcopy(self.model.state_dict()), 0
            else:
                bad += 1
                if bad >= int(self.tcfg.get("patience", 4)):
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)     # includes the calibration fitted at that epoch
        self.fit_static_threshold()
        return {"history": history, "best_val": best, "calibration": [float(self.model.cal_a), float(self.model.cal_b)]}

    @torch.no_grad()
    def _collect_val(self):
        self.model.eval()
        ys, lg, es, per_all, w_all, sc = [], [], [], [], [], []
        kw = self._econ_kwargs()
        wkw = {k: kw[k] for k in ("c_fn", "c_fp", "use_sev_weight", "use_gamma_weight")}
        for b in self.loaders["val"]:
            b = self._prep(b)
            out = self.model(b)
            per_all.append(Lz.cost_sensitive_loss(out["logit"], b["y"], b["sev"], b["gamma"], normalize=False, **kw))
            w_all.append(Lz.econ_weights(b["y"], b["sev"], b["gamma"], **wkw))
            ys.append(b["y"]); lg.append(out["logit"]); es.append(b["env"]); sc.append(self.model.stress_cov(b))
        per = torch.cat(per_all) / torch.cat(w_all).mean().clamp_min(1e-12)       # one global normaliser (not per batch)
        self._val_cov = torch.cat(sc).cpu().numpy()
        return torch.cat(ys).cpu().numpy(), torch.cat(lg).cpu().numpy(), torch.cat(es).cpu().numpy(), per.cpu().numpy()

    def _calibrated(self, logit: np.ndarray) -> np.ndarray:
        cov = getattr(self, "_val_cov", None)
        cov = cov if (cov is not None and len(cov) == len(logit)) else 0.0
        z = np.clip(float(self.model.cal_a) * logit.astype(np.float64) + float(self.model.cal_b) + float(self.model.cal_c) * cov, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-z))

    @torch.no_grad()
    def fit_calibration(self, y: np.ndarray, logit: np.ndarray) -> None:
        """Platt scaling of the logit on the validation fold -> calibrated p for the Bayes rule (refit every epoch)."""
        if self.tcfg.get("calibration", "platt") == "regime" and getattr(self, "_val_cov", None) is not None and len(self._val_cov) == len(y):
            from sklearn.linear_model import LogisticRegression
            ok = np.isfinite(logit)
            if 0 < y[ok].sum() < ok.sum():
                X = np.stack([logit[ok], self._val_cov[ok]], 1).astype(np.float64)
                mu, sd = X.mean(0), X.std(0) + 1e-8
                lr = LogisticRegression(C=1.0, max_iter=1000).fit((X - mu) / sd, y[ok])
                w = lr.coef_[0] / sd
                b = float(lr.intercept_[0] - (lr.coef_[0] * mu / sd).sum())
                if w[0] > 0 and np.isfinite(w).all() and np.isfinite(b):
                    self.model.set_calibration(float(w[0]), b, float(w[1]))
                    return
        a, b = calibrate_scores(logit, y)
        self.model.set_calibration(a, b)

    @torch.no_grad()
    def validate(self) -> Dict[str, Any]:
        y, logit, env, per = self._collect_val()
        self.fit_calibration(y, logit)
        p = self._calibrated(logit)
        econ = {int(e): float(per[env == e].mean()) for e in np.unique(env)}
        auc = {}
        for name, e in (("normal", 0), ("stress", 1)):
            m = env == e
            auc[name] = M.auc_roc(y[m], p[m]) if m.sum() > 0 else float("nan")
        sel = self.tcfg.get("select", "worst_env_econ")
        if sel == "stress_auc" and np.isfinite(auc["stress"]):
            score = auc["stress"]
        elif sel == "mean_econ":
            score = -float(per.mean())
        else:
            score = -max(econ.values())
        return {"score": float(score), "econ_by_env": econ, "auc_normal": auc["normal"], "auc_stress": auc["stress"]}

    @torch.no_grad()
    def fit_static_threshold(self) -> float:
        """Best-F1 threshold on the validation fold = the static rule every F1-tuned detector uses (static_thr ablation + cost-curve comparison)."""
        y, logit, _, _ = self._collect_val()
        p = self._calibrated(logit)
        _, thr = M.best_f1(y, p)
        self.model.static_thr.fill_(float(thr))
        return float(thr)

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict(self, loader: DataLoader, cost_model: Optional[CostModel] = None) -> pd.DataFrame:
        self.model.eval()
        bt = OverlayBacktester(cost_model or self.cost, self.latency, self.tau)
        frames = [bt.collect(self.model(b), b) for b in (self._prep(bb, cost_model) for bb in loader)]
        df = pd.concat(frames, ignore_index=True)
        df["static_thr"] = float(self.model.static_thr)
        return df
