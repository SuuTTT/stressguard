"""The 10 benchmark slots from the brief, each with paper, official code and VERIFICATION STATUS.

status = 'wrapper-needed'          paper + code verified; only the adapter (EventDataset batch -> official model -> score) is missing.
status = 'placeholder-unverified'  we could not verify a paper with that exact title; the slot names a verified substitute of the same
                                   family. Do NOT report a number under the unverified name - report the substitute and cite it.
Adapter contract: subclass ScoreBaseline, fit(train_loader, val_loader), score(batch) -> (B,) tensor; higher = more anomalous.
"""
from __future__ import annotations
from warn2trade.utils.registry import BASELINES
from .base import ScoreBaseline


def _slot(name, paper, repo, status, substitute=""):
    @BASELINES.register(name)
    class _Slot(ScoreBaseline):
        pass
    _Slot.__name__ = name
    _Slot.name, _Slot.paper, _Slot.official_repo, _Slot.status, _Slot.substitute = name, paper, repo, status, substitute

    def fit(self, train, val=None):
        raise NotImplementedError(f"[{name}] status={status}. paper: {paper}. repo: {repo or '-'}. "
                                  f"{'Verified substitute: ' + substitute + '. ' if substitute else ''}"
                                  "Implement fit/score against RegimeEventDataset batches (see baselines/base.py, simple.py).")
    _Slot.fit = fit
    _Slot.score = fit
    return _Slot


# ---- verified papers: adapter needed (official code exists); anomaly_transformer / tranad / timesnet / dcdetector are
#      now compact reimplementations in reimpl.py --------------------------------------------------------
_slot("moderntcn", "Luo, Wang. ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis. ICLR 2024", "https://github.com/luodhhh/ModernTCN", "wrapper-needed")
_slot("crossformer", "Zhang, Yan. Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting. ICLR 2023 (forecast-error AD)", "https://github.com/Thinklab-SJTU/Crossformer", "wrapper-needed")
_slot("patchtst_ad", "Nie et al. A Time Series is Worth 64 Words (PatchTST). ICLR 2023 - used with a reconstruction / forecast-error AD head ('PatchTST-AD 2025' as a standalone paper is unverified)", "https://github.com/yuqinie98/PatchTST", "wrapper-needed")
# ---- names in the brief that could not be verified: fill with the named substitute before reporting ------------------
_slot("diffusion_ad", "'Diffusion-AD (NeurIPS 2024/2025)' - no paper with this exact title verified", "", "placeholder-unverified",
      "D3R: Wang et al. Drift doesn't Matter: Dynamic Decomposition with Diffusion Reconstruction for Unstable MTS Anomaly Detection, NeurIPS 2023 (github.com/ForestsKing/D3R); or ImDiffusion, Chen et al. VLDB 2023")
_slot("regime_gnn_ad", "'Regime-Informed GNN-AD (2025)' - unverified", "", "placeholder-unverified",
      "GDN: Deng, Hooi. Graph Neural Network-Based Anomaly Detection in Multivariate Time Series, AAAI 2021 (github.com/d-ailin/GDN) + regime gating on the context prior")
_slot("mamba_finad", "'[2026 SOTA] Mamba-FinAD' - unverified", "", "placeholder-unverified",
      "selective-SSM backbone (Gu, Dao 2023, Mamba) with reconstruction AD head; verified SSM time-series models: S-Mamba (Wang et al. 2024), MambaTS (2024)")
_slot("unifin_llm", "'[2026 SOTA] UniFin-LLM / CausalTS-Regime' - unverified", "", "placeholder-unverified",
      "zero-shot foundation-model AD: MOMENT (Goswami et al. ICML 2024, github.com/moment-timeseries-foundation-model/moment); UniTS (Gao et al. NeurIPS 2024); Chronos (Ansari et al. 2024) forecast-error AD")
