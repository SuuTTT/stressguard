"""Compact, faithful-in-mechanism PyTorch reimplementations of four verified deep time-series anomaly detectors.

Written from the papers (no official code downloaded); each scores the LAST time step of a (B, L, F) window, higher = more anomalous,
and is trained unsupervised on Normal-only windows (y == 0), like recon_ae. Models are deliberately small (d_model 32, 1 layer by
default) so a few CPU epochs on ~5-20k windows stay within minutes. Report these as "reimplementation", never as official numbers.

    anomaly_transformer  Xu et al. ICLR 2022     anomaly-attention (Gaussian prior + series association), minimax AssDis,
                                                 score = softmax_t(-AssDis * T) * recon error, last step
    tranad               Tuli et al. VLDB 2022   encoder + 2 decoders, two-phase focus-score self-conditioning, adversarial
                                                 decoder-2 objective with 1/n epoch weighting, score = 0.5 (phase-1 + phase-2) error
    dcdetector           Yang et al. KDD 2023    channel-independent dual attention (patch-wise / in-patch) contrastive maps with
                                                 stop-gradient, no reconstruction, score = symmetric KL between the views, last step
    timesnet             Wu et al. ICLR 2023     FFT top-k periods, 1D->2D fold, Inception 2D conv, amplitude-softmax aggregation,
                                                 score = reconstruction MSE at the last step
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.data import DataLoader

from warn2trade.train.trainer import to_device
from warn2trade.utils.registry import BASELINES
from .base import ScoreBaseline


# ----------------------------------------------------------------------------------------------------------------- shared
def _sinusoid(L: int, d: int) -> torch.Tensor:
    pe = torch.zeros(L, d)
    pos = torch.arange(L).float()[:, None]
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)[:, : pe[:, 1::2].shape[1]]
    return pe


class _DataEmb(nn.Module):
    """Token (circular conv1d, k=3) + fixed sinusoidal position embedding, as in the Informer-style DataEmbedding all four repos use."""

    def __init__(self, c_in: int, d: int, max_len: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.tok = nn.Conv1d(c_in, d, 3, padding=1, padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.tok.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.register_buffer("pe", _sinusoid(max_len, d), persistent=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                                  # (B, L, c_in) -> (B, L, d)
        return self.drop(self.tok(x.transpose(1, 2)).transpose(1, 2) + self.pe[: x.shape[1]])


def _kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Row-wise KL over the key axis, averaged over heads: (B, H, L, L) -> (B, L). Same eps as the official my_kl_loss."""
    return (p * (torch.log(p + 1e-4) - torch.log(q + 1e-4))).sum(-1).mean(1)


def _rownorm(a: torch.Tensor) -> torch.Tensor:
    return a / (a.sum(-1, keepdim=True) + 1e-12)


class _Reimpl(ScoreBaseline):
    """Shared fit loop: train-fold feature scaler, Normal-only mask, Adam; subclasses define build / train_step / score_x."""
    status = "reimplementation"
    scaler = "standard"                                                    # standard (z-score) | minmax ([0,1], TranAD)

    def _c(self, k: str, d):
        return type(d)(self.cfg.get(k, d)) if d is not None else self.cfg.get(k)

    def _fit_scaler(self, train: DataLoader, F: int) -> None:
        n, s, s2 = 0, torch.zeros(F, dtype=torch.float64), torch.zeros(F, dtype=torch.float64)
        lo, hi = torch.full((F,), float("inf"), dtype=torch.float64), torch.full((F,), -float("inf"), dtype=torch.float64)
        for b in train:
            x = b["price"].reshape(-1, F).double()
            x = x[torch.isfinite(x).all(-1)]
            n += len(x); s += x.sum(0); s2 += (x ** 2).sum(0)
            lo, hi = torch.minimum(lo, x.min(0).values), torch.maximum(hi, x.max(0).values)
        mu = s / max(n, 1)
        sd = (s2 / max(n, 1) - mu ** 2).clamp_min(0).sqrt() + 1e-6
        if self.scaler == "minmax":
            self.mu, self.sd = lo.float().to(self.device), (hi - lo).float().to(self.device) + 1e-6
        else:
            self.mu, self.sd = mu.float().to(self.device), sd.float().to(self.device)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num((x.float() - self.mu) / self.sd)
        return x.clamp(-10.0, 10.0)

    def build(self, L: int, F: int) -> nn.Module:  # pragma: no cover
        raise NotImplementedError

    def train_step(self, x: torch.Tensor, epoch: int) -> float:  # pragma: no cover
        raise NotImplementedError

    def score_x(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def make_opt(self):
        return torch.optim.Adam(self.net.parameters(), lr=self._c("lr", 1e-3))

    def fit(self, train: DataLoader, val: Optional[DataLoader] = None) -> None:
        L, F = next(iter(train))["price"].shape[1:]
        self.L, self.F = int(L), int(F)
        self._fit_scaler(train, self.F)
        self.net = self.build(self.L, self.F).to(self.device)
        self.opt = self.make_opt()
        self.history: List[float] = []
        for ep in range(self._c("epochs", 5)):
            self.net.train()
            tot, nb = 0.0, 0
            for b in train:
                b = to_device(b, self.device)
                m = b["y"] < 0.5 if self.cfg.get("normal_only", True) else torch.ones_like(b["y"], dtype=torch.bool)
                if m.sum() < 2:
                    continue
                tot += self.train_step(self._norm(b["price"][m]), ep); nb += 1
            self.history.append(tot / max(nb, 1))
        self.net.eval()

    @torch.no_grad()
    def score(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.net.eval()
        x = batch["price"]
        s = self.score_x(self._norm(x.to(self.device)))
        return torch.nan_to_num(s.float(), nan=0.0, posinf=1e6).to(x.device)


# ------------------------------------------------------------------------------------------------- Anomaly Transformer
class _AnomalyAttention(nn.Module):
    def __init__(self, d: int, H: int, L: int, dropout: float):
        super().__init__()
        self.H, self.dk = H, d // H
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.sigma = nn.Linear(d, H)                                       # learnable, per-step, per-head Gaussian scale
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        idx = torch.arange(L).float()
        self.register_buffer("dist", (idx[:, None] - idx[None, :]).abs(), persistent=False)

    def forward(self, x):
        B, L, _ = x.shape
        q = self.q(x).view(B, L, self.H, self.dk).transpose(1, 2)
        k = self.k(x).view(B, L, self.H, self.dk).transpose(1, 2)
        v = self.v(x).view(B, L, self.H, self.dk).transpose(1, 2)
        series = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.dk), -1)      # series association S
        sig = torch.sigmoid(5 * self.sigma(x).transpose(1, 2)) + 1e-5                    # (B, H, L), official sigma transform
        sig = (torch.pow(3.0, sig) - 1).unsqueeze(-1)
        prior = 1.0 / (math.sqrt(2 * math.pi) * sig) * torch.exp(-self.dist[:L, :L] ** 2 / (2 * sig ** 2))   # prior association P
        out = (self.drop(series) @ v).transpose(1, 2).reshape(B, L, -1)
        return self.out(out), series, prior


class _ATLayer(nn.Module):
    def __init__(self, d: int, H: int, L: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = _AnomalyAttention(d, H, L, dropout)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d))
        self.n1, self.n2, self.drop = nn.LayerNorm(d), nn.LayerNorm(d), nn.Dropout(dropout)

    def forward(self, x):
        a, s, p = self.attn(x)
        x = self.n1(x + self.drop(a))
        return self.n2(x + self.drop(self.ff(x))), s, p


class _ATNet(nn.Module):
    def __init__(self, L: int, F: int, d: int, H: int, layers: int, d_ff: int, dropout: float):
        super().__init__()
        self.emb = _DataEmb(F, d, max(L, 8), dropout)
        self.layers = nn.ModuleList([_ATLayer(d, H, L, d_ff, dropout) for _ in range(layers)])
        self.norm, self.proj = nn.LayerNorm(d), nn.Linear(d, F)

    def forward(self, x):
        h = self.emb(x)
        S, P = [], []
        for lyr in self.layers:
            h, s, p = lyr(h)
            S.append(s); P.append(_rownorm(p))
        return self.proj(self.norm(h)), S, P


@BASELINES.register("anomaly_transformer")
class AnomalyTransformer(_Reimpl):
    name = "anomaly_transformer"
    paper = "Xu, Wu, Wang, Long. Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy. ICLR 2022"
    official_repo = "https://github.com/thuml/Anomaly-Transformer"

    def build(self, L, F):
        return _ATNet(L, F, self._c("d_model", 32), self._c("n_heads", 4), self._c("e_layers", 1), self._c("d_ff", 64), self._c("dropout", 0.0))

    def train_step(self, x, epoch):
        rec, S, P = self.net(x)
        r = Fn.mse_loss(rec, x)
        n = len(S)
        series_loss = sum((_kl(s, p.detach()) + _kl(p.detach(), s)).mean() for s, p in zip(S, P)) / n
        prior_loss = sum((_kl(p, s.detach()) + _kl(s.detach(), p)).mean() for s, p in zip(S, P)) / n
        k = self._c("k", 3.0)
        loss1 = r - k * series_loss                                        # maximise phase: enlarge discrepancy via S
        loss2 = r + k * prior_loss                                         # minimise phase: pull P towards S
        self.opt.zero_grad()
        loss1.backward(retain_graph=True)
        loss2.backward()
        self.opt.step()
        return float(r)

    def score_x(self, x):
        rec, S, P = self.net(x)
        T = self._c("temperature", 50.0)
        dis = sum(_kl(s, p) * T + _kl(p, s) * T for s, p in zip(S, P))   # (B, L), official test-time AssDis
        err = ((rec - x) ** 2).mean(-1)                                    # (B, L)
        return (torch.softmax(-dis, -1) * err)[:, -1]


# ------------------------------------------------------------------------------------------------------------- TranAD
class _TranADNet(nn.Module):
    def __init__(self, F: int, d: int, H: int, layers: int, d_ff: int, dropout: float, W: int):
        super().__init__()
        self.F = F
        self.inp = nn.Linear(2 * F, d)                                     # [window, focus] -> d_model (official: d_model = 2F)
        self.tgt = nn.Linear(2 * F, d)
        self.register_buffer("pe", _sinusoid(max(W, 8), d), persistent=False)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, H, d_ff, dropout, batch_first=True), layers)
        self.dec1 = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, H, d_ff, dropout, batch_first=True), 1)
        self.dec2 = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, H, d_ff, dropout, batch_first=True), 1)
        self.out1 = nn.Sequential(nn.Linear(d, F), nn.Sigmoid())
        self.out2 = nn.Sequential(nn.Linear(d, F), nn.Sigmoid())

    def encode(self, W, c):
        mem = self.enc(self.inp(torch.cat([W, c], -1)) * math.sqrt(self.F) + self.pe[: W.shape[1]])
        tgt = self.tgt(W[:, -1:].repeat(1, 1, 2))                          # official: tgt = last element repeated x2
        return tgt, mem

    def forward(self, W):
        c0 = torch.zeros_like(W)
        tgt, mem = self.encode(W, c0)
        o1 = self.out1(self.dec1(tgt, mem))                                # phase 1, decoder 1
        o2 = self.out2(self.dec2(tgt, mem))                                # phase 1, decoder 2
        focus = (o1 - W) ** 2                                              # self-conditioning focus score, broadcast over window
        tgt2, mem2 = self.encode(W, focus)
        o2h = self.out2(self.dec2(tgt2, mem2))                             # phase 2, decoder 2
        return o1, o2, o2h

    def groups(self):
        g2 = list(self.dec2.parameters()) + list(self.out2.parameters())
        ids = {id(p) for p in g2}
        return [p for p in self.parameters() if id(p) not in ids], g2


@BASELINES.register("tranad")
class TranAD(_Reimpl):
    name = "tranad"
    paper = "Tuli, Casale, Jennings. TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data. VLDB 2022"
    official_repo = "https://github.com/imperial-qore/TranAD"
    scaler = "minmax"

    def build(self, L, F):
        self.W = min(self._c("window", 10), L)                             # TranAD's short local window (official n_window = 10)
        return _TranADNet(F, self._c("d_model", 32), self._c("n_heads", 4), self._c("e_layers", 1), self._c("d_ff", 64),
                          self._c("dropout", 0.1), self.W)

    def make_opt(self):
        self.gA, self.gB = self.net.groups()
        return torch.optim.AdamW(self.net.parameters(), lr=self._c("lr", 1e-3), weight_decay=1e-5)

    def train_step(self, x, epoch):
        W = x[:, -self.W:]
        tgt = W[:, -1:]
        o1, o2, o2h = self.net(W)
        w = 1.0 / (epoch + 1)                                              # epoch-dependent weighting (official 1/n schedule)
        e1, e2, e2h = Fn.mse_loss(o1, tgt), Fn.mse_loss(o2, tgt), Fn.mse_loss(o2h, tgt)
        L1 = w * e1 + (1 - w) * e2h                                        # encoder + decoder 1: reconstruct and fool phase 2
        L2 = w * e2 - (1 - w) * e2h                                        # decoder 2: adversarially amplify phase-2 error
        gA = torch.autograd.grad(L1, self.gA, retain_graph=True, allow_unused=True)
        gB = torch.autograd.grad(L2, self.gB, allow_unused=True)
        self.opt.zero_grad()
        for p, g in list(zip(self.gA, gA)) + list(zip(self.gB, gB)):
            p.grad = g
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.opt.step()
        return float(e1)

    def score_x(self, x):
        W = x[:, -self.W:]
        o1, _, o2h = self.net(W)
        tgt = W[:, -1:]
        return (0.5 * (o1 - tgt) ** 2 + 0.5 * (o2h - tgt) ** 2).mean(-1)[:, -1]


# --------------------------------------------------------------------------------------------------------- DCdetector
class _RevIN(nn.Module):
    def __init__(self, F: int):
        super().__init__()
        self.w, self.b = nn.Parameter(torch.ones(F)), nn.Parameter(torch.zeros(F))

    def forward(self, x):
        mu, sd = x.mean(1, keepdim=True).detach(), (x.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt().detach()
        return (x - mu) / sd * self.w + self.b


class _DCNet(nn.Module):
    def __init__(self, L: int, F: int, d: int, H: int, patch_sizes: List[int], dropout: float):
        super().__init__()
        self.L, self.H, self.dk = L, H, d // H
        self.ps = [p for p in patch_sizes if L % p == 0 and 1 < p < L] or [max(p for p in range(2, L) if L % p == 0)]
        self.revin = _RevIN(F)
        self.emb_size = nn.ModuleList([_DataEmb(p, d, L, dropout) for p in self.ps])          # patch-wise tokens (dim p)
        self.emb_num = nn.ModuleList([_DataEmb(L // p, d, L, dropout) for p in self.ps])      # in-patch tokens (dim L/p)
        self.q, self.k = nn.Linear(d, d), nn.Linear(d, d)                  # attention maps only: no values, no decoder

    def _attn(self, h):
        BM, n, _ = h.shape
        q = self.q(h).view(BM, n, self.H, self.dk).transpose(1, 2)
        k = self.k(h).view(BM, n, self.H, self.dk).transpose(1, 2)
        return torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.dk), -1)        # (BM, H, n, n)

    def forward(self, x):
        B, L, M = x.shape
        x = self.revin(x).transpose(1, 2).reshape(B * M, L)                # channel independence
        S, P = [], []
        for p, es, en in zip(self.ps, self.emb_size, self.emb_num):
            n = L // p
            xs = x.view(B * M, n, p)                                       # token i = patch i
            xn = x.view(B * M, n, p).transpose(1, 2)                       # token j = in-patch position j across patches
            a_s = self._attn(es(xs))                                       # (BM, H, n, n) patch-wise
            a_n = self._attn(en(xn))                                       # (BM, H, p, p) in-patch
            a_s = a_s.repeat_interleave(p, -1).repeat_interleave(p, -2)    # upsample: position t -> patch t // p
            a_n = a_n.repeat(1, 1, n, n)                                   # tile: position t -> in-patch index t % p
            S.append(_rownorm(a_s.view(B, M, self.H, L, L).mean(1)))
            P.append(_rownorm(a_n.view(B, M, self.H, L, L).mean(1)))
        return S, P


@BASELINES.register("dcdetector")
class DCdetector(_Reimpl):
    name = "dcdetector"
    paper = "Yang, Zhang, Zhou, Wen, Sun. DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection. KDD 2023"
    official_repo = "https://github.com/DAMO-DI-ML/KDD2023-DCdetector"

    def build(self, L, F):
        return _DCNet(L, F, self._c("d_model", 32), self._c("n_heads", 1), list(self.cfg.get("patch_sizes", [4, 8])), self._c("dropout", 0.0))

    def train_step(self, x, epoch):
        S, P = self.net(x)
        n = len(S)
        series_loss = sum((_kl(s, p.detach()) + _kl(p.detach(), s)).mean() for s, p in zip(S, P)) / n
        prior_loss = sum((_kl(p, s.detach()) + _kl(s.detach(), p)).mean() for s, p in zip(S, P)) / n
        loss = prior_loss - series_loss                                    # official contrastive objective with stop-gradient
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return float(prior_loss)

    def score_x(self, x):
        S, P = self.net(x)
        dis = sum(_kl(s, p) + _kl(p, s) for s, p in zip(S, P)) / len(S)    # (B, L) dual-view discrepancy (paper Eq.)
        if self.cfg.get("score_mode", "paper") == "official":              # official test code: softmax_t(-dis * T)
            return torch.softmax(-dis * self._c("temperature", 50.0), -1)[:, -1]
        return dis[:, -1]


# ------------------------------------------------------------------------------------------------------------ TimesNet
class _Inception(nn.Module):
    def __init__(self, cin: int, cout: int, nk: int):
        super().__init__()
        self.k = nn.ModuleList([nn.Conv2d(cin, cout, 2 * i + 1, padding=i) for i in range(nk)])
        for m in self.k:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu"); nn.init.zeros_(m.bias)

    def forward(self, x):
        return torch.stack([c(x) for c in self.k], -1).mean(-1)


class _TimesBlock(nn.Module):
    def __init__(self, d: int, d_ff: int, k: int, nk: int):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(_Inception(d, d_ff, nk), nn.GELU(), _Inception(d_ff, d, nk))

    def forward(self, x):
        B, T, N = x.shape
        xf = torch.fft.rfft(x, dim=1)
        amp = xf.abs()
        freq = amp.mean(0).mean(-1)                                        # batch-level spectrum, as in TSLib FFT_for_Period
        freq[0] = 0
        k = min(self.k, len(freq) - 1)
        top = torch.topk(freq, k).indices.clamp_min(1).tolist()
        weights = amp.mean(-1)[:, top]                                     # (B, k) per-sample period amplitudes
        res = []
        for f in top:
            p = max(T // f, 1)
            Lp = math.ceil(T / p) * p
            h = torch.cat([x, x.new_zeros(B, Lp - T, N)], 1) if Lp > T else x
            h = h.reshape(B, Lp // p, p, N).permute(0, 3, 1, 2)            # 1D -> 2D: (B, N, #periods, period)
            h = self.conv(h).permute(0, 2, 3, 1).reshape(B, Lp, N)[:, :T]
            res.append(h)
        res = torch.stack(res, -1)
        w = torch.softmax(weights, 1)[:, None, None, :]                    # adaptive aggregation
        return (res * w).sum(-1) + x


class _TimesNet(nn.Module):
    def __init__(self, L: int, F: int, d: int, d_ff: int, layers: int, k: int, nk: int, dropout: float):
        super().__init__()
        self.emb = _DataEmb(F, d, max(L, 8), dropout)
        self.blocks = nn.ModuleList([_TimesBlock(d, d_ff, k, nk) for _ in range(layers)])
        self.norm, self.proj = nn.LayerNorm(d), nn.Linear(d, F)

    def forward(self, x):
        mu = x.mean(1, keepdim=True).detach()                              # non-stationary normalisation (TSLib anomaly_detection)
        sd = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5).detach()
        h = self.emb((x - mu) / sd)
        for blk in self.blocks:
            h = self.norm(blk(h))
        return self.proj(h) * sd + mu


@BASELINES.register("timesnet")
class TimesNet(_Reimpl):
    name = "timesnet"
    paper = "Wu, Hu, Liu, Zhou, Wang, Long. TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis. ICLR 2023"
    official_repo = "https://github.com/thuml/Time-Series-Library"

    def build(self, L, F):
        return _TimesNet(L, F, self._c("d_model", 32), self._c("d_ff", 32), self._c("e_layers", 1), self._c("top_k", 3),
                         self._c("num_kernels", 3), self._c("dropout", 0.1))

    def train_step(self, x, epoch):
        loss = Fn.mse_loss(self.net(x), x)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return float(loss)

    def score_x(self, x):
        return ((self.net(x) - x) ** 2).mean(-1)[:, -1]
