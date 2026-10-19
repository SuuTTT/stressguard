# StressGuard: Cost-Sensitive Invariant Anomaly Detection for Financial Trading Under Extreme Volatility Cascades

**Paper design, mathematical formulation, architecture and implementation blueprint.**
Target venues: KDD / NeurIPS / AAAI / ICML (CCF-A). Status of this document: research design + runnable blueprint; every number
reported in §10 comes from the synthetic multi-regime simulator and is a *pipeline sanity check*, not a benchmark result.

Directory mapping. `c225:/date/zyy` is the remote experiment node; this local checkout (`TradingSignal/`) is synced there and
run with the same relative paths, so the brief's layout maps as follows:

| Brief | Here | Content |
|---|---|---|
| `data/raw/` | `data/raw/<dataset>/` | raw benchmark files (see `stressguard/data/recipes.py`) |
| `data/stress_splits/` | `data/stress_splits/<dataset>.json` | Normal/Stress partition (rule + curated calendar), written by `scripts/sg_make_stress_splits.py` |
| `baselines/` | `stressguard/baselines/` + `configs/stressguard/baselines/` | score-only baseline contract, runnable baselines, 10 benchmark slots |
| `models/` | `stressguard/models/` + `configs/stressguard/{stressguard.yaml,ablations/}` | encoder, severity head, dynamic decision boundary, losses |
| `backtest/` | `stressguard/backtest/` (+ `warn2trade/backtest/costs.py`) | hedge-overlay execution with non-linear slippage |
| `results/` | `results/stressguard/<dataset>/<model>/seed*/` + `results/stressguard/{tables,figures}/` | per-fold JSON, predictions, decile AUC, cost curves, tables, figures |

---

## 1. Title, structured abstract, contributions

**Title.** StressGuard: Cost-Sensitive Invariant Anomaly Detection for Financial Trading Under Extreme Volatility Cascades

**Structured abstract (draft).**

*Background.* Deep anomaly detectors for crash and tail-risk warning report strong average performance, yet their discriminative
power collapses exactly when it matters: during volatility spikes, liquidity cascades and panic selling, AUC drops from the 0.7 range
toward random guessing. We trace this collapse to two causes. (i) The stress regime is a structural break: order-flow shortcuts that
predict crashes in calm markets become uninformative or sign-flipped under stress, and detectors trained by empirical risk
minimisation latch onto them. (ii) Detectors are trained and thresholded with symmetric criteria (BCE, F1) although the payoff is
violently asymmetric: a missed crash costs a drawdown, a false alarm costs a round trip of friction.

*Method.* StressGuard is an end-to-end framework with three components. (1) A **stress-invariant representation** trained with an
invariance penalty (IRMv1 or V-REx) across environments $\mathcal{E}=\{\text{Normal},\text{Stress}\}$ that are defined causally
from realised-volatility and drawdown-speed statistics (plus a curated event calendar for real data). (2) An **asymmetric economic
loss** whose per-sample weights are the forward crash severity $|\Delta P_{t+\tau}|$ for false negatives and the round-trip hedge
friction $\gamma_t$ for false positives, and the **Bayes-optimal dynamic threshold** $\tau^*_t = c_{FP}\gamma_t/(c_{FP}\gamma_t +
c_{FN}\hat s_t)$ it induces, which maps probabilities directly to a utility-optimal hedge ratio. (3) **Multi-scale regime
conditioning**: a FiLM prior from volatility term structure, book depth, spread, cross-asset correlation and implied-volatility skew
modulates detection sensitivity while the invariance penalty keeps the classifier regime-agnostic.

*Theory.* We show that the Bayes decision rule of the asymmetric loss is a per-bar threshold that rises with friction and falls with
predicted severity (Prop. 1); that the proper log surrogate and the brief's linear surrogate share this rule (Prop. 2); that the
hedge ratio maximising linear expected utility under a drawdown (CVaR) cap is the clipped normalised edge (Prop. 3); and, in a
structural model with a stress-dependent spurious feature, that the ERM solution's stress AUC degrades monotonically with the
spurious coefficient flip whereas the invariant solution's does not (Prop. 4, simulator-verified).

*Experiments.* Ten multi-asset datasets (LOBSTER, FI-2010, SPX volatility surface, Binance L2, CSI 300 futures, NYSE TAQ, FX, UST
curve, a curated flash-crash benchmark and a controlled multi-regime simulator) with strict Normal/Stress test partitions; ten
baselines from 2021–2026 under one shared decision layer and one shared non-linear execution simulator; purged, embargoed
walk-forward validation; metrics reported for Normal, Stress and Full samples.

*Results (targets, not yet achieved; see §10).* Target claims: stress AUC above 0.74 where ERM detectors fall to about 0.56; $\Delta\text{AUC}_{stress}$
reduced by more than half; maximum drawdown of the hedged portfolio reduced by up to 40 % at 1–5 bps plus non-linear slippage; the
dynamic rule dominates every static threshold on the economic cost curve for all $c_{FN}/c_{FP}$ ratios.

**Contributions.**
1. A diagnosis of robustness collapse as *spurious stress-dependent correlation + symmetric optimisation*, with a simulator that
   reproduces it under controlled parameters (Fig. 1A).
2. StressGuard: invariant representation + asymmetric economic loss + Bayes-optimal dynamic hedge rule + regime conditioning, with
   the derivations in §3.
3. A regime-sliced evaluation protocol ($\Delta\text{AUC}_{stress}$, tail P/R/F1, VaR/ES, economic cost curves) and an open
   benchmark suite with auditable stress partitions and a shared execution simulator.

---

## 2. Motivation figure (page 1)

`scripts/sg_motivation_fig.py` renders Fig. 1 from `results/stressguard/`:

* **Fig. 1A — Collapse.** AUC-ROC per *stress-score decile* (decile 1 = calmest, 10 = most extreme; stress score =
  trailing realised vol / train-fitted stress threshold) for ERM-trained detectors and classical scorers. Prior detectors show high
  AUC in deciles 1–7 and fall toward the dashed 0.5 line in deciles 9–10. StressGuard's curve is flat.
* **Fig. 1B — Frontier.** Annualised return vs |MDD| of the hedged equal-weight portfolio, one point per model plus the unhedged
  portfolio; dotted rays are iso-Calmar lines. StressGuard should sit up-left (higher return per unit of drawdown).
* **Fig. 2 — Economic cost curve.** Expected asymmetric cost on the Stress slice vs $c_{FN}/c_{FP}$ for the dynamic rule, the
  val-fitted static rule and the (cheating) test-fitted oracle static rule.

---

## 3. Formal problem formulation

### 3.1 Data and environments

For asset $i$ at bar $t$: observation window $x_{t,i}\in\mathbb{R}^{L\times F}$ (returns, order-flow imbalance, depth, spread,
LOB snapshot), context prior $c_{t,i}\in\mathbb{R}^{F_c}$ (multi-scale realised vol, depth, spread, cross-asset correlation,
IV skew), label $y_{t,i}\in\{0,1\}$ (a crash onset occurs within the next $\ell$ bars), severity
$$s_{t,i}=\big|\Delta P_{t+\tau}\big| := \max\Big(0,\,-\min_{1\le k\le\tau}\big[\log P_{t+k,i}-\log P_{t,i}\big]\Big)$$
(the worst forward drawdown inside the hedge horizon $\tau$), and friction $\gamma_{t,i}$ = round-trip hedge cost in log-return units
from the execution model (commission + half-spread + $\sqrt{\cdot}$ market impact + delay drift).

An **environment** $e\in\mathcal{E}$ is a set of bars with a common regime. Training environments
$\mathcal{E}_{tr}=\{\text{Normal},\text{Stress}\}$ are defined by a causal rule fitted on the training window only,
$$\text{Stress}_t = \mathbb{1}\big[\hat\sigma_t\ge Q_q(\hat\sigma)\big]\ \vee\ \mathbb{1}\Big[\textstyle\sum_{k<W}r_{t-k}\le -k_{dd}\hat\sigma_t\sqrt W\Big]\ \vee\ \mathbb{1}\big[\hat\alpha_t\le Q_{q'}(\hat\alpha)\big]\ \vee\ \text{Calendar}_t,$$
with $\hat\sigma_t$ the trailing realised vol of the cross-sectional median return, $\hat\alpha_t$ a trailing Hill tail index
(optional, lags), and the calendar the curated event list in `recipes.py`. Each environment induces a distribution
$P^e(x,c,y,s,\gamma)$. The test Stress environment may be *more severe than anything seen in training*.

### 3.2 Why robustness collapses (structural model)

Let $z^{inv}$ be a precursor whose mechanism has a regime-independent *direction* (e.g. pre-crash order-flow imbalance; its
noise level and the class prior may vary with $e$, so IRMv1 recovers it only approximately) and $z^{sp}$ a feature whose association with $y$ is regime-dependent,
$z^{sp}=\beta(u)\,y+\varepsilon$ with stress intensity $u\in[0,1]$ and $\beta(u)=\beta_0(1-\kappa u)$. In calm markets
$\beta(0)=\beta_0>0$ is large (cheap signal); under severe stress $\beta\to 0$ or flips sign. The ERM solution
$w_{ERM}=\arg\min\sum_e n_e\,\mathcal{L}^e$ is dominated by the calm environment and assigns weight to $z^{sp}$ in proportion to
$\beta_0$. On a test environment with $\beta(u_{test})<0$ the score $w^\top z$ contains an anti-predictive term, and
$\text{AUC}_{stress}$ decreases monotonically in $|\beta(u_{test})-\beta_0|$ (Prop. 4; the simulator in
`synthetic_regime.py` instantiates exactly this model, Fig. 1A). An invariant predictor must put zero weight on $z^{sp}$ because
no single classifier on top of it is simultaneously optimal in Normal and in Stress.

### 3.3 Objective

Representation $\Phi_\theta$, classifier $w$, severity head $q_\psi$ and the decision rule $d$. We minimise the worst-environment
**economic risk**
$$\min_{\theta,w,\psi}\ \max_{e\in\mathcal{E}}\ \mathcal{R}^e_{econ}(\theta,w),\qquad
\mathcal{R}^e_{econ}=\mathbb{E}_{P^e}\Big[c_{FN}\,y\,(1-d)\,s\;+\;c_{FP}\,(1-y)\,d\,\gamma\Big].$$
The IRM relaxation of the minimax (Arjovsky et al. 2019) with the brief's penalty is
$$\boxed{\ \min_{\theta}\ \sum_{e\in\mathcal{E}_{tr}}\mathcal{L}^e_{econ}(\Phi_\theta)\;+\;\lambda\,\big\|\nabla_{w\mid w=1.0}\,\mathcal{L}^e_{econ}(w\cdot\Phi_\theta)\big\|^2\ }$$
where $\mathcal{L}^e_{econ}$ is a differentiable surrogate of $\mathcal{R}^e_{econ}$ (§3.4). The alternative V-REx penalty
$\lambda\,\mathrm{Var}_e[\mathcal{L}^e_{econ}]$ is an ablation. Both are computed on *environment-balanced* mini-batches
(`env_balanced_loader`) so the rare Stress environment is never swamped.

### 3.4 Asymmetric economic loss and its surrogates

With $p=\sigma(w\cdot\Phi_\theta(x,c))$ and a per-sample loss $\ell(y,p)$,
$$\mathcal{L}_{econ}(y,p)=\sum_t\Big[c_{FN}\,y_t\,s_t\,\ell(1,p_t)\;+\;c_{FP}\,(1-y_t)\,\gamma_t\,\ell(0,p_t)\Big].$$
* **Linear surrogate** $\ell(1,p)=1-p,\ \ell(0,p)=p$ gives exactly the brief's
  $\mathcal{L}_{Econ}=\sum_t\big(c_{FN}\,y_t(1-\hat y_t)\,|\Delta P_{t+\tau}|+c_{FP}(1-y_t)\hat y_t\,\gamma_{friction}\big)$; it is
  the expected 0-1 economic cost and is *not* a proper scoring rule (its gradient $p(1-p)$ vanishes at saturation).
* **Log surrogate** $\ell(1,p)=-\log p,\ \ell(0,p)=-\log(1-p)$ is strictly proper and is the default for training
  (`train.loss.surrogate: log`; `linear_surrogate.yaml` is the ablation).
Weights are normalised by their batch mean so only the ratio $c_{FN}s_t : c_{FP}\gamma_t$ matters. Setting both weights to 1 and
$c_{FN}=c_{FP}$ recovers BCE (the symmetric ablation `no_econ.yaml`).

**Proposition 1 (Bayes-optimal dynamic threshold).** For a bar with calibrated crash probability $p$, expected severity
$\hat s=\mathbb{E}[s\mid x,c,y=1]$ and friction $\gamma$, the decision $d=1$ (hedge) minimises expected economic cost iff
$$p\,c_{FN}\,\hat s\ \ge\ (1-p)\,c_{FP}\,\gamma\quad\Longleftrightarrow\quad p\ \ge\ \tau^*(c)=\frac{c_{FP}\,\gamma}{c_{FP}\,\gamma+c_{FN}\,\hat s}.$$
*Proof.* Expected cost of $d=0$ is $p\,c_{FN}\hat s$; of $d=1$ is $(1-p)\,c_{FP}\gamma$; compare. $\square$
The threshold is **dynamic**: it rises when the book thins (larger $\gamma$ under stress) and falls when the predicted crash is
deeper (larger $\hat s$ under stress). A static F1-tuned threshold ignores both, which is why it is miscalibrated exactly in stress.

**Proposition 2 (surrogate consistency).** Let $\eta=P(y=1\mid x,c)$ and $w_1=c_{FN}s$, $w_0=c_{FP}\gamma$. (i) The weighted log
surrogate is strictly proper for the re-weighted posterior $p_w=w_1\eta/(w_1\eta+w_0(1-\eta))$, and $p_w\ge 1/2\iff\eta\ge\tau^*$.
(ii) The linear surrogate is Bayes-consistent for the *decision* $\mathbb{1}[\eta\ge\tau^*]$ (its pointwise minimiser is 0 or 1)
but is not a proper scoring rule and has no probabilistic minimiser. The log surrogate is therefore used for optimisation and the
linear one is reported as the economic cost itself. $\square$

**Calibration (needed to apply Prop. 1 after cost-sensitive training).** Minimising a cost-weighted proper loss with weights
$w_1=c_{FN}s$, $w_0=c_{FP}\gamma$ does not return $P(y\!=\!1\mid x,c)$ but the re-weighted posterior
$p_w=\tfrac{w_1 p}{w_1 p+w_0(1-p)}$ (Elkan 2001); environment-balanced sampling and negative sub-sampling shift it further.
Thresholding $p_w$ at $\tau^*$ therefore double-counts the costs (in our first run it hedged 99.8 % of bars). StressGuard keeps the
cost-weighted loss for *representation learning* (it concentrates capacity on expensive errors) and applies the Bayes rule to a
**Platt-calibrated** probability $p=\sigma(a\,\text{logit}+b)$ with $(a,b)$ fitted by logistic regression on the validation
fold and refit every epoch (`StressTrainer.fit_calibration`; the same calibration is applied to every baseline). Platt scaling
is an affine map of the logit, so it estimates $P(y\mid\text{score})$ on the validation distribution; it does not invert the
*per-sample* Elkan shift exactly (that would need $\eta=w_0p_w/(w_0p_w+w_1(1-p_w))$ with each bar's own weights, an alternative
we list as an ablation). Props. 1 and 3 are statements about a calibrated $\eta$; the implementation approximates it.

**Severity head.** $q_\psi(\Phi)\in\mathbb{R}^Q$ are monotone quantiles of $s\mid y=1$ (cumulative softplus), trained with the
pinball loss on positive samples; $\hat s$ is the configured quantile (median by default; a higher quantile is a more conservative
hedger). The plug-in $\hat s=k\,\hat\sigma_t\sqrt\tau$ is the `no_sev_head` ablation and the rule used for baselines.

### 3.5 Utility-optimal position sizing and drawdown control

Overlay PnL of hedging a fraction $h\in[0,h_{max}]$ of the holding for $\tau$ bars: $\pi(h)=-h\,R_\tau-\gamma h$ with
$R_\tau$ the realised holding-period return. Friction is paid on *every* hedge, so the PnL-neutral rule is $p\,\hat s\ge\gamma$,
i.e. $\tau_{PnL}=\gamma/\hat s$. The brief's $\mathcal{L}_{Econ}$ charges friction only on false alarms; its rule $\tau^*$ at
ratio $c_{FN}/c_{FP}=1$ is slightly more permissive than $\tau_{PnL}$ ($\gamma/(\gamma+\hat s)$ vs $\gamma/\hat s$) and ratios above 1
buy drawdown protection at a PnL cost. We therefore define the *economic edge* as a utility, not as expected PnL,
$$\text{edge}(p,\hat s,\gamma)=p\,c_{FN}\hat s-(1-p)\,c_{FP}\gamma,$$
and report the whole risk-aversion frontier over $c_{FN}/c_{FP}$ (Table 6) instead of a single ratio.

**Proposition 3 (hedge ratio).** Maximising $h\cdot\text{edge}-\tfrac{c_{FN}\hat s}{2h_{max}}h^2$ (a quadratic risk charge, e.g.
mean–variance utility with hedge-PnL variance $\propto(h\hat s)^2$) over $h\in[0,h_{max}]$ gives
$$h^*=h_{max}\,\frac{\big(p\,c_{FN}\hat s-(1-p)\,c_{FP}\gamma\big)_+}{c_{FN}\,\hat s}\in[0,h_{max}],\qquad h^*>0\iff p>\tau^*.$$
$h^*$ is continuous and piecewise differentiable (a.e.) in $(p,\hat s)$, so the drawdown term below back-propagates into the
detector. (With a *linear* risk charge the optimum would be bang-bang.) $\square$

**Drawdown control.** The training objective adds $\lambda_{CVaR}\,\mathrm{CVaR}_\alpha\big(-(1-h)R_\tau+\gamma h\big)$
(Rockafellar–Uryasev form) on the hedged holding-period return, which directly targets the left tail that MDD, VaR$_{99}$ and
ES$_{99}$ measure, and $h_{max}$ may be vol-targeted. Full objective (per environment-balanced batch):
$$\mathcal{L}=\frac1{|\mathcal{E}|}\sum_e\Big[\mathcal{L}^e_{econ}+\lambda_{IRM}\,\|\nabla_{w|w=1}\mathcal{L}^e_{econ}\|^2\Big]
+\lambda_{VREx}\mathrm{Var}_e\mathcal{L}^e_{econ}+\lambda_{sev}\mathcal{L}_{pinball}+\lambda_{CVaR}\,\mathrm{CVaR}_\alpha.$$

### 3.6 Multi-scale regime conditioning

$c_{t,i}$ = [log realised vol over 5/20/60 bars relative to its trailing median (volatility term structure), log depth and log
spread relative to trailing medians (LOB depletion, friction), trailing correlation with the market (contagion), and dataset
extras: 25-delta IV skew / VIX term slope / VVIX (SPX surface), funding rate and open interest (crypto), MOVE (rates, licensed)].
Severity $s_t$ and the FN cost are measured from the fill bar $t+\delta$ (the hedgeable window), not from the decision bar.
Conditioning is a FiLM layer $\Phi\leftarrow\Phi\odot(1+\tanh g(c))+b(c)$, zero-initialised. Two safeguards keep it a *prior*
rather than a leak: the invariance penalty is applied *after* modulation, and $c$ only contains trailing quantities. The friction
input $\gamma_t$ of the decision rule is computed from the same liquidity state, so sensitivity and threshold move together.

**Proposition 4 (sketch, simulator-verified).** In the model of §3.2 with Gaussian noise, the stress AUC of the ERM linear
classifier equals $\Phi\!\big(\mu^\top w/(\sqrt2\,\|w\|_\Sigma)\big)$ with $\mu=(a_{inv},\beta(u))$; for $w_{ERM}\propto(a_{inv},\beta_0)$ it
is decreasing in $\beta_0-\beta(u)$ and crosses 0.5 once $\beta(u)<-a_{inv}^2/\beta_0$, whereas for the invariant solution
$w\propto(a_{inv},0)$ the stress AUC is $\Phi(a_{inv}/(\sqrt2\,\sigma_{stress}))$, independent of $\beta(u)$ and degraded only by the
stress noise level. (Full proof in the appendix; Fig. 1A is the empirical counterpart.)

---

## 4. Architecture

```
            price window x (B,L,F) ──► PatchTST encoder ─┐
            LOB window   (B,L,Fl)  ──► dilated Conv1d ───┼─► concat ─► MLP fuse ─► h ─► FiLM(c) ─► LayerNorm ─► Φ (B,d)
            liquidity    (B,3)     ──► MLP ──────────────┘                              ▲
            context prior c (B,Fc) ─────────────────────────────────────────────────────┘
                                                                                     │
                                 ┌───────────────────────────────────────────────────┴───────────────────────────────┐
                                 ▼                                                                                   ▼
                     classifier head  logit, p = σ(logit)                                   severity head  q ∈ R^Q monotone quantiles, ŝ = q[idx]
                                 │                                                                                   │
                                 └──────────────► DYNAMIC DECISION BOUNDARY  τ*_t = c_FP γ_t / (c_FP γ_t + c_FN ŝ_t) ◄─────────┘
                                                              h_t = h_max (p c_FN ŝ − (1−p) c_FP γ)_+ / (c_FN ŝ)
                                                                              │
                                                  hedge-overlay execution (latency δ, hold τ, sqrt impact, halts) ─► PnL, hedged return
```
Modules and files: `stressguard/models/encoder.py` (InvariantEncoder + FiLM), `stressguard/models/stressguard.py` (heads +
decision boundary), `stressguard/models/losses.py` (surrogates, IRM/V-REx, pinball, CVaR, threshold, hedge ratio),
`stressguard/backtest/overlay.py` (execution), `stressguard/train/trainer.py` (environment-batched training).

**Training algorithm.**
```
fit regime rule on fold.train  →  env_t ∈ {0,1} for all t;  build context c (trailing only)
for epoch:  λ_IRM = λ_warm (epochs < warmup) else λ
  for env-balanced batch B:
     γ ← cost_model.round_trip(liq, σ)        Φ, p, q ← model(B)
     per-sample L_econ ← c_FN s ℓ(1,p) [y=1],  c_FP γ ℓ(0,p) [y=0]
     for e in {0,1}: L^e ← mean_{env=e} L_econ ;  P^e ← ‖∂/∂w L^e(w·logit)|_{w=1}‖²
     L ← mean_e L^e + λ_IRM mean_e P^e + λ_sev pinball(q, s | y=1) + λ_CVaR CVaR_α(−hedged(h))      (÷ λ_IRM if λ_IRM > 1)
  validate: Platt-calibrate the logit on val (a, b); score = −max_e L^e on val (worst-environment selection); early stop; keep best
fit static F1 threshold on val (for the static ablation and the cost-curve comparison)
```
Numerical note: the $\sqrt{\cdot}$ market-impact term has an infinite derivative at zero size; the overlay evaluates the cost on a
masked positive copy of $h$ so the CVaR term stays differentiable when most hedges are exactly zero.
Complexity is that of the backbone; the IRM penalty adds one extra backward through the logits per environment.

---

## 5. Datasets and stress partition

| # | Key | Source / licence | Bars | Stress calendar (partial) | Status |
|---|---|---|---|---|---|
| 1 | `lobster_sg` | LOBSTER NASDAQ L10 (academic licence) | 100 ms | 2020-03-09..20, 2024-08-05 | recipe + loader slot |
| 2 | `fi2010` | FI-2010 Nordic L10 (public) | 10 events | top-quintile vol days | recipe + loader slot |
| 3 | `spx_surface` | OptionMetrics / CBOE VIX, SKEW, VVIX | daily | 2010-05, 2015-08, 2018-02, 2020-03, 2022, 2024-08, 2025-04 | recipe + loader slot |
| 4 | `binance_l2` | Binance BTC/ETH depth (API / Tardis) | 1 min | 2020-03-12, 2021-05-19, UST 2022-05, 3AC, FTX, 2024-08-05, 2025-02-03 | recipe + loader slot |
| 5 | `csi300_futures` | CFFEX IF/IH ticks (vendor) | 1 min | 2015-06..09, 2016-01 breaker, 2020-02-03, 2024-02, 2024-10 | recipe + loader slot |
| 6 | `nyse_taq` | WRDS TAQ, S&P 100 | 1 min | 2010-05-06, 2015-08-24, 2018-02-05, 2020-03, 2024-08-05 | recipe + loader slot |
| 7 | `fxcm_fx` | FXCM / Dukascopy EURUSD, USDJPY (public) | 1 min | SNB 2015, Brexit, GBP flash, JPY flash 2019, 2020-03, MoF 2022-09, 2024-08 | recipe + loader slot |
| 8 | `ust_curve` | FRED par curve + MOVE (+ CME 10Y ticks) | daily | taper 2013, repo 2019, dash-for-cash 2020, gilt 2022, SVB 2023, 2025-04 | recipe + loader slot |
| 9 | `flashcrash_bench` | curated 1-min windows around 2010/2015/2019/2020/2021/2022/2024 events | 1 min | event windows | recipe + loader slot |
| 10 | `synthetic_regime` | multi-regime jump-diffusion × Hawkes × spurious-feature simulator | daily-like | planted episodes, severity cycle | **implemented, runnable** |

Partition protocol: the rule is calibrated on each fold's training window; the calendar is OR-ed in; the JSON in
`data/stress_splits/` lists every stress run with timestamps for audit. Leakage controls: purged + embargoed walk-forward
(purge ≥ max horizon + latency + execution window between train/val/test of a fold; folds roll forward, so a later fold's training
window may contain an earlier fold's test bars, as in any rolling scheme), train-only quantiles, trailing-only context (including
the simulator's skew proxy), forward quantities used only as targets or inside the backtest. The run-length smoothing of the regime
label and the top-5 % *severe* slice are defined ex post on the test window; neither is a model input. Calendar dates after 2024 must be re-verified against the volatility rule before use.

---

## 6. Baselines (one shared decision layer, one shared execution simulator)

| # | Slot | Paper | Verification | How it runs |
|---|---|---|---|---|
| 1 | `anomaly_transformer` | Xu et al., ICLR 2022 | verified, official code | adapter (`wrapper-needed`) |
| 2 | `tranad` | Tuli et al., VLDB 2022 | verified, official code | adapter |
| 3 | `timesnet` | Wu et al., ICLR 2023 | verified (Time-Series-Library) | adapter |
| 4 | `dcdetector` | Yang et al., KDD 2023 | verified, official code | adapter |
| 5 | `moderntcn` | Luo & Wang, ICLR 2024 | verified, official code | adapter |
| 6 | `diffusion_ad` | "Diffusion-AD (NeurIPS 2024/25)" | **unverified title** → substitute D3R (NeurIPS 2023) / ImDiffusion (VLDB 2023) | placeholder |
| 7 | `patchtst_ad` / `crossformer` | PatchTST (ICLR 2023), Crossformer (ICLR 2023) with recon / forecast-error head | backbones verified; "PatchTST-AD 2025" unverified | adapter |
| 8 | `regime_gnn_ad` | "Regime-Informed GNN-AD (2025)" | **unverified** → substitute GDN (AAAI 2021) + regime gate | placeholder |
| 9 | `mamba_finad` | "[2026] Mamba-FinAD" | **unverified** → selective-SSM backbone (Mamba) + recon head; S-Mamba / MambaTS (2024) | placeholder |
| 10 | `unifin_llm` | "[2026] UniFin-LLM / CausalTS-Regime" | **unverified** → MOMENT (ICML 2024), UniTS (NeurIPS 2024), Chronos (2024) zero-shot AD | placeholder |

Runnable today without external code: `erm` (same backbone, BCE, static threshold: the collapse baseline), `recon_ae`,
`dlinear_recon`, `iforest`, `zscore`. Every baseline is scored, Platt-calibrated on validation, then decided either by the
val-fitted best-F1 threshold (what prior work does) or by our plug-in dynamic rule (`--decision dynamic`), which isolates the
contribution of the decision layer from that of the representation. Numbers must never be reported under an unverified name.

---

## 7. Evaluation protocol (every metric × {Normal, Stress, Full})

* **Detection:** AUC-ROC, AUC-PR, P@1 %, R@1 %, Tail-F1 (F1 at the top-1 % operating point), best-F1, alert precision/recall at the
  model's own rule; $\Delta\text{AUC}_{stress}=\text{AUC}_{normal}-\text{AUC}_{stress}$; AUC by stress-score decile.
* **Trading (hedged vs unhedged equal-weight portfolio):** net PnL, ARR, Sharpe, Sortino, Calmar, MDD, VaR$_{99}$, ES$_{99}$,
  turnover, hedge rate, frozen rate; all bars and Stress bars only; MDD reduction. Cost grid {1, 5, 10, 30} bps on top of
  half-spread, $\sqrt{\cdot}$ impact (depth-aware, so slippage grows under stress) and delay drift; models are trained at 5 bps only.
* **Economic cost curve:** FN cost $c_{FN}\,y(1-d)s$ vs FP cost $c_{FP}(1-y)d\gamma$ for $c_{FN}/c_{FP}\in\{1,2,5,10,20,50\}$,
  dynamic vs val-fitted static vs test-fitted oracle static rule.
* **Statistics:** 4 seeds × all walk-forward folds; mean ± std; paired tests across folds; probabilistic / deflated Sharpe for the
  trading claims.

**Ablations** (`configs/stressguard/ablations/`): `erm`, `no_irm`, `vrex`, `no_econ`, `static_thr`, `no_ctx`, `no_cvar`,
`no_sev_head`, `linear_surrogate`, `no_env_balance`; plus every baseline with `--decision dynamic`.

---

## 8. Implementation blueprint

```
stressguard/
  data/regimes.py          RegimeLabeler (vol / tail / drawdown rule, train-only quantiles), calendar overlay, stress-split IO
  data/context.py          multi-scale causal context prior c_t (+ dataset extras)
  data/synthetic_regime.py dataset 10: Markov stress intensity × Hawkes jumps × invariant precursor × spurious feature × LOB depletion
  data/recipes.py          datasets 1–9: sources, licences, bar/feature recipes, stress calendars (raise with instructions if raw data absent)
  data/dataset.py          RegimeEventDataset (env, ctx, severity, stress score) + environment-balanced sampler
  models/encoder.py        InvariantEncoder = PatchTST + LOB conv + liquidity MLP + FiLM(c)
  models/stressguard.py    heads + dynamic decision boundary + hedge ratio; attach_gamma()
  models/losses.py         cost-sensitive surrogates, IRMv1, V-REx, pinball, CVaR, bayes_threshold, hedge_ratio
  backtest/overlay.py      overlay_pnl (differentiable) + OverlayBacktester (per-bar booking, turnover)
  train/trainer.py         StressTrainer: env-batched objective, IRM annealing, worst-env selection, static-threshold fit, predict
  eval/metrics.py          regime-sliced detection, decile AUC, trading block with VaR/ES, overlay portfolio, cost curve
  baselines/               ScoreBaseline contract, Platt calibration, runnable baselines, 10 benchmark slots with status
configs/stressguard/       default.yaml, stressguard.yaml, ablations/*.yaml, baselines/*.yaml, datasets/<key>.yaml (own namespace)
scripts/                   sg_make_stress_splits.py, sg_train.py, sg_baseline.py, sg_make_tables.py, sg_motivation_fig.py, sg_run_all.sh
tests/test_stressguard.py  9 unit/smoke tests (simulator, regime rule, threshold/hedge consistency, IRM penalty, end-to-end training)
```
Commands:
```bash
python3 tests/test_stressguard.py
python3 scripts/sg_make_stress_splits.py --dataset synthetic_regime
python3 scripts/sg_train.py --dataset synthetic_regime --seed 42
python3 scripts/sg_train.py --config configs/stressguard/ablations/erm.yaml --dataset synthetic_regime --seed 42
python3 scripts/sg_baseline.py --config configs/stressguard/baselines/recon_ae.yaml --dataset synthetic_regime --seed 42 --decision static
python3 scripts/sg_make_tables.py && python3 scripts/sg_motivation_fig.py
bash scripts/sg_run_all.sh synthetic_regime            # 4 seeds × {model, 9 ablations, 4 baselines × 2 rules}
```
Key knobs: `model.decision.{c_fn,c_fp,h_max,sev_q_index}`, `train.loss.{invariance,lam_irm,irm_warmup,lam_cvar,surrogate}`,
`regime.labeler.{vol_quantile,use}`, `cost.{commission_bps,impact_eta,notional}`, `eval.{cost_grid_bps,cost_ratios}`.
Adding a real dataset = implement `build_panel()` in `recipes.py` following its recipe string; everything downstream is shared.
Adding an external baseline = subclass `ScoreBaseline` with `fit`/`score`. Compute: the simulator runs on CPU in minutes per
seed; LOB-scale datasets need a GPU and `tail_stride`/memmap panels.

---

## 9. Threats to validity (to state in the paper)

* Numbers in §10 are from the simulator whose spurious-feature mechanism is *designed* to break ERM; real-data gains must be
  shown on datasets 1–9 before any claim. The simulator is for mechanism isolation and controlled sweeps (flip strength
  $\kappa$, noise growth, episode severity), not for headline numbers.
* The rule-based environment partition is a proxy; evaluation slices on real data should also be reported on the curated calendar
  alone, and on the simulator on the ground-truth regimes (`regime.source: truth`).
* Five of the ten baseline names in the brief could not be verified as papers; the substitutes in §6 are what will be cited.
* Hedge-overlay PnL depends on the cost model; all cost parameters are reported and swept.

---

## 10. Simulator results (seed 42, 3 purged walk-forward folds, 5 bps commission + half-spread + sqrt impact)

Two simulator settings were run. They differ only in the strength of the invariant precursor relative to the shortcut
(`ofi_strength`, `sp_beta0`); everything else (regime path, Hawkes crashes, LOB depletion, flip strength `sp_flip = 1.6`,
precursor noise growth `noise_mult = 0.5`) is identical. The contrast is the most useful thing the simulator produced.

### 10.1 Setting v0 (archived under `results/stressguard_simv0/`): weak precursor, strong shortcut

`ofi_strength = 1.0`, `sp_beta0 = 1.2`. In calm markets the shortcut is the better feature; under severe stress the precursor's
signal-to-noise ratio is so low that an *oracle* detector restricted to invariant features has a within-severe-stress AUC ceiling
of roughly 0.65.

| Model | AUC normal (3-fold) | AUC stress (rule slice) | AUC true-severe, fold 0 (u ≥ 0.6, shortcut sign flipped) | ΔAUC when the shortcut `sp` is permuted (normal) | ΔAUC when the precursor `ofi` is permuted (stress) |
|---|---|---|---|---|---|
| ERM same backbone | 0.894 / 0.907 / 0.859 | 0.776 / 0.864 / 0.822 | **0.895 → 0.589** | 0.177 / 0.092 / 0.142 | 0.105 / 0.071 / 0.061 |
| StressGuard | 0.808 / 0.793 / 0.818 | 0.775 / 0.698 / 0.827 | 0.810 → 0.621 | 0.001 / 0.009 / 0.003 | 0.001 / 0.010 / 0.010 |

Three findings. (i) **The collapse is real but only visible on the right slice.** On the rule-based Stress slice (top-10 % trailing
vol, precision 0.5 against the simulator's truth) ERM loses 0.04–0.12 AUC; on bars with true severe stress, where the shortcut's
sign has flipped, it falls from 0.90 to 0.59, i.e. the 0.71 → 0.56 pattern of the brief. The paper must therefore report a
*severe* slice (top-5 % stress score within the test window, plus the curated calendar on real data), not only "stress".
(ii) **The invariance machinery works as intended on the shortcut**: permuting `sp` moves StressGuard by ≤ 0.01 AUC versus
0.09–0.18 for ERM. (iii) **But with a weak precursor it has nothing to replace the shortcut with**: permuting `ofi` also leaves
StressGuard unchanged, so the invariant model degenerated into a regime detector (it ranks stress bars above calm bars from
volatility, depth and spread, which gives 0.8 AUC overall and 0.62 within severe stress) and pays 0.09 AUC in calm markets for a
0.03 gain in severe stress. Single-component ablations on fold 0 confirm that no single term causes this: without IRM
0.809/0.759 (normal/stress), without environment balancing 0.790/0.770, without the context prior 0.863/0.712 (the prior is what
lets the model shortcut through the regime). This is a no-free-lunch statement the paper should make explicitly: *invariance can
only help where an invariant precursor with usable stress-time SNR exists*; the simulator sweep over `noise_mult` and
`ofi_strength` maps that boundary (§10.3).

Trading in v0 (hedge overlay at c_FN/c_FP = 5, three folds): the unhedged equal-weight portfolio returns −16 %, −14 %, −38 %
annualised with MDD −34 %, −29 %, −61 % (the windows contain the planted crash episodes). ERM with the val-fitted static
threshold hedges 2–7 % of bars and reaches +27 %, +9 %, +27 % ARR with MDD −18 %, −13 %, −24 %. StressGuard at ratio 5 hedged
45 %, 15 %, 13 % of bars (the calibrated Bayes rule fires whenever p > ≈ 0.045) and reached −4 %, −17 %, +2 % with MDD −25 %,
−36 %, −25 %; re-applying its rule at ratio 1–2 (§3.5, `ratio_sweep`) cuts the hedge rate to 5–10 % but the detector itself is too
weak within stress in this setting. Two consequences for the method: the *ratio frontier* (Table 6) is the right object to
report rather than one ratio, and calibration must be refit per regime shift (the validation fold precedes the stress episode).

### 10.2 Setting v1 (default, `results/stressguard/`): precursor as strong as the shortcut in calm markets

`ofi_strength = 2.0`, `sp_beta0 = 2.0`; causal skew proxy; severity measured from the fill bar; capped bar-level hedge book.
Seed 42 only, so every number is a single-seed point estimate (3 folds for the main models and baselines, fold 0 for ablations).

| Model | AUC normal | AUC stress | AUC severe (top 5 %) | AUC true-severe | Hedged ARR (base −22.7 %) | Hedged MDD (base −41.5 %) | Hedged ES99 (base 0.030) |
|---|---|---|---|---|---|---|---|
| ERM same backbone, static threshold | **0.962** | **0.908** | **0.863** | **0.766** | **−5.3 %** | **−15.1 %** | 0.015 |
| StressGuard (full) | 0.832 | 0.757 | 0.634 | 0.691 | −20.2 % | −38.6 % | 0.024 |
| z-score rule | 0.861 | 0.739 | 0.650 | 0.635 | −8.0 % | −20.2 % | 0.015 |
| Recon-AE | 0.809 | 0.744 | 0.567 | 0.624 | −10.6 % | −25.0 % | 0.022 |
| Isolation Forest | 0.779 | 0.725 | 0.572 | 0.634 | −10.3 % | −23.5 % | 0.018 |
| DLinear-recon | 0.819 | 0.689 | 0.670 | 0.606 | −9.8 % | −21.7 % | **0.012** |
| Ablations (fold 0): w/o L_econ | 0.909 | 0.851 | 0.808 | 0.672 | | | |
| w/o IRM | 0.859 | 0.783 | 0.739 | 0.681 | | | |
| V-REx instead of IRM | 0.845 | 0.786 | 0.752 | 0.695 | | | |
| w/o context prior | 0.879 | 0.740 | 0.696 | 0.663 | | | |

**The method as specified is not supported by the simulator.** Stated plainly:

1. *The collapse exists.* ERM falls from 0.96 (normal) to 0.77 on true-severe bars, and every unsupervised scorer falls to 0.57–0.67
   on the severe slice. Permuting the shortcut costs ERM 0.13 AUC in calm markets, and on true-severe bars *raises* its AUC
   (0.70 → 0.77, 0.83 → 0.86): the shortcut is actively harmful there.
2. *The invariance penalty removes the shortcut* (StressGuard's drop when `sp` is permuted is 0.00–0.05) *but StressGuard still loses
   to ERM on every slice, including true-severe (0.69 vs 0.77).* It also barely uses the precursor (drop 0.01–0.06 on true-severe
   when `ofi` is permuted, versus 0.26–0.31 for ERM). ERM, despite the shortcut, extracts far more from the precursor.
3. *The asymmetric severity-weighted loss is the main cause*: replacing it by symmetric BCE while keeping IRM, the context prior and
   the dynamic rule gives 0.909 / 0.851 / 0.808, the best of our variants. Severity weighting concentrates the loss on a few large
   crashes and on the stress environment's base rate, which rewards regime detection over precursor detection.
4. *The dynamic rule beats the val-fitted static threshold on expected economic cost in the Stress slice for StressGuard at ratios
   ≥ 5* (0.0129 vs 0.0149 at ratio 5; 0.0122 vs 0.0387 at ratio 20) and for ERM at every ratio, but it never reaches the per-slice
   oracle static threshold, and on realised PnL the plug-in dynamic rule is worse than the static rule for all four baselines
   (it hedges 56–99 % of bar-assets). The risk-aversion frontier is nearly flat in MDD for both models.
5. Mechanism sweeps (fold 0): with no precursor noise growth or a stronger flip, ERM stays ahead (0.949 / 0.861 and 0.949 / 0.828
   vs StressGuard 0.877 / 0.791 and 0.873 / 0.782).

**What this means for the paper.** The diagnosis (contribution 1) and the evaluation protocol (contribution 3) stand. The method
(contribution 2) needs redesign before any real-data run: (a) decouple *what is learned* from *how it is decided*: train the
detector with a symmetric proper loss plus the invariance penalty and put all cost asymmetry in the decision layer; (b) remove the
class-prior shift between environments from the invariance penalty (per-environment label balancing or a per-environment
intercept), since IRMv1 penalises the heteroscedastic precursor as well as the shortcut; (c) calibrate per regime; (d) size hedges
so the frontier actually moves MDD. Targets such as "stress AUC > 0.74" and "MDD −40 %" in the abstract are aspirations, not results.

### 10.3 Artefacts

`results/stressguard/tables/tables.md` (Tables 1–6), `results/stressguard/figures/fig1_motivation_synthetic_regime.{png,pdf}`,
`fig2_cost_curve_synthetic_regime.{png,pdf}`, per-run `reliance_*.json`; v0 under `results/stressguard_simv0/`; sweeps under
`results/stressguard_sweeps/`.
