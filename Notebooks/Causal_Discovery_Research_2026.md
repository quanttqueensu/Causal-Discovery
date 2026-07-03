# Causal Discovery Research Notes (2026-07-03)

## Scope
This note summarizes practical and research-relevant directions for causal discovery in time series, with an emphasis on quantitative finance and how to extend this repository.

## 1) High-value methods to know

### PCMCI family (Tigramite)
- Tooling: Tigramite library for causal discovery, causal effect estimation, and time-series graph analysis.
- PCMCI: strong baseline for lagged causal discovery under causal stationarity and no hidden confounders.
- PCMCI+: extends PCMCI with contemporaneous edge orientation (CPDAG-style output for lag-zero relations).
- LPCMCI: designed for latent confounders; useful when market-wide hidden drivers are likely.
- RPCMCI: regime-dependent causal structure, useful under state changes (risk-on/risk-off, crisis periods).
- J-PCMCI+: multi-context/multi-dataset causal discovery.

Practical takeaway for markets:
- If you assume approximately stationary returns and mostly observed drivers, start with PCMCI+.
- If hidden common causes are plausible (they usually are), compare with LPCMCI.
- If you expect structural breaks/regimes, evaluate RPCMCI or regime-splitting plus PCMCI.

## 2) Why this matters in quant finance
- Correlation-based feature selection is unstable across regimes.
- Causal parent sets can produce more robust predictors by prioritizing direct drivers.
- Causal graphs can improve interpretation: which macro/sector signals transmit to assets and at which lags.
- Causal discovery can support portfolio construction, risk diagnostics, and scenario analysis.

## 3) Key references

### Foundations and software
- Tigramite repository: https://github.com/jakobrunge/tigramite
- Tigramite docs: https://jakobrunge.github.io/tigramite/
- PCMCI (Sci. Adv. 2019): https://www.science.org/doi/10.1126/sciadv.aau4996
- PCMCI+ (UAI 2020): http://auai.org/uai2020/proceedings/579_main_paper.pdf
- LPCMCI (NeurIPS 2020): https://proceedings.neurips.cc/paper/2020/hash/94e70705efae423efda1088614128d0b-Abstract.html
- Causal inference for time series review (2023): https://doi.org/10.1038/s43017-023-00431-y
- Bagged-PCMCI+ (CLeaR 2024): https://arxiv.org/abs/2306.08946

### Quant-finance-focused papers
- Causal Discovery in Financial Markets (CD-NOTS, nonstationarity): https://arxiv.org/abs/2312.17375
- Causality-Inspired Models for Financial TS Forecasting: https://arxiv.org/abs/2408.09960
- FX/Stock/Bond linkages (LPCMCI + VAR-LiNGAM): https://arxiv.org/abs/2310.16841
- Causal discovery in Hawkes processes (sovereign bonds): https://arxiv.org/abs/2206.06124

Additional candidate leads to track:
- A Causal Perspective of Stock Prediction Models: https://arxiv.org/abs/2503.20987
- Structural Causal Modelling for Economic Indicators: https://arxiv.org/abs/2509.07036
- DyCAST-Net: https://arxiv.org/abs/2507.09439
- FinCARE: https://arxiv.org/abs/2510.20221
- Causal Regime Detection in Energy Markets: https://arxiv.org/abs/2511.04361
- CausalAlpha: https://arxiv.org/abs/2606.07049

## 4) Method selection cheat sheet
- Data mostly linear, continuous, moderate sample:
  - PCMCI/PCMCI+ with ParCorr or RobustParCorr.
- Strong nonlinear effects:
  - PCMCI/PCMCI+ with GPDC or CMIknn (higher compute cost).
- Hidden confounding likely:
  - LPCMCI.
- Regime shifts/nonstationarity:
  - RPCMCI, or nonstationary alternatives like CD-NOD/CD-NOTS.
- Need uncertainty on discovered links:
  - Bagging/bootstrap over windows or Bagged-PCMCI+ style aggregation.

## 5) Immediate roadmap for this repository

Current repo status:
- A synthetic POC script already exists and plants known macro->stock lagged edges.
- PCMCI with ParCorr and BH-FDR correction is implemented.

Recommended next implementation steps:
1. Add rolling backtest mode
- Refit on each rolling window and forecast next-horizon sign/return.
- Compare causal-parent features vs naive lag features.

2. Add baseline comparison module
- Compare PCMCI (ParCorr) vs PCMCI+ vs LPCMCI on the same synthetic setup.
- Track precision/recall against planted graph and predictive utility.

3. Add uncertainty layer
- Bootstrap windows and aggregate discovered links by frequency.
- Keep only links above stability threshold (for example >= 60%).

4. Add regime-awareness
- Split sample by volatility regime and compare graph stability.
- Optionally evaluate RPCMCI if dependencies and runtime permit.

5. Add effect-size analysis
- Use Tigramite causal effect functionality after graph discovery.
- Distinguish adjacency strength from estimated intervention effect.

## 6) Common pitfalls (finance specific)
- Confusing predictive edge with tradable edge after costs/slippage.
- Ignoring hidden confounders when using vanilla PCMCI.
- Over-tuning tau_max, pc_alpha, and significance levels on one period.
- Treating one discovered graph as universal across market regimes.
- Skipping synthetic ground-truth validation before real market data.

## 7) Suggested experiment design
- Stage A (sanity): synthetic data with known graph and controlled confounding.
- Stage B (robustness): varying sample length, autocorrelation, and noise.
- Stage C (market data): evaluate by period splits and regime splits.
- Stage D (trading relevance): test whether causal features improve risk-adjusted performance out-of-sample.

## 8) Minimal reproducibility checklist
- Fixed random seeds.
- Explicit train/validation/test chronology.
- Report graph metrics (precision, recall, F1 for synthetic).
- Report forecasting metrics and turnover-adjusted returns for market runs.
- Log hyperparameters (tau_max, pc_alpha, alpha, CI test, window length).

## 9) Bottom line
For this project, the most practical path is:
- Keep PCMCI+ with ParCorr as baseline,
- Add bootstrap stability filtering,
- Add LPCMCI comparison for hidden-confounder robustness,
- Then test on regime-segmented market windows.

That sequence gives a clear research-to-production path while staying aligned with known constraints in financial time series.
