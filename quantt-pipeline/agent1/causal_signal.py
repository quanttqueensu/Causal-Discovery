from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

log = logging.getLogger(__name__)

MACRO_TICKERS = ["SPY", "IWM", "TLT", "IEF", "LQD", "HYG", "GLD", "USO", "UUP", "^VIX"]
SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLI", "XLB", "XLE", "XLU", "XLRE", "XLC"]
STOCK_TICKERS = [
    # tech / comm
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "INTC",
    "CSCO", "QCOM", "TXN", "IBM", "NOW", "INTU", "AMAT", "MU", "ADI", "LRCX", "KLAC", "PANW",
    "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS",
    # financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "SPGI", "CB", "PGR", "USB", "PNC", "BK",
    # health care
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY", "AMGN", "GILD", "CVS", "MDT", "ISRG", "VRTX",
    # consumer
    "WMT", "HD", "COST", "PG", "KO", "PEP", "MCD", "NKE", "SBUX", "LOW", "TGT", "BKNG", "CL", "MDLZ", "EL", "TJX",
    # industrials / materials
    "CAT", "BA", "GE", "HON", "UNP", "UPS", "RTX", "LMT", "DE", "MMM", "EMR", "ETN", "NSC", "FDX", "LIN", "APD", "SHW", "FCX", "NEM",
    # energy / utilities
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "OXY", "NEE", "DUK", "SO", "D",
    # travel / misc
    "DAL", "UAL", "LUV", "MAR", "GM", "F", "V", "MA", "PYPL", "ADP",
]

@dataclass
class SignalConfig:

    lookback_years: int = 8     # raw daily history pulled before resampling to weekly
    return_type: str = "log"    # "log" | "simple"
    standardize: bool = True
    winsor_q: float = 0.0
    vol_normalize: bool = False
    tau_min: int = 1
    tau_max: int = 4             # periods (weeks) of memory
    pc_alpha: float = 0.05
    alpha: float = 0.05          # BH level defining the tier-1 (strict) graph
    gate_mode: str = "fdr"       # "fdr" | "raw" | "none" — A/B switch on the significance gate
    fdr_scope: str = "per_target"    # "per_target" | "global" (only read when gate_mode="fdr")
    graph_shape: str = "macro_drives"  # "macro_drives" | "plus_xstock" | "full"
    edge_weight: str = "mci"     # "mci" | "sign" | "mci_over_p"
    aggregate: str = "sum"       # "sum" | "mean"
    window: int = 300            # trailing weekly bars used to fit PCMCI (~6y)

    use_sector_factors: bool = True   # append SECTOR_ETFS to the exogenous block
    min_ticker_coverage: float = 0.98  # drop tickers present on < this share of rows

    # --- parent selection (eligibility) ---
    parent_selection: str = "top_k"   # "top_k" | "bh_stability" (legacy hard gate)
    top_k_parents: int = 4            # parents kept per stock target under top_k
    p_floor: float = 0.10             # loose sanity floor on candidate p-values
    topk_stability_floor: float = 0.40   # loose sanity floor on bootstrap stability
    topk_relax_if_empty: bool = True  # if the floors leave nothing, take best-k by p anyway
    score_universe: str = "all"       # "all" | "scored_only" (legacy sparse dict)

    # --- stability / tier-1 evidence knobs ---
    use_bootstrap: bool = True   # wire block-bootstrap stability into the live path
    boot_samples: int = 50       # bootstrap resamples (~50 resolves a 0.40 floor fine)
    stability_floor: float = 0.60   # tier-1 requires stability >= this (no longer gates eligibility)
    weight_by_stability: bool = True  # multiply edge weight by its stability score
    tier_weight_bonus: float = 0.5    # extra weight multiplier for tier-1 (BH+stability) edges
    boot_seed: int = 42

    # --- scoring-mode knobs ---
    score_mode: str = "gated_corr"   # "causal" | "gated_corr" | "blend"
    corr_lookback: int = 52          # weeks used to estimate lagged lead-lag correlations
    blend_w: float = 0.5             # weight on causal rank when score_mode="blend"

    # --- cross-sectional normalization (applied to the final score) ---
    xs_normalize: str = "rank"       # "rank" | "zscore" | "none"
    xs_winsor: float = 3.0           # cap robust z at +/- this many MADs (0 disables)
    min_scored_for_rank: int = 20    # below this, rank-centering degenerates -> use zscore

    def __post_init__(self) -> None:
        if self.fdr_scope == "none":
            if self.gate_mode == "fdr":
                self.gate_mode = "raw"
            log.warning(
                "SignalConfig: fdr_scope='none' is deprecated and never disabled the "
                "gate — it applied uncorrected p<=alpha. Rewriting to "
                "gate_mode=%r, fdr_scope='per_target'. Set gate_mode explicitly "
                "('fdr' | 'raw' | 'none') instead.", self.gate_mode,
            )
            self.fdr_scope = "per_target"
        if self.gate_mode not in {"fdr", "raw", "none"}:
            raise ValueError(
                f"unknown gate_mode: {self.gate_mode!r} (expected 'fdr', 'raw' or 'none')"
            )
        if self.fdr_scope not in {"per_target", "global"}:
            raise ValueError(
                f"unknown fdr_scope: {self.fdr_scope!r} (expected 'per_target' or 'global')"
            )


def _to_returns(px: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    """Prices -> weekly returns."""
    px = px.resample("W-FRI").last().dropna(how="any")
    return (np.log(px).diff() if cfg.return_type == "log" else px.pct_change()).dropna()


def build_panel(
    cfg: SignalConfig,
    tickers: list[str] | None = None,
    macro_tickers: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], int]:

    import yfinance as yf

    macro_tickers = macro_tickers or (
        MACRO_TICKERS + (SECTOR_ETFS if cfg.use_sector_factors else [])
    )
    tickers = tickers or STOCK_TICKERS
    all_tickers = macro_tickers + tickers

    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=cfg.lookback_years)
    raw = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]

    present = [t for t in all_tickers if t in raw.columns]  # yf omits columns it could not fetch
    dropped_missing = sorted(set(all_tickers) - set(present))
    px = raw[present].dropna(how="all", axis=1)

    coverage = px.notna().mean()
    low_coverage = sorted(coverage[coverage < cfg.min_ticker_coverage].index)
    px = px.drop(columns=low_coverage)

    n_rows_before = len(px)
    px = px.dropna(how="any")  # align calendar

    if dropped_missing:
        log.warning("build_panel: no data returned for %d ticker(s): %s",
                    len(dropped_missing), ", ".join(dropped_missing))
    if low_coverage:
        log.warning("build_panel: dropped %d ticker(s) below %.0f%% coverage: %s",
                    len(low_coverage), cfg.min_ticker_coverage * 100, ", ".join(low_coverage))
    log.info("build_panel: %d/%d tickers kept; %d -> %d daily rows after calendar align",
             px.shape[1], len(all_tickers), n_rows_before, len(px))

    rets = _to_returns(px, cfg)
    ordered_cols = [c for c in macro_tickers if c in px.columns] + [c for c in px.columns if c not in macro_tickers]
    rets = rets[ordered_cols]
    n_macro = sum(1 for t in macro_tickers if t in px.columns)
    log.info("build_panel: %d weekly bars x %d series (%d exogenous + %d stocks) [%s -> %s]",
             len(rets), len(ordered_cols), n_macro, len(ordered_cols) - n_macro,
             rets.index[0].date() if len(rets) else "?",
             rets.index[-1].date() if len(rets) else "?")
    return rets, ordered_cols, n_macro


def preprocess(arr: np.ndarray, cfg: SignalConfig) -> np.ndarray:
    """arr: (T_window, N) raw returns -> transformed copy. Stats computed within the
    window only (no look-ahead)."""
    x = np.asarray(arr, dtype=float).copy()
    if cfg.vol_normalize:
        vol = np.std(x, axis=0, keepdims=True)
        vol[vol == 0] = 1.0
        x = x / vol
    if cfg.winsor_q and cfg.winsor_q > 0:
        lo = np.quantile(x, cfg.winsor_q, axis=0, keepdims=True)
        hi = np.quantile(x, 1 - cfg.winsor_q, axis=0, keepdims=True)
        x = np.clip(x, lo, hi)
    if cfg.standardize:
        mu = x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        x = (x - mu) / sd
    return x


def allowed_sources(cfg: SignalConfig, n_macro: int, n_vars: int, j: int) -> list[int]:
    """Candidate parents of target j under the graph_shape policy."""
    if cfg.graph_shape == "full" or (cfg.graph_shape == "plus_xstock" and j >= n_macro):
        return list(range(n_vars))
    if j < n_macro:                       # macros parented only by macros
        return list(range(n_macro))
    return list(range(n_macro)) + [j]     # macro_drives: macros + own lag


def build_link_assumptions(cfg: SignalConfig, n_macro: int, n_vars: int) -> dict:
    la = {}
    for j in range(n_vars):
        d = {}
        for i in allowed_sources(cfg, n_macro, n_vars, j):
            for tau in range(cfg.tau_min, cfg.tau_max + 1):
                d[(i, -tau)] = "-?>"
        la[j] = d
    return la


def _bh_accept(pvals: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg -> boolean accept mask (controls FDR at `alpha`)."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    if m == 0:
        return np.zeros(0, bool)
    order = np.argsort(pvals)
    passed = pvals[order] <= alpha * (np.arange(1, m + 1) / m)
    k = np.where(passed)[0]
    acc = np.zeros(m, bool)
    if len(k):
        acc[order[: k.max() + 1]] = True
    return acc


def _target_links(cfg: SignalConfig, n_macro: int, n_vars: int, j: int) -> list[tuple[int, int]]:
    """Candidate (source, lag) links tested for target j. Single definition shared
    by the BH loop, top-k selection and the diagnostics so the three can never
    disagree about what the candidate set was."""
    return [
        (i, tau)
        for i in allowed_sources(cfg, n_macro, n_vars, j)
        for tau in range(cfg.tau_min, cfg.tau_max + 1)
        if i != j
    ]


def significance_mask(
    p: np.ndarray,
    pcmci: PCMCI | None,
    cfg: SignalConfig,
    n_macro: int,
    n_vars: int,
) -> np.ndarray:

    mask = np.zeros_like(p, dtype=bool)
    mode = cfg.gate_mode
    global_accept = None

    if mode == "fdr" and cfg.fdr_scope == "global":
        q = pcmci.get_corrected_pvalues(
            p_matrix=p, fdr_method="fdr_bh", tau_min=cfg.tau_min, tau_max=cfg.tau_max
        )
        # Intersected with the candidate set below. NOTE: the correction itself is
        # still computed over the whole p_matrix, including links link_assumptions
        # excluded (tigramite fills those rather than masking them), so global BH's
        # denominator is larger than the candidate set. Separate fix.
        global_accept = q <= cfg.alpha

    for j in range(n_vars):
        links = _target_links(cfg, n_macro, n_vars, j)
        if not links:
            continue
        if mode == "none":
            acc = np.ones(len(links), dtype=bool)
        elif mode == "raw":
            acc = np.array([p[i, j, tau] for i, tau in links]) <= cfg.alpha
        elif global_accept is not None:
            acc = np.array([global_accept[i, j, tau] for i, tau in links])
        else:
            acc = _bh_accept(np.array([p[i, j, tau] for i, tau in links]), cfg.alpha)
        for (i, tau), a in zip(links, acc):
            mask[i, j, tau] = a
    return mask


def select_parents_top_k(
    p: np.ndarray,
    stability: np.ndarray,
    cfg: SignalConfig,
    n_macro: int,
    n_vars: int,
    targets: list[int] | None = None,
) -> np.ndarray:
    mask = np.zeros_like(p, dtype=bool)
    targets = range(n_vars) if targets is None else targets

    for j in targets:
        links = _target_links(cfg, n_macro, n_vars, j)
        if not links:
            continue
        pv = np.array([p[i, j, tau] for i, tau in links])
        sv = np.array([stability[i, j, tau] for i, tau in links])

        eligible = np.where((pv <= cfg.p_floor) & (sv >= cfg.topk_stability_floor))[0]
        if len(eligible) == 0:
            if not cfg.topk_relax_if_empty:
                continue
            eligible = np.arange(len(links))  # floors were too strict; fall back to best-by-p

        chosen = eligible[np.argsort(pv[eligible], kind="stable")[: cfg.top_k_parents]]
        for c in chosen:
            i, tau = links[c]
            mask[i, j, tau] = True
    return mask


@dataclass
class FitDiagnostics:
    """Per-stage edge counts for one PCMCI fit.

    Exists because the previous pipeline was opaque: a 3/118 outcome gave no way
    to tell whether BH-FDR, the bootstrap floor, or the data itself was binding.
    Every count below is restricted to stock targets — macro/sector targets are
    fitted but never scored.
    """

    n_vars: int
    n_macro: int
    n_stocks: int
    window: int
    graph_shape: str
    parent_selection: str
    gate_mode: str
    n_candidate_links: int      # candidate (src, lag) links summed over stock targets
    n_bh_accepted: int          # cleared the significance gate alone (see gate_mode)
    n_stability_accepted: int   # cleared the stability floor alone
    n_jointly_accepted: int     # tier 1: cleared both
    n_selected: int             # links actually feeding scores
    stocks_with_parent: int
    p_deciles: list[float]
    stab_deciles: list[float]
    parents_per_stock: tuple[int, float, int]   # (min, median, max)
    distinct_parent_pairs: int  # |{(src, lag)}| across selected edges
    top_parent_share: float     # share of selected edges using the modal (src, lag)
    fit_seconds: float
    boot_seconds: float


def collect_diagnostics(
    p: np.ndarray,
    stability: np.ndarray,
    bh_mask: np.ndarray,
    stab_mask: np.ndarray,
    tier1: np.ndarray,
    selected: np.ndarray,
    cfg: SignalConfig,
    n_macro: int,
    n_vars: int,
    fit_seconds: float,
    boot_seconds: float,
) -> FitDiagnostics:
    """Summarize one fit. Restricted to stock targets throughout."""
    stock_idx = list(range(n_macro, n_vars))
    idx = [(i, j, tau) for j in stock_idx for i, tau in _target_links(cfg, n_macro, n_vars, j)]

    def _count(m: np.ndarray) -> int:
        return int(sum(bool(m[i, j, tau]) for i, j, tau in idx))

    pv = np.array([p[i, j, tau] for i, j, tau in idx]) if idx else np.zeros(0)
    sv = np.array([stability[i, j, tau] for i, j, tau in idx]) if idx else np.zeros(0)
    deciles = np.arange(0, 1.01, 0.1)

    per_stock = {j: 0 for j in stock_idx}
    pairs: dict[tuple[int, int], int] = {}
    for i, j, tau in idx:
        if selected[i, j, tau]:
            per_stock[j] += 1
            pairs[(i, tau)] = pairs.get((i, tau), 0) + 1
    counts = np.array(list(per_stock.values())) if per_stock else np.zeros(1)
    n_selected = int(counts.sum())

    return FitDiagnostics(
        n_vars=n_vars,
        n_macro=n_macro,
        n_stocks=len(stock_idx),
        window=cfg.window,
        graph_shape=cfg.graph_shape,
        parent_selection=cfg.parent_selection,
        gate_mode=cfg.gate_mode,
        n_candidate_links=len(idx),
        n_bh_accepted=_count(bh_mask),
        n_stability_accepted=_count(stab_mask),
        n_jointly_accepted=_count(tier1),
        n_selected=n_selected,
        stocks_with_parent=int((counts > 0).sum()),
        p_deciles=[round(float(x), 5) for x in np.quantile(pv, deciles)] if len(pv) else [],
        stab_deciles=[round(float(x), 3) for x in np.quantile(sv, deciles)] if len(sv) else [],
        parents_per_stock=(int(counts.min()), float(np.median(counts)), int(counts.max())),
        distinct_parent_pairs=len(pairs),
        top_parent_share=(max(pairs.values()) / n_selected) if n_selected and pairs else 0.0,
        fit_seconds=fit_seconds,
        boot_seconds=boot_seconds,
    )


def log_diagnostics(d: FitDiagnostics, logger: logging.Logger | None = None) -> None:
    """Emit the funnel. Read top-down: whichever stage drops the most is binding."""
    lg = logger or log
    lg.info("--- causal fit diagnostics -------------------------------------")
    lg.info("  panel            : %d series (%d exogenous + %d stocks), window=%d bars",
            d.n_vars, d.n_macro, d.n_stocks, d.window)
    lg.info("  graph_shape      : %s   parent_selection: %s   gate_mode: %s",
            d.graph_shape, d.parent_selection, d.gate_mode)
    lg.info("  candidate links  : %d  (%.0f per stock target)",
            d.n_candidate_links, d.n_candidate_links / max(d.n_stocks, 1))
    lg.info("  gate-accepted (%s): %d", d.gate_mode, d.n_bh_accepted)
    lg.info("  stability-accepted: %d", d.n_stability_accepted)
    lg.info("  tier-1 (both)    : %d", d.n_jointly_accepted)
    lg.info("  selected (scored): %d   parents/stock min|med|max = %d|%.1f|%d",
            d.n_selected, *d.parents_per_stock)
    lg.info("  stocks w/ parent : %d / %d", d.stocks_with_parent, d.n_stocks)
    lg.info("  p-value deciles  : %s", d.p_deciles)
    lg.info("  stability deciles: %s", d.stab_deciles)
    lg.info("  distinct (src,lag) pairs: %d   modal-pair share: %.1f%%",
            d.distinct_parent_pairs, d.top_parent_share * 100)
    lg.info("  timing           : fit %.1fs, bootstrap %.1fs", d.fit_seconds, d.boot_seconds)
    if d.top_parent_share > 0.25:
        lg.warning("  DEGENERACY: %.0f%% of selected edges share one (source, lag). The book may be "
                   "a single factor bet dressed up as many names.", d.top_parent_share * 100)
    if d.n_stocks and d.stocks_with_parent < d.n_stocks:
        lg.warning("  %d stock(s) got no parent — expected 0 under parent_selection='top_k'.",
                   d.n_stocks - d.stocks_with_parent)
    lg.info("----------------------------------------------------------------")


def _new_pcmci(x: np.ndarray) -> PCMCI:
    n_vars = x.shape[1]
    df = pp.DataFrame(x, var_names=[str(k) for k in range(n_vars)])
    return PCMCI(dataframe=df, cond_ind_test=ParCorr(significance="analytic"), verbosity=0)


def bootstrap_stability(x: np.ndarray, cfg: SignalConfig, n_macro: int) -> np.ndarray:
    """Block-bootstrap directed-edge stability tensor (n_vars, n_vars, tau_max+1).

    Each entry [i, j, tau] is the fraction of block-resampled PCMCI fits in which
    the *directed* edge i -> j at lag tau appeared ('-->'). This is the honest
    stability score: unlike Tigramite's `link_frequency` (frequency of the most
    frequent outcome, which is high even for stable NON-edges), this measures how
    often the actual directed edge reappears. Block resampling preserves the
    weekly autocorrelation structure (block length ~ T^(1/3)).
    """
    pcmci = _new_pcmci(x)
    res = pcmci.run_bootstrap_of(
        method="run_pcmci",
        method_args=dict(
            tau_min=cfg.tau_min,
            tau_max=cfg.tau_max,
            pc_alpha=cfg.pc_alpha,
            link_assumptions=build_link_assumptions(cfg, n_macro, x.shape[1]),
        ),
        boot_samples=cfg.boot_samples,
        boot_blocklength="cube_root",
        seed=cfg.boot_seed,
    )
    graphs = res["boot_results"]["graph"]         # (B, n_vars, n_vars, tau_max+1), <U3
    return (graphs == "-->").mean(axis=0)          # directed-presence frequency


def fit_pcmci(window_array: np.ndarray, cfg: SignalConfig, n_macro: int):

    x = preprocess(window_array, cfg)
    n_vars = x.shape[1]
    stock_idx = list(range(n_macro, n_vars))

    t0 = time.time()
    pcmci = _new_pcmci(x)
    res = pcmci.run_pcmci(
        tau_min=cfg.tau_min,
        tau_max=cfg.tau_max,
        pc_alpha=cfg.pc_alpha,
        link_assumptions=build_link_assumptions(cfg, n_macro, n_vars),
    )
    p, val = res["p_matrix"], res["val_matrix"]
    fit_seconds = time.time() - t0

    # --- filter 1: the significance gate on the full-sample fit (see gate_mode) ---
    bh_mask = significance_mask(p, pcmci, cfg, n_macro, n_vars)

    # --- filter 2: bootstrap directed stability ---
    t1 = time.time()
    if cfg.use_bootstrap:
        stability = bootstrap_stability(x, cfg, n_macro)
    else:
        stability = np.ones_like(p, dtype=float)
    boot_seconds = time.time() - t1

    stab_mask = stability >= cfg.stability_floor
    tier1 = bh_mask & stab_mask

    if cfg.parent_selection == "bh_stability":
        selected = tier1
    elif cfg.parent_selection == "top_k":
        selected = select_parents_top_k(p, stability, cfg, n_macro, n_vars, stock_idx)
    else:
        raise ValueError(f"unknown parent_selection: {cfg.parent_selection!r}")

    # The gate reaches scores through two channels, and which ones are live depends
    # entirely on parent_selection. Say so out loud, because a null A/B result is
    # otherwise indistinguishable from a config that never let the gate act.
    if cfg.gate_mode == "none":
        if cfg.parent_selection == "top_k":
            log.warning(
                "gate_mode='none' with parent_selection='top_k': eligibility is "
                "unchanged (top-k by p-value), so the gate's only remaining channel "
                "is the tier-1 weight bonus, which now tiers on the stability floor "
                "alone.%s For a gate A/B that actually moves eligibility, use "
                "parent_selection='bh_stability'.",
                "" if cfg.use_bootstrap else
                " With use_bootstrap=False the stability mask is all-True, so every "
                "edge gets the bonus, it becomes a constant, and it cancels out of the "
                "cross-sectional ranking — expect scores identical to gate_mode='fdr' "
                "with tier_weight_bonus=0.",
            )
        elif cfg.parent_selection == "bh_stability":
            log.warning(
                "gate_mode='none' with parent_selection='bh_stability': every "
                "candidate link clearing the stability floor is now selected, with no "
                "significance filter at all. The book may be degenerate."
            )

    diagnostics = collect_diagnostics(
        p, stability, bh_mask, stab_mask, tier1, selected,
        cfg, n_macro, n_vars, fit_seconds, boot_seconds,
    )
    return val, p, selected, tier1, stability, diagnostics


def extract_edges(val, p, selected, tier1, stability, n_macro: int, n_vars: int) -> list[dict]:
    """-> list of {src,tgt,lag,mci,p,stab,strict} for selected, cross-variable edges.

    `strict` marks tier-1 edges (cleared BH-FDR and the stability floor). Selected
    but non-strict edges still score; they just carry less weight.
    """
    edges = []
    for i, j, tau in np.argwhere(selected):
        if i != j:
            edges.append(dict(
                src=int(i), tgt=int(j), lag=int(tau),
                mci=float(val[i, j, tau]), p=float(p[i, j, tau]),
                stab=float(stability[i, j, tau]),
                strict=bool(tier1[i, j, tau]),
            ))
    return edges


def tier_multiplier(e: dict, cfg: SignalConfig) -> float:
    """Weight bonus for tier-1 edges. This is what keeps the causal filter doing
    real work once it no longer gates eligibility: well-identified edges dominate
    the score and merely-selected ones only break ties."""
    return 1.0 + cfg.tier_weight_bonus if e.get("strict") else 1.0


def edge_weight(e: dict, cfg: SignalConfig) -> float:
    """Signed edge weight. Sign is always preserved; magnitude follows edge_weight.
    When weight_by_stability is on, scale by the edge's bootstrap stability so
    fragile survivors count for less."""
    if cfg.edge_weight == "mci":
        w = e["mci"]
    elif cfg.edge_weight == "sign":
        w = float(np.sign(e["mci"]))
    elif cfg.edge_weight == "mci_over_p":
        w = e["mci"] / max(e["p"], 1e-6)
    else:
        raise ValueError(cfg.edge_weight)
    if cfg.weight_by_stability:
        w *= e.get("stab", 1.0)
    return w * tier_multiplier(e, cfg)


def parents_by_stock(edges: list[dict], stock_idx: list[int]) -> dict[int, list[dict]]:
    pbs = {s: [] for s in stock_idx}
    for e in edges:
        if e["tgt"] in pbs and e["src"] != e["tgt"]:
            pbs[e["tgt"]].append(e)
    return pbs


def _lagged_corr(rets_arr: np.ndarray, src: int, tgt: int, lag: int, day: int, lookback: int) -> float:
    """Sample corr between src's return at t-lag and tgt's return at t, over the
    trailing `lookback` weeks ending strictly before `day`. This is the lead-lag
    statistic the gate trades on (lag >= 1, so no look-ahead)."""
    hi = day                      # tgt aligned to periods [.., day-1] (exclusive of day)
    lo = max(lag, hi - lookback)
    if hi - lo < 8:               # too few paired points to estimate
        return 0.0
    y = rets_arr[lo:hi, tgt]
    xsrc = rets_arr[lo - lag:hi - lag, src]
    if y.std() == 0 or xsrc.std() == 0:
        return 0.0
    return float(np.corrcoef(xsrc, y)[0, 1])


def score_on_day(rets_arr: np.ndarray, day: int, pbs: dict[int, list[dict]], cfg: SignalConfig) -> dict[int, float]:
    """Cross-sectional raw score for period `day` using only info < day (lags>=1).

    - "causal":     sum of (signed causal edge weight) * (parent lagged return).
    - "gated_corr": sum of (lead-lag correlation on the gated pair) * (parent
                    lagged return). The causal graph only decides WHICH pairs are
                    eligible; the lagged correlation sets the weight.
    - "blend":      handled by the caller (needs both score vectors to rank).
    """
    mode = cfg.score_mode
    scores: dict[int, float] = {}
    for s, elist in pbs.items():
        vals = []
        for e in elist:
            if day - e["lag"] < 0:
                continue
            parent_ret = rets_arr[day - e["lag"], e["src"]]
            if mode == "gated_corr":
                w = _lagged_corr(rets_arr, e["src"], e["tgt"], e["lag"], day, cfg.corr_lookback)
                if cfg.weight_by_stability:
                    w *= e.get("stab", 1.0)
                w *= tier_multiplier(e, cfg)
            else:  # "causal" (and the causal half of "blend")
                w = edge_weight(e, cfg)
            vals.append(w * parent_ret)
        if vals:
            scores[s] = float(np.sum(vals) if cfg.aggregate == "sum" else np.mean(vals))
        elif cfg.score_universe == "all":
            # Belt-and-braces: top_k + relax_if_empty should never leave a stock
            # parentless, but a neutral 0.0 keeps the universe whole regardless so
            # book size can never silently collapse the way it did before.
            scores[s] = 0.0
    return scores


def _rank_center(v: np.ndarray) -> np.ndarray:
    """Ranks mapped to [-0.5, 0.5]."""
    order = v.argsort().argsort().astype(float)
    return (order / max(len(v) - 1, 1)) - 0.5


def _cross_sectional_normalize(scores: dict[int, float], cfg: SignalConfig) -> dict[int, float]:
    """Rank- or robust-z-normalize scores cross-sectionally. Rank transforms are
    robust to the fat tails of MCI/correlation-weighted sums and reduce turnover-
    driven Sharpe instability vs raw scores."""
    if not scores or cfg.xs_normalize == "none":
        return scores

    # Rank-centering a handful of names collapses to a fixed ladder ({-0.5, 0, +0.5}
    # for n=3), destroying every bit of magnitude information. Fall back to a robust
    # z-score when the cross-section is too thin for ranks to carry signal.
    mode = cfg.xs_normalize
    if mode == "rank" and len(scores) < cfg.min_scored_for_rank:
        log.warning("Only %d scored names (< min_scored_for_rank=%d); using zscore "
                    "instead of rank normalization.", len(scores), cfg.min_scored_for_rank)
        mode = "zscore"

    keys = list(scores)
    v = np.array([scores[k] for k in keys], dtype=float)
    if mode == "rank":
        v = _rank_center(v)
    else:  # robust z-score
        med = np.median(v)
        mad = np.median(np.abs(v - med)) or 1.0
        v = (v - med) / (1.4826 * mad)
        if cfg.xs_winsor and cfg.xs_winsor > 0:
            v = np.clip(v, -cfg.xs_winsor, cfg.xs_winsor)
    return {k: float(x) for k, x in zip(keys, v)}


def _rank_map(scores: dict[int, float]) -> dict[int, float]:
    keys = list(scores)
    if not keys:
        return {}
    v = np.array([scores[k] for k in keys], dtype=float)
    r = _rank_center(v)
    return {k: float(x) for k, x in zip(keys, r)}


@dataclass
class SignalResult:
    """Everything one signal run produced, so callers need not refit or re-download.

    `rets` in particular is the same panel the fit used — agent1 reuses it for
    vol/beta sizing instead of pulling a second full yfinance download.
    """

    scores: dict[str, float]        # {ticker: normalized score}
    rets: pd.DataFrame              # weekly return panel (exogenous block first)
    names: list[str]                # all column names, ordered
    n_macro: int                    # size of the exogenous block
    stock_names: list[str]          # tradeable targets only
    edges: list[dict]               # selected edges, each tagged with `strict`
    diagnostics: FitDiagnostics


def run_signal(
    cfg: SignalConfig | None = None,
    tickers: list[str] | None = None,
    macro_tickers: list[str] | None = None,
) -> SignalResult:
    """Fit PCMCI on the trailing window and score every stock as of the most recent
    completed weekly bar.

    Under the default `parent_selection="top_k"` / `score_universe="all"` every
    stock target is scored, so `len(result.scores) == len(result.stock_names)`.
    Scores are already cross-sectionally normalized so downstream ranking/sizing
    is well-behaved.
    """
    cfg = cfg or SignalConfig()
    rets, names, n_macro = build_panel(cfg, tickers, macro_tickers)
    n_vars = len(names)
    stock_idx = list(range(n_macro, n_vars))

    window_arr = rets.values[-cfg.window:]
    if len(window_arr) < cfg.window:
        raise ValueError(f"Not enough weekly history: got {len(window_arr)} bars, need {cfg.window}. Increase lookback_years.")

    val, p, selected, tier1, stab, diagnostics = fit_pcmci(window_arr, cfg, n_macro)
    log_diagnostics(diagnostics)
    edges = extract_edges(val, p, selected, tier1, stab, n_macro, n_vars)
    pbs = parents_by_stock(edges, stock_idx)

    day = rets.shape[0] - 1

    if cfg.score_mode == "blend":
        causal_cfg = SignalConfig(**{**cfg.__dict__, "score_mode": "causal"})
        gate_cfg = SignalConfig(**{**cfg.__dict__, "score_mode": "gated_corr"})
        s_causal = score_on_day(rets.values, day, pbs, causal_cfg)
        s_gate = score_on_day(rets.values, day, pbs, gate_cfg)
        rc, rg = _rank_map(s_causal), _rank_map(s_gate)
        keys = set(rc) | set(rg)
        idx_scores = {k: cfg.blend_w * rc.get(k, 0.0) + (1 - cfg.blend_w) * rg.get(k, 0.0) for k in keys}
    else:
        idx_scores = score_on_day(rets.values, day, pbs, cfg)
        idx_scores = _cross_sectional_normalize(idx_scores, cfg)

    return SignalResult(
        scores={names[s]: sc for s, sc in idx_scores.items()},
        rets=rets,
        names=names,
        n_macro=n_macro,
        stock_names=[names[s] for s in stock_idx],
        edges=edges,
        diagnostics=diagnostics,
    )


def compute_causal_scores(
    cfg: SignalConfig | None = None,
    tickers: list[str] | None = None,
    macro_tickers: list[str] | None = None,
) -> dict[str, float]:
    """Scores-only wrapper around `run_signal`, kept for the notebook path."""
    return run_signal(cfg, tickers, macro_tickers).scores