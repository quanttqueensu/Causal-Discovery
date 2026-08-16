from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class PortfolioConfig:
    capital: float = 100_000.0
    quintile_fraction: float = 0.20   # fraction of the universe taken long and short
    inverse_vol: bool = True          # size within each book inversely to trailing vol
    beta_neutralize: bool = True      # split long/short notional to target net dollar-beta ~0
    ewma_span: int = 4                # weeks of EWMA smoothing applied to scores (0 disables)
    vol_lookback: int = 26            # trailing weeks used to estimate vol/beta for sizing
    beta_floor: float = 0.05          # skip beta-neutral split if either book's beta is below this
    market_ticker: str = "SPY"        # market proxy for beta estimation
    use_buffer: bool = True           # hysteresis band on quintile membership
    buffer_fraction: float = 0.33     # incumbents retained while inside this wider rank band

    # --- book-size guardrails ---
    # `quintile_fraction` used to be applied to however many names the signal
    # happened to score, so a sparse fit silently produced a 1-long/1-short book.
    # Sizing is now taken off the *universe* (see `universe_n` on construct_book)
    # and clamped, so a degenerate book is impossible rather than merely unlikely.
    target_book_names: int | None = None  # explicit names per side; overrides quintile_fraction
    min_book_names: int = 8               # floor per side (still capped by n_ranked // 2)
    max_book_names: int = 40              # ceiling per side, also caps the hysteresis band


def _ewma_smooth(scores: dict[str, float], prev_scores: dict[str, float], span: int) -> dict[str, float]:
    """Blend today's scores with last run's, damping week-to-week turnover.
    New tickers (absent from prev_scores) pass through unsmoothed."""
    if not prev_scores or span <= 0:
        return dict(scores)
    alpha = 2.0 / (span + 1.0)
    smoothed = {}
    for ticker, score in scores.items():
        prev = prev_scores.get(ticker)
        smoothed[ticker] = alpha * score + (1 - alpha) * prev if prev is not None else score
    return smoothed



def _entry_k(n_ranked: int, cfg: PortfolioConfig, universe_n: int | None = None) -> int:
    base = universe_n or n_ranked
    k = cfg.target_book_names or int(round(base * cfg.quintile_fraction))
    k = max(k, cfg.min_book_names)
    return max(1, min(k, cfg.max_book_names, n_ranked // 2))


def _select_quintiles(
    ranked: list[tuple[str, float]],
    cfg: PortfolioConfig,
    held_long: frozenset[str] = frozenset(),
    held_short: frozenset[str] = frozenset(),
    universe_n: int | None = None,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    n = len(ranked)
    entry_k = _entry_k(n, cfg, universe_n)
    longs = ranked[:entry_k]
    shorts = ranked[-entry_k:]
    if not cfg.use_buffer or not (held_long or held_short):
        return longs, shorts

    buffer_k = max(entry_k, min(int(round((universe_n or n) * cfg.buffer_fraction)),
                                n // 2, cfg.max_book_names))
    entering_long = {t for t, _ in longs}
    entering_short = {t for t, _ in shorts}

    # Retain incumbents still inside the wider band, preserving rank order.
    longs = [kv for kv in ranked[:buffer_k] if kv[0] in entering_long or kv[0] in held_long]
    shorts = [kv for kv in ranked[-buffer_k:] if kv[0] in entering_short or kv[0] in held_short]
    return longs, shorts


def _book_weights(tickers: list[str], returns: pd.DataFrame, cfg: PortfolioConfig) -> dict[str, float]:
    """Inverse-vol weights -  try to scale a positions contribution to its added volatility."""
    if cfg.inverse_vol and not returns.empty and all(t in returns.columns for t in tickers):
        vols = returns[tickers].tail(cfg.vol_lookback).std()
        fallback = vols[vols > 0].mean() if (vols > 0).any() else 1.0
        vols = vols.replace(0, np.nan).fillna(fallback)
        inv = 1.0 / vols
        w = inv / inv.sum()
        return {t: float(w[t]) for t in tickers}
    return {t: 1.0 / len(tickers) for t in tickers}


def _beta(returns: pd.DataFrame, ticker: str, market: str, lookback: int) -> float:
    window = returns[[ticker, market]].tail(lookback).dropna()
    if len(window) < 8:
        return 1.0
    var = window[market].var()
    if not var or np.isnan(var):
        return 1.0
    return float(window[ticker].cov(window[market]) / var)


def _split_notional(
    long_w: dict[str, float], short_w: dict[str, float], returns: pd.DataFrame, cfg: PortfolioConfig
) -> tuple[float, float]:
    """Long/short gross notional split. Beta-neutral (long_notional * beta_long ==
    short_notional * beta_short) when enabled and both books have a usable beta;
    50/50 dollar-neutral otherwise."""
    half = cfg.capital / 2.0
    if not cfg.beta_neutralize or returns.empty or cfg.market_ticker not in returns.columns:
        return half, half

    beta_long = sum(w * _beta(returns, t, cfg.market_ticker, cfg.vol_lookback) for t, w in long_w.items())
    beta_short = sum(w * _beta(returns, t, cfg.market_ticker, cfg.vol_lookback) for t, w in short_w.items())
    if beta_long <= cfg.beta_floor or beta_short <= cfg.beta_floor:
        return half, half

    short_notional = cfg.capital * beta_long / (beta_long + beta_short)
    return cfg.capital - short_notional, short_notional


def book_weights_from_scores(
    scores: dict[str, float],
    returns: pd.DataFrame,
    cfg: PortfolioConfig,
    universe_n: int | None = None,
    held_long: frozenset[str] = frozenset(),
    held_short: frozenset[str] = frozenset(),
) -> dict[str, float]:
    """Scores -> signed portfolio weights (longs positive, shorts negative).

    Split out of `construct_book` so the backtester exercises the *same* selection
    and sizing code that trades, rather than a parallel reimplementation that can
    drift from it. `construct_book` layers price lookup and share rounding on top.

    Weights are fractions of `cfg.capital`, so sum(|w|) == 1 when both books are
    non-empty. Returns {} when there are too few names to form a two-sided book.
    """
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) < 2:
        return {}

    longs, shorts = _select_quintiles(ranked, cfg, held_long, held_short, universe_n)
    long_w = _book_weights([t for t, _ in longs], returns, cfg)
    short_w = _book_weights([t for t, _ in shorts], returns, cfg)
    long_notional, short_notional = _split_notional(long_w, short_w, returns, cfg)

    weights: dict[str, float] = {}
    for w_map, notional, sign in ((long_w, long_notional, 1.0), (short_w, short_notional, -1.0)):
        for ticker, w in w_map.items():
            weights[ticker] = sign * w * notional / cfg.capital
    return weights


def construct_book(
    scores: dict[str, float],
    prices: dict[str, float],
    returns: pd.DataFrame,
    cfg: PortfolioConfig,
    prev_scores: dict[str, float] | None = None,
    current_sides: dict[str, str] | None = None,
    universe_n: int | None = None,
) -> tuple[list[dict], dict[str, float]]:
    """Build the dollar-neutral (or beta-neutral) long/short book.

    `current_sides` maps ticker -> "long"/"short" for positions currently held,
    and feeds the hysteresis band in `_select_quintiles`.

    `universe_n` is the number of names the signal scored before the price filter;
    it is the denominator for `quintile_fraction`. Leave it None only if you
    genuinely want the book sized off the surviving ranked list.

    Returns (positions, smoothed_scores):
      - positions: one {ticker, side, score, price, shares, notional} dict per
        position that survived rounding to a whole share.
      - smoothed_scores: the EWMA state that produced this book, covering every
        scored ticker (not just the ones that made the book). Persist this, not
        the raw scores, or the recursion never accumulates memory.
    """
    smoothed = _ewma_smooth(scores, prev_scores or {}, cfg.ewma_span)
    ranked = sorted(((t, s) for t, s in smoothed.items() if t in prices), key=lambda kv: kv[1], reverse=True)
    if len(ranked) < 2:
        log.error("Only %d scored+priced name(s) — cannot form a two-sided book.", len(ranked))
        return [], smoothed

    sides = current_sides or {}
    held_long = frozenset(t for t, side in sides.items() if side == "long")
    held_short = frozenset(t for t, side in sides.items() if side == "short")

    longs, shorts = _select_quintiles(ranked, cfg, held_long, held_short, universe_n)
    long_w = _book_weights([t for t, _ in longs], returns, cfg)
    short_w = _book_weights([t for t, _ in shorts], returns, cfg)
    long_notional, short_notional = _split_notional(long_w, short_w, returns, cfg)

    positions = []
    for book, weights, notional_total, side in (
        (longs, long_w, long_notional, "long"),
        (shorts, short_w, short_notional, "short"),
    ):
        for ticker, score in book:
            price = prices[ticker]
            shares = int(round(notional_total * weights[ticker] / price))
            if shares <= 0:
                continue
            positions.append(dict(
                ticker=ticker, side=side, score=float(score),
                price=float(price), shares=shares, notional=shares * price,
            ))

    # A silently degenerate book is what let the 1-long/1-short run reach production.
    n_long = sum(1 for p in positions if p["side"] == "long")
    n_short = sum(1 for p in positions if p["side"] == "short")
    if min(n_long, n_short) < cfg.min_book_names:
        log.warning(
            "DEGENERATE BOOK: %d longs / %d shorts, below min_book_names=%d "
            "(ranked=%d, universe_n=%s). The signal is scoring too few names or "
            "prices are missing — check the causal fit diagnostics above.",
            n_long, n_short, cfg.min_book_names, len(ranked), universe_n,
        )
    return positions, smoothed