from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from causal_signal import (
    MACRO_TICKERS,
    STOCK_TICKERS,
    SignalConfig,
    SignalResult,
    build_panel,
    compute_causal_scores,
    run_signal,
)
from portfolio import PortfolioConfig, construct_book

# ~/.pyenv/versions/3.12.4/bin/python3 agent1.py [--dry-run]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()  # pull APCA_* credentials from a local .env file

APCA_API_KEY_ID = os.getenv("APCA_API_KEY_ID")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
APCA_API_BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

CAPITAL = 100000           # total paper capital in USD
GATE_MODE = os.getenv("GATE_MODE", "fdr")  # "fdr" | "raw" | "none" — significance-gate A/B switch
QUINTILE_FRACTION = 0.20   # fraction of the (causally-scored) universe taken long and short
ORDER_BATCH_DELAY = 1.0    # seconds to pause between order batches (rate limits)
MIN_TRADE_NOTIONAL = 250.0  # skip share-count changes worth less than this (~0.25% of capital)
FLIP_FLAT_TIMEOUT = 15.0   # seconds to wait for a flipping ticker to go flat before re-opening

# Pre-gate_mode EWMA state: a single file, written by what is now gate_mode="fdr".
LEGACY_PREV_SCORES_PATH = Path(".prev_scores.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("agent1")


@dataclass
class Position:
    """A single intended position in the rebalanced book."""

    ticker: str
    side: str       # "long" or "short"
    score: float
    price: float
    shares: int
    notional: float


@dataclass
class Trade:
    """A single delta order in the rebalance plan."""

    ticker: str
    side: str       # "buy" or "sell"
    shares: int     # always positive
    stage: int      # 1 = close/reduce (frees buying power), 2 = open/increase
    reason: str     # entry | exit | increase | reduce | flip-close | flip-open


# ---------------------------------------------------------------------------
# Persistence for EWMA smoothing
# ---------------------------------------------------------------------------

def prev_scores_path(gate_mode: str) -> Path:
    """EWMA state file for one gate_mode.

    Keyed by mode because the state is recursive: running --gate-mode fdr and then
    --gate-mode none against a shared file would blend the two configs' scores
    through the smoother, contaminating exactly the comparison the switch exists to
    make. $PREV_SCORES_PATH still overrides, as an explicit single-file escape
    hatch for callers that want the old behaviour.
    """
    override = os.getenv("PREV_SCORES_PATH")
    return Path(override) if override else Path(f".prev_scores.{gate_mode}.json")


def _migrate_legacy_scores() -> None:
    """One-shot copy of the pre-split state file into the "fdr" slot.

    The single `.prev_scores.json` was produced by BH-FDR, so that is the mode
    whose state it is. Without this, the first post-change production run would
    start its EWMA cold.
    """
    if os.getenv("PREV_SCORES_PATH"):
        return
    target = prev_scores_path("fdr")
    if not LEGACY_PREV_SCORES_PATH.exists() or target.exists():
        return
    try:
        shutil.copyfile(LEGACY_PREV_SCORES_PATH, target)
        log.info("Migrated EWMA state %s -> %s (legacy file left in place).",
                 LEGACY_PREV_SCORES_PATH, target)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not migrate legacy EWMA state (%s); EWMA may start cold.", exc)


def load_prev_scores(gate_mode: str) -> dict[str, float]:
    _migrate_legacy_scores()
    try:
        return json.loads(prev_scores_path(gate_mode).read_text())
    except Exception:
        return {}


def save_scores(scores: dict[str, float], gate_mode: str) -> None:
    """Persist the *smoothed* scores that built this week's book, so next run's
    EWMA blends against accumulated state rather than a single raw observation.

    Written via a temp file and rename so a crash mid-write can't leave corrupt
    JSON — this file is now genuine recursive state, not a one-week convenience.
    """
    path = prev_scores_path(gate_mode)
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(scores))
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not persist scores for EWMA (%s)", exc)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def get_signal(cfg: SignalConfig) -> SignalResult:
    """Fit PCMCI and return the full signal result.

    Returns the return panel alongside the scores, so sizing reuses the panel the
    fit already downloaded instead of pulling a second copy of every ticker.

    The denominator in the completion log is the number of stocks that actually
    survived the panel build, not `len(STOCK_TICKERS)` — a ticker yfinance failed
    to return was never a candidate, and counting it flatters the ratio.
    """
    log.info("Fitting PCMCI causal composite (mode=%s, selection=%s, gate=%s, bootstrap=%s) "
             "on %d candidate stocks...",
             cfg.score_mode, cfg.parent_selection, cfg.gate_mode, cfg.use_bootstrap,
             len(STOCK_TICKERS))
    t0 = time.time()
    result = run_signal(cfg)
    log.info("Causal fit done in %.1fs — %d/%d stocks scored",
             time.time() - t0, len(result.scores), len(result.stock_names))
    return result


def get_prices(tickers: list[str]) -> dict[str, float]:
    """Fetch the latest price for each ticker via yfinance.

    Confirms each name is active and tradeable. Any ticker that fails to return a
    usable price is logged and dropped from the run.
    """
    log.info("Fetching prices for %d tickers via yfinance...", len(tickers))
    prices: dict[str, float] = {}

    data = yf.download(
        tickers,
        period="5d",
        interval="1d",
        progress=False,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
    )

    for ticker in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                close = data[ticker]["Close"].dropna()
            else:  # single-ticker fallback
                close = data["Close"].dropna()

            if close.empty:
                raise ValueError("no close data returned")

            price = float(close.iloc[-1])
            if not np.isfinite(price) or price <= 0:
                raise ValueError(f"invalid price {price!r}")

            prices[ticker] = price
        except Exception as exc:  # noqa: BLE001 - drop and continue on any failure
            log.warning("Dropping %s: failed to fetch price (%s)", ticker, exc)

    log.info("Got valid prices for %d/%d tickers", len(prices), len(tickers))
    return prices


def construct_portfolio(
    scores: dict[str, float],
    prices: dict[str, float],
    returns: pd.DataFrame,
    prev_scores: dict[str, float],
    pcfg: PortfolioConfig,
    current_sides: dict[str, str],
    universe_n: int | None = None,
) -> tuple[list[Position], dict[str, float]]:
    """Build the long/short book via portfolio.construct_book, adapt the dict
    positions to the Position dataclass the executor expects, and hand back the
    smoothed scores so the caller can persist the EWMA state.

    `universe_n` is the scored-universe size that `quintile_fraction` applies to,
    passed through so book size tracks the universe rather than however many
    names survived the price fetch.
    """
    raw, smoothed = construct_book(
        scores, prices, returns, pcfg, prev_scores=prev_scores,
        current_sides=current_sides, universe_n=universe_n,
    )
    positions = [
        Position(p["ticker"], p["side"], p["score"], p["price"], p["shares"], p["notional"])
        for p in raw
    ]
    log.info(
        "Constructed portfolio: %d longs, %d shorts (inverse_vol=%s, beta_neutral=%s, "
        "ewma_span=%d, buffer=%s @ %.0f%%)",
        sum(p.side == "long" for p in positions),
        sum(p.side == "short" for p in positions),
        pcfg.inverse_vol, pcfg.beta_neutralize, pcfg.ewma_span,
        pcfg.use_buffer, pcfg.buffer_fraction * 100,
    )
    return positions, smoothed


# ---------------------------------------------------------------------------
# Broker state and diff-based execution
# ---------------------------------------------------------------------------

def _trading_client():
    """Authenticated Alpaca trading client. Raises on missing credentials so a
    bad config fails loudly instead of silently trading nothing."""
    from alpaca.trading.client import TradingClient

    if not APCA_API_KEY_ID or not APCA_API_SECRET_KEY:
        raise RuntimeError(
            "Missing Alpaca credentials. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY "
            "in your .env file (see .env.example)."
        )
    paper = "paper" in (APCA_API_BASE_URL or "")
    return TradingClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY, paper=paper)


def get_current_positions(client) -> dict[str, int]:
    """Live signed share counts from Alpaca: positive long, negative short.

    This is the source of truth for the diff — never last week's targets, which
    may not have filled.
    """
    current: dict[str, int] = {}
    for pos in client.get_all_positions():
        qty = int(float(pos.qty))
        if qty:
            current[pos.symbol] = qty
    log.info("Live book: %d open positions (%d long, %d short)",
             len(current), sum(q > 0 for q in current.values()), sum(q < 0 for q in current.values()))
    return current


def _wait_until_flat(client, tickers: set[str], timeout: float = FLIP_FLAT_TIMEOUT) -> None:
    """Poll until the given tickers hold no position, or give up after `timeout`.

    Alpaca rejects an order that would flip a position's side while the old side
    is still open, so a flip has to be sequenced close-then-open with the close
    confirmed first.
    """
    if not tickers:
        return
    still_open = set(tickers)
    deadline = time.time() + timeout
    while time.time() < deadline:
        still_open = {p.symbol for p in client.get_all_positions()} & tickers
        if not still_open:
            return
        time.sleep(1.0)
    log.warning("Timed out waiting for %s to flatten; flip orders may be rejected.",
                ", ".join(sorted(still_open)))


def compute_order_plan(
    positions: list[Position],
    current: dict[str, int],
    prices: dict[str, float],
) -> list[Trade]:
    """Diff the target book against live positions and return only the deltas.

    Signed share convention: positive = long, negative = short. Stage 1 holds
    every trade that closes or reduces an existing position; stage 2 holds every
    trade that opens or increases one. Running stage 1 first frees buying power
    and flattens any ticker whose side is flipping.

    Share-count changes worth less than MIN_TRADE_NOTIONAL are skipped as
    rounding noise. Full exits are never skipped.
    """
    targets = {p.ticker: (p.shares if p.side == "long" else -p.shares) for p in positions}
    trades: list[Trade] = []

    for ticker in sorted(set(targets) | set(current)):
        tgt, cur = targets.get(ticker, 0), current.get(ticker, 0)
        if tgt == cur:
            continue

        # Side flip: not expressible as one order — close to flat, then re-open.
        if cur and tgt and (cur > 0) != (tgt > 0):
            trades.append(Trade(ticker, "sell" if cur > 0 else "buy", abs(cur), 1, "flip-close"))
            trades.append(Trade(ticker, "buy" if tgt > 0 else "sell", abs(tgt), 2, "flip-open"))
            continue

        delta = tgt - cur
        price = prices.get(ticker, 0.0)
        if tgt and price > 0 and abs(delta) * price < MIN_TRADE_NOTIONAL:
            log.debug("Skipping %s: %+d shares (~$%.0f) below the $%.0f minimum trade.",
                      ticker, delta, abs(delta) * price, MIN_TRADE_NOTIONAL)
            continue

        reason = "entry" if not cur else "exit" if not tgt else (
            "increase" if abs(tgt) > abs(cur) else "reduce")
        stage = 1 if abs(tgt) < abs(cur) else 2
        trades.append(Trade(ticker, "buy" if delta > 0 else "sell", abs(delta), stage, reason))

    log.info("Order plan: %d trades (%d closes/reductions, %d opens/increases) vs. "
             "%d positions held", len(trades), sum(t.stage == 1 for t in trades),
             sum(t.stage == 2 for t in trades), len(current))
    return trades


def submit_orders(trades: list[Trade], client) -> tuple[list[Trade], list[tuple[str, str]]]:
    """Submit the delta orders to Alpaca in two stages.

    Cancels resting orders, then sends stage 1 (closes and reductions) so buying
    power is freed and any flipping ticker is flat before stage 2 re-opens it on
    the other side. Nothing is liquidated that the target book still wants.

    Connection problems raise so the caller can report them clearly; individual
    order failures are logged and skipped without aborting the run.
    """
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    account = client.get_account()
    log.info("Connected to Alpaca. Account status: %s", account.status)

    log.info("Cancelling all open orders...")
    client.cancel_orders()
    time.sleep(ORDER_BATCH_DELAY)

    submitted: list[Trade] = []
    failed: list[tuple[str, str]] = []
    order_sides = {"buy": OrderSide.BUY, "sell": OrderSide.SELL}

    for stage in (1, 2):
        batch = [t for t in trades if t.stage == stage]
        if not batch:
            continue
        log.info("Submitting %d stage-%d orders (%s)...", len(batch), stage,
                 "closes/reductions" if stage == 1 else "opens/increases")
        for trade in batch:
            try:
                client.submit_order(MarketOrderRequest(
                    symbol=trade.ticker,
                    qty=trade.shares,
                    side=order_sides[trade.side],
                    time_in_force=TimeInForce.DAY,
                ))
                submitted.append(trade)
            except Exception as exc:  # noqa: BLE001 - log and keep going
                reason = str(exc)
                log.error("Order failed for %s (%s %d, %s): %s",
                          trade.ticker, trade.side, trade.shares, trade.reason, reason)
                failed.append((trade.ticker, reason))

        if stage == 1:
            time.sleep(ORDER_BATCH_DELAY * 2)
            _wait_until_flat(client, {t.ticker for t in trades if t.reason == "flip-open"})

    return submitted, failed


def print_summary(
    positions: list[Position],
    submitted: list[Trade],
    failed: list[tuple[str, str]],
    current: dict[str, int],
    prices: dict[str, float],
) -> None:
    """Print a clean human-readable summary of the target book and the delta
    orders actually sent to reach it."""
    longs = [p for p in positions if p.side == "long"]
    shorts = [p for p in positions if p.side == "short"]
    long_notional = sum(p.notional for p in longs)
    short_notional = sum(p.notional for p in shorts)
    gross = long_notional + short_notional
    traded = sum(t.shares * prices.get(t.ticker, 0.0) for t in submitted)

    print("\n" + "=" * 64)
    print("AGENT 1 — CAUSAL COMPOSITE REBALANCE SUMMARY (Alpaca paper, diff)")
    print("=" * 64)
    print(f"Held before    : {len(current)} positions")
    print(f"Long positions : {len(longs)}")
    print(f"Short positions: {len(shorts)}")
    print(f"Long notional  : ${long_notional:>12,.2f}")
    print(f"Short notional : ${short_notional:>12,.2f}")
    print(f"Net exposure   : ${long_notional - short_notional:>12,.2f}   (target ~$0)")
    print(f"Delta orders   : {len(submitted)} submitted, {len(failed)} failed")
    print(f"Traded notional: ${traded:>12,.2f}   ({traded / gross:.0%} of ${gross:,.0f} gross)"
          if gross else f"Traded notional: ${traded:>12,.2f}")

    def _table(title: str, book: list[Position]) -> None:
        print(f"\n{title}")
        print(f"  {'Ticker':<8}{'Score':>10}{'Shares':>8}{'Price':>12}{'Notional':>14}")
        print("  " + "-" * 52)
        for p in sorted(book, key=lambda x: x.notional, reverse=True):
            print(f"  {p.ticker:<8}{p.score:>10.3f}{p.shares:>8}{p.price:>12,.2f}{p.notional:>14,.2f}")

    if longs:
        _table("LONGS (target)", longs)
    if shorts:
        _table("SHORTS (target)", shorts)

    if submitted:
        print(f"\nORDERS SENT ({len(submitted)})")
        print(f"  {'Ticker':<8}{'Side':>6}{'Shares':>8}{'Stage':>7}  {'Reason'}")
        print("  " + "-" * 44)
        for t in sorted(submitted, key=lambda x: (x.stage, x.ticker)):
            print(f"  {t.ticker:<8}{t.side:>6}{t.shares:>8}{t.stage:>7}  {t.reason}")

    if failed:
        print(f"\nFAILED ORDERS ({len(failed)})")
        print("  " + "-" * 42)
        for ticker, reason in failed:
            print(f"  {ticker:<8} {reason}")
    else:
        print("\nNo failed orders.")
    print("=" * 64 + "\n")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Run the full Agent 1 flow: causal scores -> live positions -> book ->
    delta orders -> persist smoothed EWMA state."""
    parser = argparse.ArgumentParser(description="Agent 1 — causal composite long/short rebalance.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fit the signal and print the intended book without contacting Alpaca, "
             "submitting orders, or updating the persisted EWMA state.",
    )
    parser.add_argument(
        "--gate-mode", choices=["fdr", "raw", "none"], default=None,
        help="Override the significance gate: fdr (BH-FDR), raw (uncorrected "
             "p<=alpha), none (bypass). Default from $GATE_MODE, else 'fdr'. "
             "EWMA state is kept per mode, so A/B runs do not contaminate each other.",
    )
    args = parser.parse_args(argv)

    scfg = SignalConfig(gate_mode=(args.gate_mode or GATE_MODE))
    gate_mode = scfg.gate_mode   # post-validation/back-compat normalization
    pcfg = PortfolioConfig(capital=CAPITAL, quintile_fraction=QUINTILE_FRACTION)

    log.info("Starting Agent 1 (causal composite, gate_mode=%s, EWMA state=%s, %s).",
             gate_mode, prev_scores_path(gate_mode),
             "DRY RUN — no orders" if args.dry_run else "Alpaca paper")

    result = get_signal(scfg)
    scores = result.scores
    if not scores:
        log.error("No stocks received a causal score this refit. Aborting.")
        return

    # Panel from the signal run — no second download.
    returns = result.rets

    prev_scores = load_prev_scores(gate_mode)
    if not prev_scores:
        log.warning("No previous score state found at %s — EWMA starts cold this run.",
                    prev_scores_path(gate_mode))

    prices = get_prices(list(scores.keys()))
    tradeable_scores = {t: s for t, s in scores.items() if t in prices}
    if not tradeable_scores:
        log.error("No tradeable tickers after price fetch. Aborting.")
        return

    # Live positions are read BEFORE the book is built: the hysteresis band
    # retains incumbents, so selection depends on what is already held.
    client = None
    if args.dry_run:
        # No broker state to diff against, so the book is built as if starting flat.
        current: dict[str, float] = {}
    else:
        try:
            client = _trading_client()
            current = get_current_positions(client)
        except Exception as exc:  # noqa: BLE001
            log.error("Could not read live positions from Alpaca (%s). Aborting — a diff "
                      "rebalance without a reliable position snapshot is not safe.", exc)
            print(
                "\nCould not reach Alpaca. Check that your paper-trading credentials "
                "in .env are valid and the API is reachable.\n"
            )
            return
    current_sides = {t: ("long" if q > 0 else "short") for t, q in current.items()}

    positions, smoothed = construct_portfolio(
        tradeable_scores, prices, returns, prev_scores, pcfg, current_sides,
        universe_n=len(scores),
    )

    if args.dry_run:
        print_summary(positions, [], [], current, prices)
        log.info("Dry run complete — no orders submitted, EWMA state not persisted.")
        return

    trades = compute_order_plan(positions, current, prices)

    if not trades:
        log.info("Target book already matches the live book — nothing to trade.")
        print_summary(positions, [], [], current, prices)
        save_scores(smoothed, gate_mode)
        return

    submitted: list[Trade] = []
    failed: list[tuple[str, str]] = []
    try:
        submitted, failed = submit_orders(trades, client)
    except Exception as exc:  # noqa: BLE001 - connection problems
        log.error("Order submission aborted: %s", exc)
        print(
            "\nCould not submit orders. Check that your Alpaca paper-trading "
            "credentials in .env are valid and the API is reachable.\n"
        )
        print_summary(positions, [], [], current, prices)
        save_scores(smoothed, gate_mode)  # still persist so next EWMA has a reference
        return

    print_summary(positions, submitted, failed, current, prices)
    save_scores(smoothed, gate_mode)  # update EWMA reference for the next rebalance
    log.info("Agent 1 run complete.")


if __name__ == "__main__":
    main()