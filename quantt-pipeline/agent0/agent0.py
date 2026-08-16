"""Agent 0 — null benchmark paper-trading strategy.

Agent 0 runs a dollar-neutral long/short equity portfolio driven by *purely
random* composite scores. There is no real signal here, intentionally. Its only
job is to exercise the full trading pipeline end-to-end — universe loading, price
fetching, scoring, portfolio construction, and live order submission against the
Alpaca paper-trading API — so the plumbing is proven before a real signal (the
PCMCI-based causal composite score) is swapped in.

Run it like a monthly rebalance: it cancels open orders, flattens existing
positions, then opens a fresh top-quintile-long / bottom-quintile-short book.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()  # pull APCA_* credentials from a local .env file

APCA_API_KEY_ID = os.getenv("APCA_API_KEY_ID")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
APCA_API_BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

CAPITAL = 100000          # total paper capital in USD
QUINTILE_FRACTION = 0.20   # fraction of the universe taken long and short
RANDOM_SEED = None         # set to an int for reproducible runs while debugging
ORDER_BATCH_DELAY = 1.0    # seconds to pause between order batches (rate limits)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("agent0")


# ---------------------------------------------------------------------------
# Universe — ~90 large-cap US names, roughly 10 per GICS sector
# ---------------------------------------------------------------------------

UNIVERSE = {
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "WMB"],
    "Financials": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "SPGI"],
    "Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "CSCO", "ACN", "AMD"],
    "Healthcare": ["UNH", "JNJ", "LLY", "MRK", "ABBV", "PFE", "TMO", "ABT", "DHR", "AMGN"],
    "Industrials": ["CAT", "HON", "UNP", "GE", "BA", "RTX", "DE", "LMT", "UPS", "ETN"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "BKNG", "TJX", "GM"],
    "Consumer Staples": ["PG", "KO", "PEP", "COST", "WMT", "MDLZ", "CL", "MO", "PM", "KMB"],
    "Materials": ["LIN", "SHW", "APD", "ECL", "FCX", "NEM", "DOW", "DD", "NUE", "PPG"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ED", "PEG"],
    "Real Estate": ["PLD", "AMT", "EQIX", "PSA", "O", "SPG", "WELL", "CCI", "DLR", "VICI"],
}


@dataclass
class Position:
    """A single intended position in the rebalanced book."""

    ticker: str
    side: str       # "long" or "short"
    score: float
    price: float
    shares: int
    notional: float


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def get_universe() -> list[str]:
    """Return the flat list of tickers across all sectors."""
    tickers = [t for sector in UNIVERSE.values() for t in sector]
    log.info("Universe loaded: %d tickers across %d sectors", len(tickers), len(UNIVERSE))
    return tickers


def get_prices(tickers: list[str]) -> dict[str, float]:
    """Fetch the latest price for each ticker via yfinance.

    Confirms each name is active and tradeable. Any ticker that fails to return a
    usable price is logged and dropped from the run.
    """
    log.info("Fetching prices for %d tickers via yfinance...", len(tickers))
    prices: dict[str, float] = {}

    # Bulk download recent closes; this is faster and gentler than per-ticker calls.
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
            # With multiple tickers, columns are a (ticker, field) MultiIndex.
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


def generate_random_scores(tickers: list[str], seed: int | None = RANDOM_SEED) -> dict[str, float]:
    """Assign each ticker a random composite score from Uniform(-1, 1).

    This is the ENTIRE signal, on purpose. Agent 0 is a null benchmark: the scores
    carry no information whatsoever. In the real strategy this function is replaced
    by PCMCI-based causal composite scores. It is meant to be embarrassingly simple.
    """
    rng = np.random.default_rng(seed)
    scores = rng.uniform(-1.0, 1.0, size=len(tickers))
    return dict(zip(tickers, scores))


def construct_portfolio(
    scores: dict[str, float],
    prices: dict[str, float],
    capital: float = CAPITAL,
    quintile_fraction: float = QUINTILE_FRACTION,
) -> list[Position]:
    """Build a dollar-neutral long/short book from scores and prices.

    Top quintile by score goes long, bottom quintile short, the middle gets no
    position. Each book gets half the capital, equally weighted within the book.
    Shares are rounded to whole numbers using the current price.
    """
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    n = len(ranked)
    k = max(1, int(round(n * quintile_fraction)))

    longs = ranked[:k]
    shorts = ranked[-k:]

    book_notional = capital / 2.0  # $50k long, $50k short for $100k total
    long_per_name = book_notional / len(longs)
    short_per_name = book_notional / len(shorts)

    positions: list[Position] = []
    for ticker, score in longs:
        price = prices[ticker]
        shares = int(round(long_per_name / price))
        if shares <= 0:
            log.warning("Skipping long %s: price %.2f too high for target notional", ticker, price)
            continue
        positions.append(Position(ticker, "long", score, price, shares, shares * price))

    for ticker, score in shorts:
        price = prices[ticker]
        shares = int(round(short_per_name / price))
        if shares <= 0:
            log.warning("Skipping short %s: price %.2f too high for target notional", ticker, price)
            continue
        positions.append(Position(ticker, "short", score, price, shares, shares * price))

    log.info(
        "Constructed portfolio: %d longs, %d shorts (quintile size k=%d of %d)",
        sum(p.side == "long" for p in positions),
        sum(p.side == "short" for p in positions),
        k,
        n,
    )
    return positions


def submit_orders(positions: list[Position]) -> tuple[list[Position], list[tuple[str, str]]]:
    """Submit paper market orders to Alpaca for the constructed book.

    Cancels open orders, flattens existing positions (full monthly replacement),
    then submits market orders for every long and short. Returns the list of
    successfully submitted positions and a list of (ticker, reason) failures.

    Connection/credential problems raise so the caller can report them clearly;
    individual order failures are caught and reported without aborting the run.
    """
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    if not APCA_API_KEY_ID or not APCA_API_SECRET_KEY:
        raise RuntimeError(
            "Missing Alpaca credentials. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY "
            "in your .env file (see .env.example)."
        )

    paper = "paper" in (APCA_API_BASE_URL or "")
    client = TradingClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY, paper=paper)

    # Confirm credentials/connection up front so a bad key fails loudly here.
    account = client.get_account()
    log.info("Connected to Alpaca (paper=%s). Account status: %s", paper, account.status)

    log.info("Cancelling all open orders...")
    client.cancel_orders()
    time.sleep(ORDER_BATCH_DELAY)

    log.info("Closing all existing positions (full rebalance)...")
    client.close_all_positions(cancel_orders=True)
    time.sleep(ORDER_BATCH_DELAY * 2)  # give fills time to settle before re-entry

    submitted: list[Position] = []
    failed: list[tuple[str, str]] = []

    longs = [p for p in positions if p.side == "long"]
    shorts = [p for p in positions if p.side == "short"]

    for batch_name, batch, side in (
        ("longs", longs, OrderSide.BUY),
        ("shorts", shorts, OrderSide.SELL),
    ):
        log.info("Submitting %d %s orders...", len(batch), batch_name)
        for pos in batch:
            try:
                order = MarketOrderRequest(
                    symbol=pos.ticker,
                    qty=pos.shares,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )
                client.submit_order(order)
                submitted.append(pos)
            except Exception as exc:  # noqa: BLE001 - log and keep going
                reason = str(exc)
                log.error("Order failed for %s (%s): %s", pos.ticker, pos.side, reason)
                failed.append((pos.ticker, reason))
        time.sleep(ORDER_BATCH_DELAY)

    return submitted, failed


def print_summary(
    positions: list[Position],
    submitted: list[Position],
    failed: list[tuple[str, str]],
) -> None:
    """Print a clean human-readable summary of the rebalance."""
    longs = [p for p in submitted if p.side == "long"]
    shorts = [p for p in submitted if p.side == "short"]

    long_notional = sum(p.notional for p in longs)
    short_notional = sum(p.notional for p in shorts)
    net = long_notional - short_notional

    print("\n" + "=" * 64)
    print("AGENT 0 — NULL BENCHMARK REBALANCE SUMMARY")
    print("=" * 64)
    print(f"Long positions : {len(longs)}")
    print(f"Short positions: {len(shorts)}")
    print(f"Long notional  : ${long_notional:>12,.2f}")
    print(f"Short notional : ${short_notional:>12,.2f}")
    print(f"Net exposure   : ${net:>12,.2f}   (target ~$0)")

    def _table(title: str, book: list[Position]) -> None:
        print(f"\n{title}")
        print(f"  {'Ticker':<8}{'Shares':>8}{'Price':>12}{'Notional':>14}")
        print("  " + "-" * 42)
        for p in sorted(book, key=lambda x: x.notional, reverse=True):
            print(f"  {p.ticker:<8}{p.shares:>8}{p.price:>12,.2f}{p.notional:>14,.2f}")

    if longs:
        _table("LONGS", longs)
    if shorts:
        _table("SHORTS", shorts)

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

def main() -> None:
    """Run the full Agent 0 flow: universe -> prices -> scores -> book -> orders."""
    log.info("Starting Agent 0 (null benchmark, random scores).")

    tickers = get_universe()
    prices = get_prices(tickers)
    if not prices:
        log.error("No tradeable tickers after price fetch. Aborting.")
        return

    tradeable = list(prices.keys())
    scores = generate_random_scores(tradeable)
    positions = construct_portfolio(scores, prices)

    submitted: list[Position] = []
    failed: list[tuple[str, str]] = []
    try:
        submitted, failed = submit_orders(positions)
    except Exception as exc:  # noqa: BLE001 - credential/connection problems
        log.error("Order submission aborted: %s", exc)
        print(
            "\nCould not submit orders. Check that your Alpaca paper-trading "
            "credentials in .env are valid and the API is reachable.\n"
        )
        # Still show the intended book so the dry-run output is useful.
        print_summary(positions, [], [])
        return

    print_summary(positions, submitted, failed)
    log.info("Agent 0 run complete.")


if __name__ == "__main__":
    main()
