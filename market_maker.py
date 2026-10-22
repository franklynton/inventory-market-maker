"""Core engine for an inventory-contingent market maker.

Quotes two-way prices on a handful of equities in response to simulated hedge
fund requests-for-quote (RFQs). Quotes are built around a reference price and
skewed asymmetrically by inventory tier: as a position builds, the side that
would add to it is widened and the side that would reduce it is shaded in,
which discourages further accumulation and attracts offsetting flow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TICKS_PER_HOUR = 720                  # one tick every 5 seconds
TICKS_PER_DAY = int(6.5 * TICKS_PER_HOUR)
TRADING_DAYS = 252


@dataclass(frozen=True)
class Asset:
    ticker: str
    start_price: float
    annual_vol: float
    base_half_spread_bps: float
    position_limit: int               # hard limit on |position|, in shares


ASSETS = (
    Asset("AAPL", 225.0, 0.25, 2.0, 50_000),
    Asset("MSFT", 430.0, 0.22, 1.8, 30_000),
    Asset("JPM", 210.0, 0.24, 2.5, 50_000),
    Asset("XOM", 115.0, 0.26, 3.0, 80_000),
    Asset("NVDA", 125.0, 0.50, 4.0, 80_000),
)


@dataclass(frozen=True)
class Tier:
    max_utilisation: float            # upper bound of |position| / limit
    widen: float                      # half-spread multiplier on the side that adds inventory
    shade: float                      # fraction of half-spread removed on the side that reduces it


TIERS = (
    Tier(0.20, 1.00, 0.00),
    Tier(0.40, 1.30, 0.15),
    Tier(0.60, 1.75, 0.30),
    Tier(0.80, 2.50, 0.50),
    Tier(1.00, 4.00, 0.70),
)


def tier_for(position: int, limit: int) -> int:
    utilisation = abs(position) / limit
    for i, tier in enumerate(TIERS):
        if utilisation < tier.max_utilisation:
            return i
    return len(TIERS) - 1


class ReferencePriceFeed:
    """Geometric Brownian motion paths, generated up front so informed
    clients can trade on the near future (adverse selection)."""

    def __init__(self, assets: tuple[Asset, ...], steps: int, rng: np.random.Generator):
        dt = 1 / (TRADING_DAYS * TICKS_PER_DAY)
        vols = np.array([a.annual_vol for a in assets])
        shocks = rng.standard_normal((steps, len(assets)))
        log_returns = -0.5 * vols**2 * dt + vols * math.sqrt(dt) * shocks
        cumulative = np.vstack([np.zeros(len(assets)), np.cumsum(log_returns, axis=0)])
        self.paths = np.array([a.start_price for a in assets]) * np.exp(cumulative)

    def price(self, tick: int, i: int) -> float:
        return float(self.paths[tick, i])

    def future(self, tick: int, i: int, horizon: int) -> float:
        return float(self.paths[min(tick + horizon, len(self.paths) - 1), i])


@dataclass
class Book:
    asset: Asset
    position: int = 0
    cash: float = 0.0

    def apply(self, mm_side: str, size: int, price: float) -> None:
        sign = 1 if mm_side == "buy" else -1
        self.position += sign * size
        self.cash -= sign * size * price

    def mark_to_market(self, ref: float) -> float:
        return self.cash + self.position * ref


@dataclass(frozen=True)
class Quote:
    ref: float
    bid: float | None                 # None when filling would breach the position limit
    ask: float | None
    tier: int

    @property
    def spread_bps(self) -> float:
        if self.bid is None or self.ask is None:
            return math.nan
        return (self.ask - self.bid) / self.ref * 1e4


class MarketMaker:
    def quote(self, book: Book, ref: float, size: int) -> Quote:
        asset = book.asset
        tier_idx = tier_for(book.position, asset.position_limit)
        tier = TIERS[tier_idx]
        h = asset.base_half_spread_bps
        accumulate, reduce = h * tier.widen, h * (1 - tier.shade)

        if book.position > 0:         # long: buying adds risk, selling sheds it
            bid_hs, ask_hs = accumulate, reduce
        elif book.position < 0:
            bid_hs, ask_hs = reduce, accumulate
        else:
            bid_hs = ask_hs = h

        bid = math.floor(ref * (1 - bid_hs / 1e4) * 100) / 100
        ask = math.ceil(ref * (1 + ask_hs / 1e4) * 100) / 100
        if book.position + size > asset.position_limit:
            bid = None
        if book.position - size < -asset.position_limit:
            ask = None
        return Quote(ref, bid, ask, tier_idx)


@dataclass(frozen=True)
class RFQ:
    client: str
    side: str                         # client's intent: "buy" or "sell"
    size: int
    informed: bool
    tolerance_bps: float              # worst distance from reference the client will accept


class HedgeFundFlow:
    CLIENTS = ("HF-Alpha", "HF-Bravo", "HF-Carry", "HF-Delta", "HF-Echo")

    def __init__(self, rng: np.random.Generator, informed_share: float = 0.1, horizon: int = 60):
        self.rng = rng
        self.informed_share = informed_share
        self.horizon = horizon        # ticks an informed client can see ahead
        self.buy_bias: dict[int, float] = {}

    def _drift_bias(self, i: int) -> float:
        """Uninformed flow crowds one way for a while: the probability of a client
        buy is a mean-reverting random walk around 0.5, clipped to [0.15, 0.85]."""
        bias = self.buy_bias.get(i, 0.5)
        bias += -0.01 * (bias - 0.5) + self.rng.normal(0, 0.02)
        self.buy_bias[i] = bias = float(np.clip(bias, 0.15, 0.85))
        return bias

    def request(self, tick: int, i: int, asset: Asset, feed: ReferencePriceFeed) -> RFQ:
        informed = self.rng.random() < self.informed_share
        bias = self._drift_bias(i)
        if informed:
            side = "buy" if feed.future(tick, i, self.horizon) > feed.price(tick, i) else "sell"
        else:
            side = "buy" if self.rng.random() < bias else "sell"
        mean_lots = asset.position_limit / 10_000  # average ticket ~1% of the limit
        size = int(self.rng.geometric(1 / mean_lots)) * 100
        tolerance = asset.base_half_spread_bps * self.rng.uniform(0.8, 2.5)
        if informed:
            tolerance *= 2
        return RFQ(str(self.rng.choice(self.CLIENTS)), side, size, informed, tolerance)

    @staticmethod
    def respond(rfq: RFQ, quote: Quote) -> float | None:
        """Return the fill price if the client trades on the quote, else None."""
        if rfq.side == "buy":
            limit_price = quote.ref * (1 + rfq.tolerance_bps / 1e4)
            return quote.ask if quote.ask is not None and quote.ask <= limit_price else None
        limit_price = quote.ref * (1 - rfq.tolerance_bps / 1e4)
        return quote.bid if quote.bid is not None and quote.bid >= limit_price else None
