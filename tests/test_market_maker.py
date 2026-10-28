import math

import numpy as np
import pytest

from market_maker import (ASSETS, TIERS, Book, HedgeFundFlow, MarketMaker, Quote, RFQ,
                          ReferencePriceFeed, tier_for)
from simulate import simulate

AAPL = ASSETS[0]                      # limit 50,000, base half-spread 2 bps


@pytest.mark.parametrize("position, tier", [(0, 0), (9_999, 0), (10_000, 1), (-25_000, 2),
                                            (35_000, 3), (45_000, 4), (50_000, 4)])
def test_tier_follows_utilisation(position, tier):
    assert tier_for(position, AAPL.position_limit) == tier


def test_flat_book_quotes_symmetric_spread():
    q = MarketMaker().quote(Book(AAPL), 100.0, 100)
    assert q.bid == pytest.approx(99.98)
    assert q.ask == pytest.approx(100.02)
    assert q.spread_bps == pytest.approx(4.0)


def test_long_book_widens_bid_and_tightens_offer():
    # 45% long -> T2: bid at 1.75x half-spread, offer at 0.70x
    q = MarketMaker().quote(Book(AAPL, position=22_500), 100.0, 100)
    assert 100.0 - q.bid > q.ask - 100.0
    assert q.tier == 2


def test_short_book_skews_the_other_way():
    q = MarketMaker().quote(Book(AAPL, position=-22_500), 100.0, 100)
    assert q.ask - 100.0 > 100.0 - q.bid


def test_side_that_would_breach_limit_is_not_quoted():
    q = MarketMaker().quote(Book(AAPL, position=49_950), 100.0, 100)
    assert q.bid is None and q.ask is not None
    assert math.isnan(q.spread_bps)


def test_book_tracks_position_cash_and_pnl():
    book = Book(AAPL)
    book.apply("buy", 100, 99.0)
    book.apply("sell", 40, 101.0)
    assert book.position == 60
    assert book.cash == pytest.approx(-9_900 + 4_040)
    assert book.mark_to_market(100.0) == pytest.approx(-5_860 + 6_000)


def test_client_trades_only_inside_tolerance():
    quote = Quote(ref=100.0, bid=99.97, ask=100.03, tier=0)
    assert HedgeFundFlow.respond(RFQ("HF", "buy", 100, False, 4.0), quote) == 100.03
    assert HedgeFundFlow.respond(RFQ("HF", "buy", 100, False, 2.0), quote) is None
    assert HedgeFundFlow.respond(RFQ("HF", "sell", 100, False, 4.0), quote) == 99.97


def test_price_feed_starts_at_start_price_and_stays_positive():
    feed = ReferencePriceFeed(ASSETS, 1_000, np.random.default_rng(0))
    assert feed.paths.shape == (1_001, len(ASSETS))
    assert feed.paths[0].tolist() == [a.start_price for a in ASSETS]
    assert (feed.paths > 0).all()


def test_tiers_widen_more_and_shade_more_as_inventory_grows():
    assert [t.widen for t in TIERS] == sorted(t.widen for t in TIERS)
    assert [t.shade for t in TIERS] == sorted(t.shade for t in TIERS)


def test_simulation_respects_position_limits(tmp_path):
    import pandas as pd
    simulate(steps=500, seed=1, out_dir=tmp_path)
    positions = pd.read_csv(tmp_path / "positions.csv")
    for a in ASSETS:
        assert positions[f"pos_{a.ticker}"].abs().max() <= a.position_limit
