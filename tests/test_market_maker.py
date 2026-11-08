import math

import numpy as np
import pytest

from market_maker import ASSETS, RFQ, TIERS, Book, HedgeFundFlow, MarketMaker, Quote, ReferencePriceFeed, tier_for

AAPL = ASSETS[0]


def test_tier_boundaries():
    limit = 1000
    assert tier_for(0, limit) == 0
    assert tier_for(199, limit) == 0
    assert tier_for(200, limit) == 1
    assert tier_for(-450, limit) == 2
    assert tier_for(799, limit) == 3
    assert tier_for(1000, limit) == len(TIERS) - 1


def test_flat_book_quotes_symmetrically():
    q = MarketMaker().quote(Book(AAPL), 100.0, 100)
    assert q.tier == 0
    assert 100 - q.bid == pytest.approx(q.ask - 100, abs=0.011)  # equal up to cent rounding


@pytest.mark.parametrize("sign", [1, -1])
def test_skew_discourages_more_inventory(sign):
    """Long: the bid (which would add more) is further away than the ask. Short: the reverse."""
    book = Book(AAPL, position=sign * int(0.7 * AAPL.position_limit))
    q = MarketMaker().quote(book, 200.0, 100)
    adding, reducing = (200 - q.bid, q.ask - 200) if sign > 0 else (q.ask - 200, 200 - q.bid)
    assert adding > reducing


def test_spread_widens_with_inventory():
    mm = MarketMaker()
    spreads = [mm.quote(Book(AAPL, position=int(u * AAPL.position_limit)), 200.0, 100).spread_bps
               for u in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert spreads == sorted(spreads)
    assert spreads[-1] > spreads[0]


def test_no_quote_that_would_breach_the_limit():
    book = Book(AAPL, position=AAPL.position_limit - 50)
    q = MarketMaker().quote(book, 200.0, 100)
    assert q.bid is None and q.ask is not None
    assert math.isnan(q.spread_bps)


def test_book_accounting():
    book = Book(AAPL)
    book.apply("buy", 100, 200.0)
    book.apply("sell", 40, 205.0)
    assert book.position == 60
    assert book.cash == pytest.approx(-100 * 200 + 40 * 205)
    assert book.mark_to_market(210.0) == pytest.approx(book.cash + 60 * 210)


def test_client_fills_only_inside_tolerance():
    q = Quote(ref=100.0, bid=99.9, ask=100.1, tier=0)
    assert HedgeFundFlow.respond(RFQ("HF", "buy", 100, False, 15.0), q) == 100.1   # 10 bps away, accepts
    assert HedgeFundFlow.respond(RFQ("HF", "buy", 100, False, 5.0), q) is None     # too wide
    assert HedgeFundFlow.respond(RFQ("HF", "sell", 100, False, 15.0), q) == 99.9


def test_informed_clients_trade_with_the_future_move():
    rng = np.random.default_rng(0)
    feed = ReferencePriceFeed(ASSETS, 500, rng)
    flow = HedgeFundFlow(rng, informed_share=1.0, horizon=60)
    for tick in range(0, 400, 37):
        rfq = flow.request(tick, 0, AAPL, feed)
        up = feed.future(tick, 0, 60) > feed.price(tick, 0)
        assert rfq.side == ("buy" if up else "sell")


def test_price_paths_start_at_start_price():
    feed = ReferencePriceFeed(ASSETS, 10, np.random.default_rng(1))
    assert feed.paths.shape == (11, len(ASSETS))
    assert feed.price(0, 0) == pytest.approx(AAPL.start_price)
