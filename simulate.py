"""Run the market-making simulation and write every RFQ, trade, position
snapshot and per-tick P&L to CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from market_maker import ASSETS, TICKS_PER_DAY, Book, HedgeFundFlow, MarketMaker, ReferencePriceFeed

RFQ_PROBABILITY = 0.50                # chance per tick, per asset, of an incoming request


def simulate(steps: int, seed: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    feed = ReferencePriceFeed(ASSETS, steps, rng)
    flow = HedgeFundFlow(rng)
    mm = MarketMaker()
    books = [Book(a) for a in ASSETS]
    tickers = [a.ticker for a in ASSETS]

    with (
        open(out_dir / "rfqs.csv", "w", newline="") as rfq_f,
        open(out_dir / "trades.csv", "w", newline="") as trade_f,
        open(out_dir / "positions.csv", "w", newline="") as pos_f,
        open(out_dir / "pnl.csv", "w", newline="") as pnl_f,
    ):
        rfq_log, trade_log, pos_log, pnl_log = (csv.writer(f) for f in (rfq_f, trade_f, pos_f, pnl_f))
        rfq_log.writerow(["rfq_id", "tick", "ticker", "client", "client_side", "size", "informed",
                          "ref_price", "bid", "ask", "spread_bps", "tier", "position_before",
                          "filled", "fill_price"])
        trade_log.writerow(["trade_id", "rfq_id", "tick", "ticker", "client", "mm_side", "size",
                            "price", "ref_price", "edge_bps", "tier", "position_after"])
        pos_log.writerow(["trade_id", "tick", *(f"pos_{t}" for t in tickers), "total_pnl"])
        pnl_log.writerow(["tick", *(f"pnl_{t}" for t in tickers), "total_pnl"])

        rfq_id = trade_id = 0
        for tick in range(steps):
            for i, book in enumerate(books):
                if rng.random() >= RFQ_PROBABILITY:
                    continue
                ref = feed.price(tick, i)
                rfq = flow.request(tick, i, book.asset, feed)
                quote = mm.quote(book, ref, rfq.size)
                fill_price = flow.respond(rfq, quote)
                rfq_log.writerow([rfq_id, tick, book.asset.ticker, rfq.client, rfq.side, rfq.size,
                                  rfq.informed, round(ref, 4), quote.bid, quote.ask,
                                  round(quote.spread_bps, 3), quote.tier, book.position,
                                  fill_price is not None, fill_price])

                if fill_price is not None:
                    mm_side = "sell" if rfq.side == "buy" else "buy"
                    book.apply(mm_side, rfq.size, fill_price)
                    edge = (fill_price - ref) if mm_side == "sell" else (ref - fill_price)
                    trade_log.writerow([trade_id, rfq_id, tick, book.asset.ticker, rfq.client,
                                        mm_side, rfq.size, fill_price, round(ref, 4),
                                        round(edge / ref * 1e4, 3), quote.tier, book.position])
                    total = sum(b.mark_to_market(feed.price(tick, j)) for j, b in enumerate(books))
                    pos_log.writerow([trade_id, tick, *(b.position for b in books), round(total, 2)])
                    trade_id += 1
                rfq_id += 1

            marks = [b.mark_to_market(feed.price(tick + 1, j)) for j, b in enumerate(books)]
            pnl_log.writerow([tick, *(round(m, 2) for m in marks), round(sum(marks), 2)])

    print(f"{rfq_id} RFQs, {trade_id} trades ({trade_id / max(rfq_id, 1):.0%} hit rate)")
    for book in books:
        print(f"  {book.asset.ticker:<5} position {book.position:>8,}  "
              f"P&L {book.mark_to_market(feed.price(steps, ASSETS.index(book.asset))):>12,.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=TICKS_PER_DAY, help="ticks to simulate (default: one session)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    simulate(args.steps, args.seed, args.out)
    if not args.no_plots:
        from plots import plot_all
        plot_all(args.out)


if __name__ == "__main__":
    main()
