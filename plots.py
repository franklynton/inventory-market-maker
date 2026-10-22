"""Plot cumulative P&L, inventory and quoted-spread dynamics from the CSV logs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from market_maker import ASSETS, TICKS_PER_HOUR, TIERS

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")   # fixed order, one per asset
SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BANDS = ("#f3f2ef", "#fcfcfb")
SPREAD_WINDOW = 25

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.labelsize": 9, "xtick.color": INK_3, "ytick.color": INK_3,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.6, "font.size": 9,
})

thousands = FuncFormatter(lambda v, _: f"${v / 1e3:,.0f}k")


def _hours(ticks: pd.Series) -> pd.Series:
    return ticks / TICKS_PER_HOUR


def plot_pnl(out_dir: Path) -> Path:
    pnl = pd.read_csv(out_dir / "pnl.csv")
    hours = _hours(pnl["tick"])
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True, constrained_layout=True)
    panels = [(a.ticker, f"pnl_{a.ticker}", c) for a, c in zip(ASSETS, SERIES)]
    panels.append(("Total book", "total_pnl", INK_2))

    for ax, (title, col, color) in zip(axes.flat, panels):
        ax.axhline(0, color=INK_3, linewidth=0.8)
        ax.plot(hours, pnl[col], color=color)
        final = pnl[col].iloc[-1]
        ax.set_title(f"{title}   {final / 1e3:+,.1f}k", color=INK)
        ax.yaxis.set_major_formatter(thousands)
    for ax in axes[-1]:
        ax.set_xlabel("Session hours")
    fig.suptitle("Cumulative mark-to-market P&L", x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    path = out_dir / "cumulative_pnl.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_spread_dynamics(out_dir: Path) -> Path:
    rfqs = pd.read_csv(out_dir / "rfqs.csv")
    positions = pd.read_csv(out_dir / "positions.csv")
    end_hour = (rfqs["tick"].max() + 1) / TICKS_PER_HOUR
    fig, axes = plt.subplots(2, len(ASSETS), figsize=(17, 6.5), sharex=True,
                             constrained_layout=True, gridspec_kw={"height_ratios": [1, 1]})

    for col, (asset, color) in enumerate(zip(ASSETS, SERIES)):
        pos_ax, spread_ax = axes[0, col], axes[1, col]

        # Inventory as % of limit, over the five tier bands.
        lower = 0.0
        for i, tier in enumerate(TIERS):
            upper = tier.max_utilisation * 100
            for sign in (1, -1):
                pos_ax.axhspan(sign * lower, sign * upper, color=BANDS[i % 2], zorder=0, linewidth=0)
            pos_ax.text(end_hour, (lower + upper) / 2, f" T{i}", va="center", fontsize=7, color=INK_3)
            lower = upper
        utilisation = positions[f"pos_{asset.ticker}"] / asset.position_limit * 100
        pos_ax.step(_hours(pd.concat([pd.Series([0]), positions["tick"]])),
                    pd.concat([pd.Series([0.0]), utilisation]), where="post", color=color)
        pos_ax.axhline(0, color=INK_3, linewidth=0.8)
        pos_ax.set_ylim(-100, 100)
        pos_ax.set_xlim(0, end_hour)
        pos_ax.grid(axis="y", visible=False)
        pos_ax.set_title(asset.ticker)

        # Quoted bid-offer spread, smoothed over RFQs; gaps are one-sided quotes at the limit.
        quotes = rfqs[rfqs["ticker"] == asset.ticker]
        smoothed = quotes["spread_bps"].rolling(SPREAD_WINDOW, min_periods=1).mean()
        spread_ax.plot(_hours(quotes["tick"]), smoothed, color=color)
        base = 2 * asset.base_half_spread_bps
        spread_ax.axhline(base, color=INK_3, linewidth=0.8, linestyle="--")
        spread_ax.text(0.02, base, " base", transform=spread_ax.get_yaxis_transform(),
                       va="bottom", fontsize=7, color=INK_3)
        spread_ax.set_ylim(0, None)
        spread_ax.set_xlabel("Session hours")

    axes[0, 0].set_ylabel("Position, % of limit")
    axes[1, 0].set_ylabel(f"Quoted spread, bps ({SPREAD_WINDOW}-RFQ avg)")
    fig.suptitle("Inventory tiers and spread dynamics", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", color=INK)
    path = out_dir / "spread_dynamics.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_all(out_dir: Path) -> None:
    for path in (plot_pnl(out_dir), plot_spread_dynamics(out_dir)):
        print(f"wrote {path}")


if __name__ == "__main__":
    plot_all(Path(__file__).parent / "output")
