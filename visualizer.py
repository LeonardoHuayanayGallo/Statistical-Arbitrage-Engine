"""
visualizer.py
Generates all charts for the statistical arbitrage strategy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

plt.style.use("seaborn-v0_8-whitegrid")

COLORS = {
    "price_a":   "#1F3864",
    "price_b":   "#F44336",
    "spread":    "#2196F3",
    "zscore":    "#9C27B0",
    "long":      "#4CAF50",
    "short":     "#F44336",
    "portfolio": "#1F3864",
    "gold":      "#F2C94C",
}


def plot_price_series(
    price_a: pd.Series,
    price_b: pd.Series,
    ticker_a: str,
    ticker_b: str,
    save_dir: str = "output"
):
    """Plot 1: Normalized price series of the pair."""
    fig, ax = plt.subplots(figsize=(14, 5))

    norm_a = price_a / price_a.iloc[0] * 100
    norm_b = price_b / price_b.iloc[0] * 100

    ax.plot(norm_a.index, norm_a, color=COLORS["price_a"],
            linewidth=1.5, label=f"{ticker_a} (normalized)")
    ax.plot(norm_b.index, norm_b, color=COLORS["price_b"],
            linewidth=1.5, label=f"{ticker_b} (normalized)")

    ax.set_title(
        f"Cointegrated Pair: {ticker_a} vs {ticker_b}\n"
        "Normalized to 100 at start — prices move together long-term",
        fontsize=13, fontweight="bold"
    )
    ax.set_ylabel("Normalized Price (base = 100)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=10)
    plt.tight_layout()

    path = os.path.join(save_dir, "01_price_series.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {path}")


def plot_spread_and_zscore(
    spread: pd.Series,
    zscore: pd.Series,
    ticker_a: str,
    ticker_b: str,
    entry_threshold: float = 2.0,
    save_dir: str = "output"
):
    """Plot 2: Spread and Z-score with entry/exit threshold lines."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top: Spread
    ax1.plot(spread.index, spread, color=COLORS["spread"],
             linewidth=1.2, label="Spread")
    ax1.axhline(spread.mean(), color="black", linestyle="--",
                linewidth=1, alpha=0.5, label="Mean")
    ax1.fill_between(spread.index, spread, spread.mean(),
                     alpha=0.1, color=COLORS["spread"])
    ax1.set_title(
        f"Cointegrated Spread: {ticker_a} − β × {ticker_b}\n"
        "Stationary spread confirms cointegration",
        fontweight="bold"
    )
    ax1.set_ylabel("Spread ($)")
    ax1.legend(fontsize=9)

    # Bottom: Z-score with thresholds
    ax2.plot(zscore.index, zscore, color=COLORS["zscore"],
             linewidth=1.2, label="Z-score")
    ax2.axhline(0,                   color="black", linewidth=1)
    ax2.axhline(entry_threshold,     color=COLORS["short"], linestyle="--",
                linewidth=1.5, label=f"Short entry (+{entry_threshold}σ)")
    ax2.axhline(-entry_threshold,    color=COLORS["long"],  linestyle="--",
                linewidth=1.5, label=f"Long entry (−{entry_threshold}σ)")
    ax2.axhline(0.5,                 color="gray", linestyle=":",
                linewidth=1, alpha=0.7, label="Exit threshold")
    ax2.axhline(-0.5,                color="gray", linestyle=":",
                linewidth=1, alpha=0.7)

    ax2.fill_between(zscore.index, zscore, entry_threshold,
                     where=(zscore > entry_threshold),
                     alpha=0.2, color=COLORS["short"], label="Short signal")
    ax2.fill_between(zscore.index, zscore, -entry_threshold,
                     where=(zscore < -entry_threshold),
                     alpha=0.2, color=COLORS["long"], label="Long signal")

    ax2.set_title("Rolling Z-Score — Trading Signals", fontweight="bold")
    ax2.set_ylabel("Z-Score (σ)")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=8, ncol=3)
    ax2.set_ylim(-5, 5)

    plt.suptitle("Spread Analysis & Signal Generation",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(save_dir, "02_spread_zscore.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {path}")


def plot_signals_on_price(
    price_a: pd.Series,
    signal_df: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    save_dir: str = "output"
):
    """Plot 3: Price of asset A with long/short entry points marked."""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(price_a.index, price_a, color=COLORS["price_a"],
            linewidth=1.2, alpha=0.8, label=f"{ticker_a} Price")

    # Long entries
    long_entries  = signal_df[
        (signal_df["signal"] == 1) &
        (signal_df["signal"].shift(1) != 1)
    ]
    # Short entries
    short_entries = signal_df[
        (signal_df["signal"] == -1) &
        (signal_df["signal"].shift(1) != -1)
    ]
    # Exits
    exits = signal_df[
        (signal_df["signal"] == 0) &
        (signal_df["signal"].shift(1) != 0)
    ]

    ax.scatter(
        long_entries.index,
        price_a[long_entries.index],
        marker="^", color=COLORS["long"], s=80,
        zorder=5, label="Long Entry (buy A, sell B)"
    )
    ax.scatter(
        short_entries.index,
        price_a[short_entries.index],
        marker="v", color=COLORS["short"], s=80,
        zorder=5, label="Short Entry (sell A, buy B)"
    )
    ax.scatter(
        exits.index,
        price_a[exits.index],
        marker="x", color="gray", s=50,
        zorder=5, label="Exit", alpha=0.6
    )

    ax.set_title(
        f"Trading Signals on {ticker_a} Price\n"
        "Automated Long/Short entry and exit signals",
        fontsize=13, fontweight="bold"
    )
    ax.set_ylabel(f"{ticker_a} Price ($)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=9)
    plt.tight_layout()

    path = os.path.join(save_dir, "03_trading_signals.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {path}")


def plot_portfolio_performance(
    portfolio_value: pd.Series,
    price_a: pd.Series,
    price_b: pd.Series,
    ticker_a: str,
    ticker_b: str,
    metrics: dict,
    save_dir: str = "output"
):
    """Plot 4: Strategy portfolio value vs buy-and-hold benchmarks."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # Normalize benchmarks to start at 1
    bench_a = price_a / price_a.iloc[0]
    bench_b = price_b / price_b.iloc[0]

    # Align all series
    common = portfolio_value.index.intersection(bench_a.index)
    portfolio_value = portfolio_value[common]
    bench_a = bench_a[common]
    bench_b = bench_b[common]

    ax1.plot(portfolio_value.index, portfolio_value,
             color=COLORS["portfolio"], linewidth=2,
             label=f"Strategy (Sharpe: {metrics['sharpe_ratio']:.2f})")
    ax1.plot(bench_a.index, bench_a, color=COLORS["price_a"],
             linewidth=1, alpha=0.6, linestyle="--", label=f"{ticker_a} B&H")
    ax1.plot(bench_b.index, bench_b, color=COLORS["price_b"],
             linewidth=1, alpha=0.6, linestyle="--", label=f"{ticker_b} B&H")
    ax1.axhline(1, color="black", linewidth=0.8, linestyle=":")
    ax1.set_title(
        f"Strategy Portfolio Value vs Buy-and-Hold\n"
        f"Total Return: {metrics['total_return']}% | "
        f"Annual: {metrics['annual_return']}% | "
        f"Sharpe: {metrics['sharpe_ratio']}",
        fontweight="bold"
    )
    ax1.set_ylabel("Portfolio Value ($1 initial)")
    ax1.legend(fontsize=9)
    ax1.fill_between(portfolio_value.index, portfolio_value, 1,
                     where=(portfolio_value >= 1),
                     alpha=0.1, color=COLORS["long"])
    ax1.fill_between(portfolio_value.index, portfolio_value, 1,
                     where=(portfolio_value < 1),
                     alpha=0.1, color=COLORS["short"])

    # Drawdown
    rolling_max = portfolio_value.cummax()
    drawdown    = (portfolio_value - rolling_max) / rolling_max * 100
    ax2.fill_between(drawdown.index, drawdown, 0,
                     color=COLORS["short"], alpha=0.4, label="Drawdown")
    ax2.plot(drawdown.index, drawdown, color=COLORS["short"], linewidth=0.8)
    ax2.axhline(metrics["max_drawdown"], color="darkred", linestyle="--",
                linewidth=1.5,
                label=f"Max Drawdown: {metrics['max_drawdown']}%")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.set_title("Strategy Drawdown", fontweight="bold")
    ax2.legend(fontsize=9)

    plt.suptitle("Backtest Performance — Statistical Arbitrage Strategy",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(save_dir, "04_portfolio_performance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {path}")


def plot_cointegration_heatmap(
    pairs_df: pd.DataFrame,
    save_dir: str = "output"
):
    """Plot 5: Heatmap of cointegration p-values across all pairs."""
    import seaborn as sns

    tickers = sorted(set(
        list(pairs_df["ticker_a"]) + list(pairs_df["ticker_b"])
    ))
    n = len(tickers)
    matrix = pd.DataFrame(np.ones((n, n)), index=tickers, columns=tickers)

    for _, row in pairs_df.iterrows():
        a, b = row["ticker_a"], row["ticker_b"]
        if a in tickers and b in tickers:
            matrix.loc[a, b] = row["coint_p_value"]
            matrix.loc[b, a] = row["coint_p_value"]

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.eye(n, dtype=bool)
    sns.heatmap(
        matrix,
        annot=True, fmt=".3f",
        cmap="RdYlGn_r",
        vmin=0, vmax=0.1,
        mask=mask,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Cointegration p-value (lower = stronger)"}
    )
    ax.set_title(
        "Cointegration P-Value Heatmap\n"
        "Green = strong cointegration (p < 0.05) | "
        "Red = no cointegration",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()

    path = os.path.join(save_dir, "05_cointegration_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {path}")