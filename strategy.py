"""
strategy.py
Implements the statistical arbitrage pairs trading strategy.

Logic:
1. Compute rolling Z-score of the cointegrated spread
2. Generate Long/Short signals when Z-score exceeds threshold
3. Exit when Z-score reverts toward zero
4. Track positions, P&L, and portfolio value over time

This replicates the core logic used by quant hedge funds
in equity market-neutral statistical arbitrage strategies.
"""

import numpy as np
import pandas as pd


def generate_signals(
    zscore: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold:  float = 0.5
) -> pd.DataFrame:
    """
    Generate Long/Short trading signals from Z-score.

    Signal conventions:
     +1 = LONG the spread  (Z < -entry) → buy A, sell B
     -1 = SHORT the spread (Z > +entry) → sell A, buy B
      0 = FLAT (no position)

    Parameters:
    - entry_threshold: Z-score level to open a position (default: 2.0)
    - exit_threshold:  Z-score level to close a position (default: 0.5)
    """
    signals  = pd.Series(0, index=zscore.index)
    position = 0

    for i in range(len(zscore)):
        z = zscore.iloc[i]

        if np.isnan(z):
            signals.iloc[i] = 0
            continue

        if position == 0:
            # No position — look for entry
            if z > entry_threshold:
                position = -1   # Short spread
            elif z < -entry_threshold:
                position = 1    # Long spread

        elif position == 1:
            # Long spread — exit if Z reverts above -exit_threshold
            if z > -exit_threshold:
                position = 0

        elif position == -1:
            # Short spread — exit if Z reverts below +exit_threshold
            if z < exit_threshold:
                position = 0

        signals.iloc[i] = position

    return pd.DataFrame({
        "zscore":  zscore,
        "signal":  signals,
        "long":    (signals == 1).astype(int),
        "short":   (signals == -1).astype(int),
    })


def backtest(
    price_a:         pd.Series,
    price_b:         pd.Series,
    beta:            float,
    zscore_window:   int   = 60,
    entry_threshold: float = 2.0,
    exit_threshold:  float = 0.5,
    transaction_cost: float = 0.001,   # 10 bps per trade (institutional estimate)
    slippage:         float = 0.0005,  # 5 bps slippage
) -> dict:
    """
    Full backtest of the pairs trading strategy.

    For each day:
    - If signal = +1:  LONG 1 unit of A, SHORT beta units of B
    - If signal = -1:  SHORT 1 unit of A, LONG beta units of B
    - If signal =  0:  No position (flat)

    Returns:
    - Daily P&L series
    - Portfolio value over time
    - Performance metrics (Sharpe, Drawdown, Win Rate)
    """
    from cointegration import compute_spread, compute_zscore

    # Align series
    common_idx = price_a.index.intersection(price_b.index)
    price_a = price_a[common_idx]
    price_b = price_b[common_idx]

    # Compute spread and Z-score
    spread = compute_spread(price_a, price_b, beta)
    zscore = compute_zscore(spread, window=zscore_window)

    # Generate signals
    signal_df = generate_signals(zscore, entry_threshold, exit_threshold)

    # Compute daily returns of each leg
    ret_a = price_a.pct_change()
    ret_b = price_b.pct_change()

    # Strategy daily P&L
    # Long spread:  +ret_a - beta*ret_b
    # Short spread: -ret_a + beta*ret_b
    signal      = signal_df["signal"]
    gross_pnl   = signal.shift(1) * (ret_a - beta * ret_b)

    # Transaction costs on position changes
    position_changes = signal.diff().abs()
    total_cost = (transaction_cost + slippage) * position_changes
    net_pnl = gross_pnl - total_cost

    # Portfolio value (starting at $1)
    portfolio_value = (1 + net_pnl.fillna(0)).cumprod()

    # ── Performance Metrics ────────────────────────────────────────
    trading_days = 252
    net_pnl_clean = net_pnl.dropna()

    # Annualized return
    total_return   = portfolio_value.iloc[-1] - 1
    n_years        = len(net_pnl_clean) / trading_days
    annual_return  = (1 + total_return) ** (1 / n_years) - 1

    # Sharpe Ratio (annualized, risk-free rate = 0 for simplicity)
    sharpe = (
        net_pnl_clean.mean() / net_pnl_clean.std() * np.sqrt(trading_days)
        if net_pnl_clean.std() > 0 else 0
    )

    # Maximum Drawdown
    rolling_max   = portfolio_value.cummax()
    drawdown      = (portfolio_value - rolling_max) / rolling_max
    max_drawdown  = drawdown.min()

    # Win rate (days with positive P&L when in position)
    in_position   = net_pnl_clean[signal.shift(1).reindex(net_pnl_clean.index) != 0]
    win_rate      = (in_position > 0).sum() / len(in_position) if len(in_position) > 0 else 0

    # Number of trades
    n_trades = int((position_changes > 0).sum() / 2)

    metrics = {
        "total_return":   round(total_return * 100, 2),
        "annual_return":  round(annual_return * 100, 2),
        "sharpe_ratio":   round(sharpe, 4),
        "max_drawdown":   round(max_drawdown * 100, 2),
        "win_rate":       round(win_rate * 100, 2),
        "n_trades":       n_trades,
        "transaction_cost_bps": int((transaction_cost + slippage) * 10000),
    }

    return {
        "metrics":         metrics,
        "portfolio_value": portfolio_value,
        "net_pnl":         net_pnl,
        "signal_df":       signal_df,
        "spread":          spread,
        "zscore":          zscore,
        "price_a":         price_a,
        "price_b":         price_b,
    }