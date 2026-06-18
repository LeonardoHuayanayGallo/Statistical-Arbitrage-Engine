# Quantitative Trading Strategy: Statistical Arbitrage & Pairs Trading Engine

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Data](https://img.shields.io/badge/Data-yfinance%20free-brightgreen)


Hi there 👋

Pairs trading is one of the cleanest examples of statistical arbitrage in practice. This project tests a universe of assets for cointegration, builds a Z-score-driven mean-reversion strategy on the strongest pair, and backtests it with real transaction costs — not a frictionless toy model.


## Overview

An end-to-end algorithmic pairs trading engine based on statistical
arbitrage — one of the most widely used strategies at quantitative
hedge funds.

The engine:
1. Tests a universe of assets for cointegration using the
   Augmented Dickey-Fuller (ADF) test and Engle-Granger method
2. Selects the most statistically robust cointegrated pair
3. Computes a dynamic spread and rolling Z-score
4. Generates automated Long/Short entry and exit signals
5. Backtests the strategy over 5+ years with realistic transaction
   costs and slippage

All data pulled free via `yfinance`.

---

## Strategy Logic

Two assets are **cointegrated** if their prices move together
long-term even though each is individually a random walk.
The spread between them is stationary and mean-reverting.

```
Spread = Price_A − β × Price_B

Z-score = (Spread − Rolling_Mean) / Rolling_Std

Signal:
  Z > +2.0 → SHORT spread (sell A, buy B)
  Z < −2.0 → LONG spread  (buy A, sell B)
  |Z| < 0.5 → EXIT position
```

---

## Output

![Image Alt](https://github.com/LeonardoHuayanayGallo/Statistical-Arbitrage-Engine/blob/59b280467ed3fbb4c56a7f0ae3665b4d8e1552b5/output/01_price_series.png)

![Image Alt](https://github.com/LeonardoHuayanayGallo/Statistical-Arbitrage-Engine/blob/59b280467ed3fbb4c56a7f0ae3665b4d8e1552b5/output/02_spread_zscore.png)

![Image Alt](https://github.com/LeonardoHuayanayGallo/Statistical-Arbitrage-Engine/blob/59b280467ed3fbb4c56a7f0ae3665b4d8e1552b5/output/03_trading_signals.png)

![Image Alt](https://github.com/LeonardoHuayanayGallo/Statistical-Arbitrage-Engine/blob/59b280467ed3fbb4c56a7f0ae3665b4d8e1552b5/output/04_portfolio_performance.png)

![Image Alt](https://github.com/LeonardoHuayanayGallo/Statistical-Arbitrage-Engine/blob/59b280467ed3fbb4c56a7f0ae3665b4d8e1552b5/output/05_cointegration_heatmap.png)

---

## Methodology

**Cointegration Testing:**
- Engle-Granger two-step test on all pairwise combinations
- ADF test on residuals to confirm spread stationarity
- Hedge ratio estimated via OLS regression

**Signal Generation:**
- Rolling 60-day Z-score of the spread
- Entry at ±2.0 standard deviations
- Exit at ±0.5 standard deviations (mean reversion)

**Backtest Assumptions:**
- Transaction cost: 10 bps per trade
- Slippage: 5 bps per trade
- Equal dollar weighting, daily rebalancing
- No leverage

---

## Technologies

`Python` `Statsmodels` `NumPy` `Pandas` `yfinance` `Matplotlib` `Seaborn`
