# Quantitative Trading Strategy: Statistical Arbitrage & Pairs Trading Engine

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Data](https://img.shields.io/badge/Data-yfinance%20free-brightgreen)

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

No API keys required — all data pulled free via `yfinance`.

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

## Architecture

```
yfinance → Price Data
               ↓
   Cointegration Testing (ADF + Engle-Granger)
   All pairwise combinations tested
               ↓
   Best pair selected by p-value
               ↓
   Spread → Z-score → Signals
               ↓
   Backtest (5 years, transaction costs + slippage)
               ↓
   5 Charts + 2 CSV files
```

---

## Setup

```bash
git clone https://github.com/your-username/statistical-arbitrage-engine
cd statistical-arbitrage-engine
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

```bash
# Default: Energy sector pairs (XOM, CVX, COP, BP, SLB)
python3 main.py

# Custom universe
python3 main.py --tickers JPM GS MS BAC --start 2018-01-01

# Force a specific pair
python3 main.py --tickers XOM CVX COP --pair XOM CVX
```

---

## Output

| File | Content |
|---|---|
| `01_price_series.png` | Normalized prices of the selected pair |
| `02_spread_zscore.png` | Spread and Z-score with signal thresholds |
| `03_trading_signals.png` | Long/short entry and exit points on price |
| `04_portfolio_performance.png` | Portfolio value vs buy-and-hold + drawdown |
| `05_cointegration_heatmap.png` | P-value heatmap across all pairs tested |
| `cointegration_results.csv` | Full cointegration test results |
| `backtest_results.csv` | Daily P&L, signals, portfolio value |

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
