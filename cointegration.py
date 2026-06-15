"""
cointegration.py
Tests all pairs in a universe for cointegration using
the Augmented Dickey-Fuller (ADF) test on the spread.

Two assets are cointegrated if their price spread is stationary
(mean-reverting) — even if each individual price series is non-stationary.
This is the statistical foundation of pairs trading.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from itertools import combinations


def test_stationarity(series: pd.Series, name: str = "") -> dict:
    """
    Augmented Dickey-Fuller test for stationarity.

    H0: Series has a unit root (non-stationary, random walk)
    H1: Series is stationary (mean-reverting)

    We REJECT H0 if p-value < 0.05 → series is stationary.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "name":      name,
        "adf_stat":  result[0],
        "p_value":   result[1],
        "stationary": result[1] < 0.05
    }


def estimate_hedge_ratio(price_a: pd.Series, price_b: pd.Series) -> float:
    """
    Estimate hedge ratio (beta) via OLS regression:
    price_A = alpha + beta * price_B + epsilon

    Beta tells us how many units of B to short per unit of A long.
    This ratio minimizes spread variance.
    """
    X = add_constant(price_b)
    model = OLS(price_a, X).fit()
    beta = model.params.iloc[1]
    return beta


def compute_spread(
    price_a: pd.Series,
    price_b: pd.Series,
    beta: float
) -> pd.Series:
    """
    Compute the dollar spread between two assets.
    Spread = Price_A - beta * Price_B

    If cointegrated, this spread is stationary and mean-reverts.
    """
    return price_a - beta * price_b


def compute_zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    """
    Compute rolling Z-score of the spread.

    Z = (Spread - Rolling_Mean) / Rolling_Std

    Z-score tells us how many standard deviations the spread
    is from its rolling mean. Used to generate trade signals:
    - Z > +2.0 → spread too wide → SHORT A, LONG B
    - Z < -2.0 → spread too narrow → LONG A, SHORT B
    - |Z| < 0.5 → spread reverted → EXIT position
    """
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std  = spread.rolling(window=window).std()
    zscore = (spread - rolling_mean) / rolling_std
    return zscore


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    p_value_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Tests all possible pairs in the universe for cointegration.
    Returns a DataFrame of cointegrated pairs sorted by p-value.

    Uses the Engle-Granger two-step cointegration test.
    """
    tickers = list(prices.columns)
    pairs   = list(combinations(tickers, 2))
    results = []

    print(f"[COINT] Testing {len(pairs)} pairs for cointegration...")

    for ticker_a, ticker_b in pairs:
        series_a = prices[ticker_a].dropna()
        series_b = prices[ticker_b].dropna()

        # Align series
        common_idx = series_a.index.intersection(series_b.index)
        series_a   = series_a[common_idx]
        series_b   = series_b[common_idx]

        if len(series_a) < 252:  # Need at least 1 year of data
            continue

        try:
            # Engle-Granger cointegration test
            score, p_value, _ = coint(series_a, series_b)

            # Estimate hedge ratio
            beta = estimate_hedge_ratio(series_a, series_b)

            # Spread and its stationarity
            spread  = compute_spread(series_a, series_b, beta)
            adf_res = test_stationarity(spread)

            results.append({
                "pair":          f"{ticker_a}/{ticker_b}",
                "ticker_a":      ticker_a,
                "ticker_b":      ticker_b,
                "coint_p_value": round(p_value, 6),
                "hedge_ratio":   round(beta, 4),
                "spread_adf_p":  round(adf_res["p_value"], 6),
                "cointegrated":  p_value < p_value_threshold,
            })

        except Exception as e:
            continue

    df = pd.DataFrame(results)
    if df.empty:
        raise ValueError("No valid pairs found.")

    df = df.sort_values("coint_p_value").reset_index(drop=True)

    coint_pairs = df[df["cointegrated"]]
    print(f"[COINT] Found {len(coint_pairs)} cointegrated pairs "
          f"(p < {p_value_threshold})")
    print(f"\n[COINT] Top 5 pairs:")
    print(df.head(5)[["pair", "coint_p_value", "hedge_ratio"]].to_string(index=False))

    return df