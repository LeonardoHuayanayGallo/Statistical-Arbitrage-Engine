"""
data_loader.py
Downloads historical price data via yfinance.
Uses batch download to minimize API requests and avoid rate limiting.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import time


def download_prices(
    tickers: list,
    start: str = "2018-01-01",
    end: str   = "2024-12-31",
    save_dir: str = "data"
) -> pd.DataFrame:

    os.makedirs(save_dir, exist_ok=True)
    print(f"[DATA] Downloading price data for: {tickers}")

    # Batch download — single request for all tickers
    for attempt in range(3):
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                ignore_tz=True,
                group_by="ticker"
            )

            if raw.empty:
                print(f"[DATA] Attempt {attempt+1} returned empty — waiting 30s...")
                time.sleep(30)
                continue

            # Extract Close prices for each ticker
            all_prices = {}
            for ticker in tickers:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        close = raw[ticker]["Close"]
                    else:
                        close = raw["Close"]

                    close = close.dropna()
                    if len(close) > 100:
                        all_prices[ticker] = close
                        print(f"[DATA] {ticker}: {len(close)} observations")
                    else:
                        print(f"[DATA] {ticker}: insufficient data — skipping")
                except Exception as e:
                    print(f"[DATA] {ticker}: error — {e}")

            if all_prices:
                break

        except Exception as e:
            print(f"[DATA] Attempt {attempt+1} failed: {e}")
            print(f"[DATA] Waiting 30 seconds before retry...")
            time.sleep(30)

    if not all_prices:
        raise ValueError("No data downloaded after 3 attempts.")

    prices = pd.DataFrame(all_prices).ffill().dropna()

    print(f"\n[DATA] Total observations : {len(prices)}")
    print(f"[DATA] Date range         : {prices.index[0].date()} "
          f"to {prices.index[-1].date()}")

    prices.to_csv(os.path.join(save_dir, "prices.csv"))
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()