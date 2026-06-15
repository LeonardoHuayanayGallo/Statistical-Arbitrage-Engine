"""
main.py
Full pipeline:
  1. Download price data (yfinance — free, no API key)
  2. Test all pairs for cointegration (ADF + Engle-Granger)
  3. Select the best cointegrated pair
  4. Compute spread, Z-score, and trading signals
  5. Backtest the strategy over 5 years
  6. Generate performance metrics and charts

Usage:
  python3 main.py
  python3 main.py --tickers XOM CVX COP BP --start 2018-01-01 --end 2024-12-31
"""

import argparse
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

import data_loader
import cointegration
import strategy
import visualizer


def print_metrics(metrics: dict, ticker_a: str, ticker_b: str):
    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS: {ticker_a} / {ticker_b}")
    print(f"{'='*60}")
    print(f"  Total Return       : {metrics['total_return']}%")
    print(f"  Annualized Return  : {metrics['annual_return']}%")
    print(f"  Sharpe Ratio       : {metrics['sharpe_ratio']}")
    print(f"  Max Drawdown       : {metrics['max_drawdown']}%")
    print(f"  Win Rate           : {metrics['win_rate']}%")
    print(f"  Number of Trades   : {metrics['n_trades']}")
    print(f"  Transaction Cost   : {metrics['transaction_cost_bps']} bps/trade")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Statistical Arbitrage — Pairs Trading Engine"
    )
    parser.add_argument(
        "--tickers", nargs="+",
        default=["XOM", "CVX", "COP", "BP", "SLB"],
        help="Universe of tickers to test (default: energy sector)"
    )
    parser.add_argument("--start",           default="2018-01-01")
    parser.add_argument("--end",             default="2024-12-31")
    parser.add_argument("--zscore_window",   type=int,   default=60)
    parser.add_argument("--entry_threshold", type=float, default=2.0)
    parser.add_argument("--exit_threshold",  type=float, default=0.5)
    parser.add_argument(
        "--pair", nargs=2, default=None,
        help="Force a specific pair, e.g. --pair XOM CVX"
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  STATISTICAL ARBITRAGE ENGINE")
    print("  Cointegration-Based Pairs Trading")
    print("="*60 + "\n")

    # STEP 1: Download data
    print("─"*40)
    print("STEP 1: Downloading Market Data")
    print("─"*40)
    prices = data_loader.download_prices(
        args.tickers, start=args.start, end=args.end
    )

    # STEP 2: Find cointegrated pairs
    print("\n" + "─"*40)
    print("STEP 2: Cointegration Testing")
    print("─"*40)
    pairs_df = cointegration.find_cointegrated_pairs(prices)
    pairs_df.to_csv("output/cointegration_results.csv", index=False)
    print(f"\n[COINT] Full results saved to output/cointegration_results.csv")

    # STEP 3: Select best pair
    print("\n" + "─"*40)
    print("STEP 3: Selecting Best Pair")
    print("─"*40)

    if args.pair:
        ticker_a, ticker_b = args.pair
        row = pairs_df[
            (pairs_df["ticker_a"] == ticker_a) &
            (pairs_df["ticker_b"] == ticker_b)
        ]
        if row.empty:
            row = pairs_df[
                (pairs_df["ticker_a"] == ticker_b) &
                (pairs_df["ticker_b"] == ticker_a)
            ]
        if row.empty:
            print(f"[WARN] Pair {ticker_a}/{ticker_b} not found — "
                  f"using best pair instead")
            best = pairs_df.iloc[0]
        else:
            best = row.iloc[0]
    else:
        coint_pairs = pairs_df[pairs_df["cointegrated"]]
        if coint_pairs.empty:
            print("[WARN] No cointegrated pairs found at p<0.05 — "
                  "using best available pair")
            best = pairs_df.iloc[0]
        else:
            best = coint_pairs.iloc[0]

    ticker_a = best["ticker_a"]
    ticker_b = best["ticker_b"]
    beta     = best["hedge_ratio"]

    print(f"\n  Selected pair     : {ticker_a} / {ticker_b}")
    print(f"  Cointegration p   : {best['coint_p_value']:.6f}")
    print(f"  Hedge ratio (beta): {beta:.4f}")

    # STEP 4: Backtest
    print("\n" + "─"*40)
    print("STEP 4: Running Backtest")
    print("─"*40)
    results = strategy.backtest(
        price_a          = prices[ticker_a],
        price_b          = prices[ticker_b],
        beta             = beta,
        zscore_window    = args.zscore_window,
        entry_threshold  = args.entry_threshold,
        exit_threshold   = args.exit_threshold,
        transaction_cost = 0.001,
        slippage         = 0.0005,
    )

    print_metrics(results["metrics"], ticker_a, ticker_b)

    # Save results
    perf_df = pd.DataFrame({
        "date":            results["portfolio_value"].index,
        "portfolio_value": results["portfolio_value"].values,
        "net_pnl":         results["net_pnl"].values,
        "signal":          results["signal_df"]["signal"].values,
        "zscore":          results["zscore"].values,
    })
    perf_df.to_csv("output/backtest_results.csv", index=False)
    print("[RESULTS] Saved to output/backtest_results.csv")

    # STEP 5: Charts
    print("\n" + "─"*40)
    print("STEP 5: Generating Charts")
    print("─"*40)

    visualizer.plot_price_series(
        results["price_a"], results["price_b"],
        ticker_a, ticker_b
    )
    visualizer.plot_spread_and_zscore(
        results["spread"], results["zscore"],
        ticker_a, ticker_b,
        entry_threshold=args.entry_threshold
    )
    visualizer.plot_signals_on_price(
        results["price_a"], results["signal_df"],
        ticker_a, ticker_b
    )
    visualizer.plot_portfolio_performance(
        results["portfolio_value"],
        results["price_a"], results["price_b"],
        ticker_a, ticker_b,
        results["metrics"]
    )
    visualizer.plot_cointegration_heatmap(pairs_df)

    print("\n" + "="*60)
    print("  ANALYSIS COMPLETE")
    print("="*60)
    print(f"\n  Pair     : {ticker_a} / {ticker_b}")
    print(f"  Period   : {args.start} to {args.end}")
    print(f"  Sharpe   : {results['metrics']['sharpe_ratio']}")
    print(f"  Return   : {results['metrics']['total_return']}%")
    print(f"\n  Output files saved to /output/")
    print(f"  01_price_series.png")
    print(f"  02_spread_zscore.png")
    print(f"  03_trading_signals.png")
    print(f"  04_portfolio_performance.png")
    print(f"  05_cointegration_heatmap.png")
    print(f"  cointegration_results.csv")
    print(f"  backtest_results.csv\n")


if __name__ == "__main__":
    main()