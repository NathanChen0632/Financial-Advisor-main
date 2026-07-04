import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_prediction.data_collection import (
    download_stock_data, download_spy_context, TICKERS, BROAD_UNIVERSE
)
from stock_prediction.features import build_features, get_feature_columns
from stock_prediction.models import chronological_split, fit_feature_scaler
from stock_prediction.evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    print_summary_table,
)
from stock_prediction.backtesting import (
    run_backtest,
    compute_backtest_metrics,
    print_backtest_metrics,
    plot_equity_curve,
)
from stock_prediction.rl_agent import train_dqn_agent
from stock_prediction.live_signal import generate_live_signal
from stock_prediction.monitor import run_monitor
from stock_prediction.stress_test import run_stress_tests
from stock_prediction.benchmarks import run_benchmark_comparison
from stock_prediction.walk_forward import run_walk_forward
from stock_prediction.research_agent import run_research_agent


def run_pipeline(ticker: str, benchmark: bool = False):
    print(f"\n{'#'*60}")
    print(f"  TICKER: {ticker}")
    print(f"{'#'*60}")

    print("\n[1/4] Downloading 10 years of data (2015-2025)...")
    df     = download_stock_data(ticker)
    # SPY is used as a market context feature, excluded for SPY itself
    spy_df = None if ticker == "SPY" else download_spy_context()

    print("\n[2/4] Engineering features...")
    feat_df      = build_features(df, spy_df=spy_df)
    feature_cols = get_feature_columns(feat_df)
    print(f"  Features: {len(feature_cols)}  |  Samples: {len(feat_df)}")

    print("\n[3/4] Splitting data...")
    splits = chronological_split(feat_df, feature_cols)
    print(f"  Train: {len(splits['X_train'])}  Val: {len(splits['X_val'])}  Test: {len(splits['X_test'])}")

    # Train and val are merged so the agent sees all pre-test history.
    # Val was used during development to tune hyperparameters without
    # touching the test set, combining them now maximizes training data.
    n_train_full  = len(splits["X_train"]) + len(splits["X_val"])
    X_train_full  = np.concatenate([splits["X_train"], splits["X_val"]])
    returns_train = feat_df["daily_return"].values[:n_train_full]

    # Scaler is fit on training data only and stored on the agent.
    # The agent applies it internally during inference so the test set
    # is never touched before evaluation, prevents data leakage.
    scaler         = fit_feature_scaler(X_train_full)
    X_train_scaled = scaler.transform(X_train_full)

    # Close prices are needed by the trading environment to compute
    # stop/target levels and P&L in dollar terms during training.
    prices_all   = df.loc[feat_df.index, "Close"].values.flatten()
    prices_train = prices_all[:n_train_full]

    print("\n[4/4] Training DQN agent with R/R gating, ATR stops, volume & MA rules...")
    dqn = train_dqn_agent(
        X_train=X_train_full,
        daily_returns_train=returns_train,
        prices_train=prices_train,
        feature_cols=feature_cols,
        n_episodes=1000,
        X_train_scaled=X_train_scaled,
    )
    dqn.scaler = scaler

    # Classification metrics show how often the agent correctly predicts
    # direction — useful for comparison against supervised baselines.
    print("\n  Evaluating on held-out test set...")
    results = {"DQN (RL Agent)": evaluate_model(
        "DQN (RL Agent)", dqn, splits["X_test"], splits["y_test"]
    )}
    plot_confusion_matrix(f"{ticker}_DQN", dqn, splits["X_test"], splits["y_test"])
    print_summary_table(results)

    # Backtest runs the full trading simulation on the test period.
    # Passing raw test_df (not pre-scaled) because the agent scales internally.
    print(f"\n  Running backtest with disciplined trading rules...")
    prices_test_full = prices_all[n_train_full:]
    prices_test      = prices_test_full[:len(splits["test_df"])]
    bt_df   = run_backtest(dqn, splits["test_df"], feature_cols, prices=prices_test)
    metrics = compute_backtest_metrics(bt_df)
    print_backtest_metrics(metrics, f"{ticker} — DQN (RL Agent)")
    plot_equity_curve(bt_df, f"{ticker}_DQN")

    # Benchmark comparison: DQN vs MA crossover, RSI, momentum, buy-and-hold
    if benchmark:
        print(f"\n  Running strategy benchmark comparison...")
        run_benchmark_comparison(
            ticker=ticker,
            dqn_model=dqn,
            test_df=splits["test_df"],
            feature_cols=feature_cols,
            prices=prices_test,
        )

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="CS5100 — DQN RL Stock Trading Strategy"
    )
    parser.add_argument("--ticker", nargs="+", default=TICKERS, metavar="TICKER")
    parser.add_argument("--signal", action="store_true",
                        help="One-shot live signal for today.")
    parser.add_argument("--monitor", action="store_true",
                        help="Continuous live monitor (polls every --interval minutes).")
    parser.add_argument("--interval", type=int, default=5, metavar="MINUTES")
    parser.add_argument("--no-market-check", action="store_true")
    parser.add_argument("--email", action="store_true",
                        help="Send email alerts on BUY/SELL signals.")
    parser.add_argument("--alpaca", action="store_true",
                        help="Submit orders to Alpaca on BUY/SELL signals (paper by default).")
    parser.add_argument("--live", action="store_true",
                        help="Use live Alpaca account instead of paper. Real money.")
    parser.add_argument("--stress-test", action="store_true",
                        help="Run bear-market stress tests (2008, 2020, 2022).")
    parser.add_argument("--stress-episodes", type=int, default=50, metavar="N")
    parser.add_argument("--benchmark", action="store_true",
                        help="Compare DQN vs MA crossover, RSI, and momentum strategies.")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run walk-forward validation across 3 non-overlapping time windows.")
    parser.add_argument("--walk-forward-episodes", type=int, default=200, metavar="N",
                        help="DQN training episodes per walk-forward window (default: 200).")
    parser.add_argument("--walk-forward-seeds", type=int, default=3, metavar="N",
                        help="DQN seeds per walk-forward window; results reported mean±std (default: 3).")
    parser.add_argument("--broad", action="store_true",
                        help="Use the broad 30+ ticker universe instead of default tickers.")
    parser.add_argument("--research", action="store_true",
                        help="Use Claude AI to screen stocks and pick the best candidates.")
    parser.add_argument("--no-claude", action="store_true",
                        help="Use rule-based fallback instead of Claude API for --research.")
    parser.add_argument("--straddle", action="store_true",
                        help="Run long straddle options strategy alongside equity trades (requires --alpaca).")
    return parser.parse_args()


def main():
    args = parse_args()

    # --broad swaps in the wider validation universe
    tickers = [t.upper() for t in args.ticker]
    if args.broad:
        tickers = BROAD_UNIVERSE
        print(f"  Using broad universe: {len(tickers)} tickers across 10 sectors")

    # --research uses Claude AI to screen the universe and pick the best candidates.
    # Runs first so the selected tickers flow into every subsequent step.
    if args.research:
        tickers = run_research_agent(use_claude=not args.no_claude)

    # Steps run in order — all flags are stackable.
    # Walk-forward and stress-test run first as validation passes,
    # then backtest/benchmark, then live signal or monitor last
    # (monitor is blocking so it always goes at the end).

    if args.walk_forward:
        run_walk_forward(
            tickers=tickers,
            n_episodes=args.walk_forward_episodes,
            n_seeds=args.walk_forward_seeds,
        )

    if args.stress_test:
        run_stress_tests(
            tickers=tickers,
            n_episodes=args.stress_episodes,
        )

    # Run the full backtest + optional benchmark when explicitly requested,
    # or when no live mode is active (default behaviour).
    run_backtest_pass = (
        args.benchmark
        or (not args.monitor and not args.signal
            and not args.walk_forward and not args.stress_test)
    )
    if run_backtest_pass:
        for ticker in tickers:
            try:
                run_pipeline(ticker, benchmark=args.benchmark)
            except Exception as e:
                print(f"\n[ERROR] {ticker}: {e}")
                import traceback; traceback.print_exc()

    if args.signal:
        for ticker in tickers:
            try:
                generate_live_signal(ticker)
            except Exception as e:
                print(f"\n[ERROR] {ticker}: {e}")

    # Monitor is always last — it blocks until Ctrl-C
    if args.monitor:
        run_monitor(
            tickers=tickers,
            interval_minutes=args.interval,
            skip_market_check=args.no_market_check,
            email_alerts=args.email,
            alpaca_trading=args.alpaca,
            alpaca_live=args.live,
            straddle_trading=args.straddle,
        )
        return

    print("\n\nAll results saved to the 'results/' directory.")


if __name__ == "__main__":
    main()
