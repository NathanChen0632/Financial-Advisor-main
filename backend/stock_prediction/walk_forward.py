from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stock_prediction.utils import OUTPUT_DIR, ensure_output_dir
from stock_prediction.data_collection import download_stock_data, download_spy_context
from stock_prediction.features import build_features, get_feature_columns
from stock_prediction.models import (
    fit_feature_scaler, build_random_forest, rf_up_probability, rf_filtered_signals,
)
from stock_prediction.rl_agent import train_dqn_agent
from stock_prediction.backtesting import run_backtest, compute_backtest_metrics
from stock_prediction.benchmarks import (
    ma_crossover_signals, rsi_signals, momentum_signals,
    combined_rule_signals, _metrics,
)


def _define_windows() -> list[dict]:
    # Three non-overlapping test windows covering different market regimes:
    # Window 1: 2015-2019 — bull market, low volatility
    # Window 2: 2019-2022 — COVID crash + recovery + rate hikes
    # Window 3: 2022-2025 — bear market + AI rally
    # Training always uses the 4 years before the test window so the agent
    # sees a full market cycle before being evaluated.
    return [
        {"name": "2017–2019 (Bull)",        "train_start": "2015-01-01", "train_end": "2017-01-01", "test_start": "2017-01-01", "test_end": "2019-01-01"},
        {"name": "2019–2022 (COVID+Rates)",  "train_start": "2017-01-01", "train_end": "2019-01-01", "test_start": "2019-01-01", "test_end": "2022-01-01"},
        {"name": "2022–2025 (Bear+AI Rally)","train_start": "2019-01-01", "train_end": "2022-01-01", "test_start": "2022-01-01", "test_end": "2025-01-01"},
    ]


def _run_window(
    ticker:      str,
    window:      dict,
    df:          pd.DataFrame,
    spy_df:      pd.DataFrame | None,
    n_episodes:  int,
    n_seeds:     int = 3,
) -> dict | None:
    try:
        train_start = window["train_start"]
        train_end   = window["train_end"]
        test_start  = window["test_start"]
        test_end    = window["test_end"]

        train_raw = df.loc[train_start:train_end]
        test_raw  = df.loc[test_start:test_end]

        if len(train_raw) < 200 or len(test_raw) < 50:
            print(f"    [{window['name']}] Not enough data — skipping.")
            return None

        # Align SPY to training and test windows
        spy_train = spy_df.loc[train_start:train_end] if spy_df is not None else None
        spy_test  = spy_df.loc[test_start:test_end]   if spy_df is not None else None

        # Build features separately for train and test to avoid leakage.
        # The scaler is fit only on train features — test features are transformed.
        combined_raw  = df.loc[train_start:test_end]
        combined_spy  = spy_df.loc[train_start:test_end] if spy_df is not None else None
        combined_feat = build_features(combined_raw, spy_df=combined_spy)
        feature_cols  = get_feature_columns(combined_feat)

        train_feat = combined_feat.loc[:train_end]
        test_feat  = combined_feat.loc[test_start:]

        if len(train_feat) < 100 or len(test_feat) < 30:
            return None

        X_train       = train_feat[feature_cols].values
        returns_train = train_feat["daily_return"].values
        prices_train  = df.loc[train_feat.index, "Close"].values.flatten()

        scaler         = fit_feature_scaler(X_train)
        X_train_scaled = scaler.transform(X_train)
        prices_test    = df.loc[test_feat.index, "Close"].values.flatten()
        daily_ret      = test_feat["daily_return"].values

        # Random-forest direction filter, trained once on the window's train set
        # (seed-independent). We reuse it to gate every seed's DQN entries.
        rf_up = None
        try:
            rf = build_random_forest(random_state=42)
            rf.fit(X_train, train_feat["Target"].values)
            rf_up = rf_up_probability(rf, test_feat[feature_cols].values)
        except Exception as e:
            print(f"    [{window['name']}] RF filter unavailable ({e}) — skipping ensemble.")

        # Train an independent agent per seed. DQN results swing a lot with the
        # initialization / exploration seed, so we report mean±std across seeds
        # rather than a single (possibly lucky) run.
        seeds         = [42 + i for i in range(max(1, n_seeds))]
        seed_metrics  = []
        seed_equities = []
        ens_metrics   = []
        ens_equities  = []
        bah_metrics   = None
        bah_equity    = None

        for seed in seeds:
            print(f"    [{window['name']}] Training {n_episodes} episodes (seed {seed})...")
            dqn = train_dqn_agent(
                X_train=X_train,
                daily_returns_train=returns_train,
                prices_train=prices_train,
                feature_cols=feature_cols,
                n_episodes=n_episodes,
                random_state=seed,
                X_train_scaled=X_train_scaled,
            )
            dqn.scaler = scaler

            bt_df = run_backtest(dqn, test_feat, feature_cols, prices=prices_test)
            m     = compute_backtest_metrics(bt_df)
            seed_metrics.append(m["Strategy"])
            seed_equities.append(bt_df["strategy_equity"].values)
            bah_metrics = m["Buy & Hold"]              # seed-independent
            bah_equity  = bt_df["bah_equity"].values

            # DQN + RF filter: keep this seed's DQN longs only when the RF agrees.
            if rf_up is not None:
                filt   = rf_filtered_signals(bt_df["signal"].values, rf_up, threshold=0.5)
                ens_bt = run_backtest(None, test_feat, feature_cols, signals=filt)
                ens_metrics.append(compute_backtest_metrics(ens_bt)["Strategy"])
                ens_equities.append(ens_bt["strategy_equity"].values)

        # Aggregate strategy metrics across seeds (mean ± std).
        keys            = list(seed_metrics[0].keys())
        dqn_mean        = {k: float(np.mean([s[k] for s in seed_metrics])) for k in keys}
        dqn_std         = {k: float(np.std([s[k]  for s in seed_metrics])) for k in keys}
        dqn_equity_mean = np.mean(np.array(seed_equities), axis=0)

        ens_mean = ens_std = None
        if ens_metrics:
            ens_mean = {k: float(np.mean([s[k] for s in ens_metrics])) for k in keys}
            ens_std  = {k: float(np.std([s[k]  for s in ens_metrics])) for k in keys}

        return {
            "window":        window["name"],
            "dqn":           dqn_mean,
            "dqn_std":       dqn_std,
            "ensemble":      ens_mean,
            "ensemble_std":  ens_std,
            "n_seeds":       len(seeds),
            "bah":           bah_metrics,
            "ma_cross":      _metrics(ma_crossover_signals(test_feat), daily_ret),
            "rsi":           _metrics(rsi_signals(test_feat),           daily_ret),
            "momentum":      _metrics(momentum_signals(test_feat),      daily_ret),
            "combined":      _metrics(combined_rule_signals(test_feat), daily_ret),
            "dqn_equity":    dqn_equity_mean,
            "bah_equity":    bah_equity,
            "test_index":    test_feat.index,
            "n_train_days":  len(train_feat),
            "n_test_days":   len(test_feat),
        }

    except Exception as e:
        print(f"    [{window['name']}] Failed: {e}")
        import traceback; traceback.print_exc()
        return None


def run_walk_forward(
    tickers:    list[str],
    n_episodes: int = 200,
    n_seeds:    int = 3,
) -> None:
    ensure_output_dir()
    windows = _define_windows()

    print(f"\n{'#'*70}")
    print(f"  WALK-FORWARD VALIDATION")
    print(f"  Tickers  : {tickers}")
    print(f"  Windows  : {len(windows)} non-overlapping test periods")
    print(f"  Episodes : {n_episodes} per window")
    print(f"  Seeds    : {n_seeds} per window (results reported mean±std)")
    print(f"{'#'*70}\n")

    # Download SPY once — shared across all tickers and all windows
    print("[1/3] Downloading SPY market context...")
    try:
        spy_df = download_spy_context(start="2015-01-01", end="2025-01-01")
    except Exception:
        spy_df = None

    all_ticker_results = {}

    for ticker in tickers:
        print(f"\n{'─'*70}")
        print(f"  TICKER: {ticker}")
        print(f"{'─'*70}")

        try:
            df = download_stock_data(ticker, start="2015-01-01", end="2025-01-01")
        except Exception as e:
            print(f"  [ERROR] Could not download {ticker}: {e}")
            continue

        window_results = []
        for window in windows:
            print(f"\n  Window: {window['name']}")
            result = _run_window(ticker, window, df, spy_df, n_episodes, n_seeds)
            if result is not None:
                window_results.append(result)

        if not window_results:
            print(f"  [WARN] No valid windows for {ticker}")
            continue

        all_ticker_results[ticker] = window_results
        _print_ticker_summary(ticker, window_results)

    if all_ticker_results:
        _print_aggregate_summary(all_ticker_results)
        _plot_walk_forward(all_ticker_results)


def _print_ticker_summary(ticker: str, results: list[dict]):
    print(f"\n  {'='*68}")
    print(f"  Walk-Forward Results — {ticker}")
    print(f"  {'='*68}")
    print(f"  {'Window':<28} {'DQN Sharpe':>14} {'BaH Sharpe':>10} {'DQN Ann%':>9} {'BaH Ann%':>9} {'Edge':>7}")
    print(f"  {'-'*68}")

    for r in results:
        dqn_s  = r["dqn"]["sharpe_ratio"]
        dqn_sd = r.get("dqn_std", {}).get("sharpe_ratio", 0.0)
        bah_s  = r["bah"]["sharpe_ratio"]
        dqn_a  = r["dqn"]["annual_return"]
        bah_a  = r["bah"]["annual_return"]
        edge   = dqn_s - bah_s
        sign   = "+" if edge >= 0 else ""
        dqn_str = f"{dqn_s:.3f}±{dqn_sd:.2f}"
        print(
            f"  {r['window']:<28} {dqn_str:>14} {bah_s:>10.3f}"
            f"  {dqn_a:>8.1f}%  {bah_a:>8.1f}%  {sign}{edge:>5.3f}"
        )

    # Aggregate stats across windows — consistency is the key metric
    dqn_sharpes = [r["dqn"]["sharpe_ratio"] for r in results]
    dqn_returns = [r["dqn"]["annual_return"] for r in results]
    wins        = sum(1 for r in results if r["dqn"]["sharpe_ratio"] > r["bah"]["sharpe_ratio"])

    print(f"  {'-'*65}")
    print(f"  {'Mean ± Std':<28} {np.mean(dqn_sharpes):>10.3f}±{np.std(dqn_sharpes):.3f}")
    print(f"  Beat Buy & Hold in {wins}/{len(results)} windows")
    print(f"  Avg annual return : {np.mean(dqn_returns):.1f}%  (std: {np.std(dqn_returns):.1f}%)")

    # DQN + RF filter comparison, when the ensemble ran.
    ens_results = [r for r in results if r.get("ensemble")]
    if ens_results:
        ens_sharpes = [r["ensemble"]["sharpe_ratio"] for r in ens_results]
        ens_wins    = sum(1 for r in ens_results if r["ensemble"]["sharpe_ratio"] > r["dqn"]["sharpe_ratio"])
        print(f"  {'-'*65}")
        print(f"  DQN + RF filter Sharpe (mean): {np.mean(ens_sharpes):>7.3f}  "
              f"(beats raw DQN in {ens_wins}/{len(ens_results)} windows)")
    print(f"  {'='*68}\n")


def _print_aggregate_summary(all_results: dict[str, list[dict]]):
    print(f"\n{'#'*70}")
    print(f"  AGGREGATE WALK-FORWARD SUMMARY — ALL TICKERS")
    print(f"{'#'*70}")
    print(f"  {'Ticker':<8} {'Windows':>8} {'DQN Sharpe':>12} {'Consistency':>13} {'Wins':>6}")
    print(f"  {'-'*55}")

    for ticker, results in all_results.items():
        sharpes     = [r["dqn"]["sharpe_ratio"] for r in results]
        wins        = sum(1 for r in results if r["dqn"]["sharpe_ratio"] > r["bah"]["sharpe_ratio"])
        # Consistency = 1 - coefficient of variation (higher = more consistent)
        cv          = np.std(sharpes) / (abs(np.mean(sharpes)) + 1e-9)
        consistency = max(0.0, 1.0 - cv)
        print(
            f"  {ticker:<8} {len(results):>7}   {np.mean(sharpes):>10.3f}±{np.std(sharpes):.3f}"
            f"   {consistency:>10.1%}   {wins}/{len(results)}"
        )

    print(f"{'#'*70}\n")


def _plot_walk_forward(all_results: dict[str, list[dict]]):
    ensure_output_dir()
    n_tickers = len(all_results)
    if n_tickers == 0:
        return

    fig, axes = plt.subplots(n_tickers, 2, figsize=(16, 4 * n_tickers), squeeze=False)

    for row, (ticker, results) in enumerate(all_results.items()):
        # Left: equity curves per window
        ax = axes[row][0]
        colors = ["#2c7bb6", "#f4a742", "#74c476"]
        for i, r in enumerate(results):
            col = colors[i % len(colors)]
            ax.plot(r["test_index"], r["dqn_equity"],
                    label=f"DQN {r['window']}", color=col, linewidth=1.5)
            ax.plot(r["test_index"], r["bah_equity"],
                    label=f"B&H {r['window']}", color=col, linewidth=1.0, linestyle="--", alpha=0.5)
        ax.set_title(f"{ticker} — Walk-Forward Equity Curves")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend(fontsize=7)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

        # Right: Sharpe comparison per window
        ax2 = axes[row][1]
        window_names   = [r["window"] for r in results]
        dqn_sharpes    = [r["dqn"]["sharpe_ratio"] for r in results]
        bah_sharpes    = [r["bah"]["sharpe_ratio"] for r in results]
        x = np.arange(len(window_names))
        w = 0.35
        ax2.bar(x - w/2, dqn_sharpes, w, label="DQN",        color="#2c7bb6")
        ax2.bar(x + w/2, bah_sharpes, w, label="Buy & Hold", color="#d7191c", alpha=0.7)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_title(f"{ticker} — Sharpe by Window")
        ax2.set_ylabel("Sharpe Ratio")
        ax2.set_xticks(x)
        ax2.set_xticklabels(window_names, rotation=15, ha="right", fontsize=8)
        ax2.legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "walk_forward.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved walk-forward chart → {path}")
