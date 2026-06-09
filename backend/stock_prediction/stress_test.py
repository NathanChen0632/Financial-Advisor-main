import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from stock_prediction.data_collection import download_stock_data
from stock_prediction.features import build_features, get_feature_columns
from stock_prediction.rl_agent import train_dqn_agent
from stock_prediction.backtesting import (
    run_backtest,
    compute_backtest_metrics,
    print_backtest_metrics,
)

OUTPUT_DIR = "results"



# Bearish periods definition
# Each entry: label, description, train window, test (crisis) window, chart colour
BEARISH_PERIODS = [
    {
        "label":       "2008 Financial Crisis",
        "train_start": "2003-01-01",
        "train_end":   "2007-09-30",
        "test_start":  "2007-10-01",
        "test_end":    "2009-03-31",
        "description": "S&P 500 peak-to-trough decline of ~56%",
        "color":       "#d7191c",
    },
    {
        "label":       "2020 COVID Crash",
        "train_start": "2015-01-01",
        "train_end":   "2020-01-31",
        "test_start":  "2020-02-01",
        "test_end":    "2020-09-30",
        "description": "S&P 500 fell ~34% in 33 days; V-shaped recovery",
        "color":       "#fc8d59",
    },
    {
        "label":       "2022 Bear Market",
        "train_start": "2015-01-01",
        "train_end":   "2021-12-31",
        "test_start":  "2022-01-01",
        "test_end":    "2022-12-31",
        "description": "S&P -19%, Nasdaq -33% amid Fed rate-hike cycle",
        "color":       "#756bb1",
    },
]



# Single-period stress test
def _run_single_period(
    ticker: str,
    period: dict,
    n_episodes: int = 50,
) -> dict | None:

    label = period["label"]
    print(f"\n    [{ticker}] ▶  {label}  ({period['description']})")
    print(f"           Train: {period['train_start']} → {period['train_end']}")
    print(f"           Test : {period['test_start']}  → {period['test_end']}")

    # --- Download data -------------------------------------------------------
    try:
        train_raw = download_stock_data(
            ticker, period["train_start"], period["train_end"]
        )
        test_raw = download_stock_data(
            ticker, period["test_start"], period["test_end"]
        )
    except Exception as exc:
        print(f"    [SKIP] Download failed for {ticker}: {exc}")
        return None

    if len(train_raw) < 100:
        print(f"    [SKIP] Only {len(train_raw)} training days — not enough for {ticker}")
        return None
    if len(test_raw) < 20:
        print(f"    [SKIP] Only {len(test_raw)} test days — not enough for {ticker}")
        return None

    # --- Build features on training window -----------------------------------
    try:
        train_feat   = build_features(train_raw)
        feature_cols = get_feature_columns(train_feat)
    except Exception as exc:
        print(f"    [SKIP] Feature engineering failed (train): {exc}")
        return None

    X_train       = train_feat[feature_cols].values
    returns_train = train_feat["daily_return"].values
    prices_train  = train_raw.loc[train_feat.index, "Close"].values.flatten()

    # --- Train DQN on pre-crisis data ----------------------------------------
    print(f"    Training DQN ({n_episodes} episodes) on pre-crisis history…")
    try:
        dqn = train_dqn_agent(
            X_train=X_train,
            daily_returns_train=returns_train,
            prices_train=prices_train,
            feature_cols=feature_cols,
            n_episodes=n_episodes,
        )
    except Exception as exc:
        print(f"    [SKIP] DQN training failed: {exc}")
        return None

    # --- Build features on test (crisis) window ------------------------------
    try:
        test_feat = build_features(test_raw)
    except Exception as exc:
        print(f"    [SKIP] Feature engineering failed (test): {exc}")
        return None

    # Verify the same feature columns are present
    missing = [c for c in feature_cols if c not in test_feat.columns]
    if missing:
        print(f"    [SKIP] Missing feature columns in test data: {missing}")
        return None

    prices_test = test_raw.loc[test_feat.index, "Close"].values.flatten()

    # --- Run backtest --------------------------------------------------------
    try:
        bt_df   = run_backtest(dqn, test_feat, feature_cols, prices=prices_test)
        metrics = compute_backtest_metrics(bt_df)
    except Exception as exc:
        print(f"    [SKIP] Backtest failed: {exc}")
        return None

    return {"bt_df": bt_df, "metrics": metrics, "period": period, "ticker": ticker}


# Plotting helpers
def _plot_period_panel(ax, result: dict, initial_capital: float = 10_000.0):
    bt_df  = result["bt_df"]
    period = result["period"]
    ticker = result["ticker"]

    color  = period.get("color", "#2c7bb6")

    # Normalise both curves to start at 100 for easier cross-period comparison
    strat_norm = bt_df["strategy_equity"] / initial_capital * 100
    bah_norm   = bt_df["bah_equity"]      / initial_capital * 100

    ax.plot(bt_df.index, strat_norm, color=color,    linewidth=1.8, label="DQN Strategy")
    ax.plot(bt_df.index, bah_norm,   color="#555555", linewidth=1.4,
            linestyle="--", alpha=0.8, label="Buy & Hold")

    # Shade area below 100 (loss territory)
    ax.axhline(100, color="black", linewidth=0.7, linestyle=":", alpha=0.5)
    ax.fill_between(bt_df.index, bah_norm, 100,
                    where=(bah_norm < 100),
                    color="#d7191c", alpha=0.12, label="B&H drawdown zone")

    m_strat = result["metrics"]["Strategy"]
    m_bah   = result["metrics"]["Buy & Hold"]

    title = f"{ticker}  ·  {period['label']}"
    subtitle = (
        f"DQN: {m_strat['total_return']:+.1f}%  |  "
        f"B&H: {m_bah['total_return']:+.1f}%  |  "
        f"Max DD  DQN {m_strat['max_drawdown']:.1f}%  B&H {m_bah['max_drawdown']:.1f}%"
    )
    ax.set_title(f"{title}\n{subtitle}", fontsize=8.5, loc="left")
    ax.set_ylabel("Normalised Value (start=100)", fontsize=7)
    ax.legend(fontsize=7, loc="upper left")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.tick_params(axis="both", labelsize=7)


def plot_stress_results(all_results: list, ticker: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    valid = [r for r in all_results if r is not None and r["ticker"] == ticker]
    if not valid:
        print(f"  No valid stress-test results for {ticker} — skipping chart.")
        return

    n = len(valid)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), squeeze=False)
    fig.suptitle(
        f"DQN Bear-Market Stress Test  —  {ticker}",
        fontsize=13, fontweight="bold", y=1.01
    )

    for i, result in enumerate(valid):
        _plot_period_panel(axes[i][0], result)

    plt.tight_layout()
    safe = ticker.replace("/", "_")
    path = os.path.join(OUTPUT_DIR, f"stress_test_{safe}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved stress-test chart → {path}")



# Summary table
def print_stress_summary(all_results: list, ticker: str):
    valid = [r for r in all_results if r is not None and r["ticker"] == ticker]
    if not valid:
        return

    print(f"\n{'='*78}")
    print(f"  STRESS-TEST SUMMARY  —  {ticker}")
    print(f"{'='*78}")
    hdr = f"  {'Period':<28} {'DQN Ret':>9} {'B&H Ret':>9} {'DQN DD':>9} {'B&H DD':>9} {'DQN Sharpe':>11}"
    print(hdr)
    print(f"  {'-'*74}")

    for r in valid:
        lbl  = r["period"]["label"]
        ms   = r["metrics"]["Strategy"]
        mb   = r["metrics"]["Buy & Hold"]
        line = (
            f"  {lbl:<28}"
            f"  {ms['total_return']:>+7.1f}%"
            f"  {mb['total_return']:>+7.1f}%"
            f"  {ms['max_drawdown']:>7.1f}%"
            f"  {mb['max_drawdown']:>7.1f}%"
            f"  {ms['sharpe_ratio']:>10.3f}"
        )
        # Highlight outperformance vs B&H
        beat = ms["total_return"] > mb["total_return"]
        flag = " ✓ outperforms B&H" if beat else ""
        print(line + flag)

    print(f"{'='*78}")
    print("  DD = Max Drawdown  |  ✓ = DQN outperformed Buy & Hold in this crash\n")



# Main entry point
def run_stress_tests(
    tickers: list,
    periods: list = None,
    n_episodes: int = 50,
):
    
    if periods is None:
        periods = BEARISH_PERIODS

    print("\n" + "=" * 60)
    print("  BEAR-MARKET STRESS TEST")
    print("  Testing DQN performance during 3 major downturns:")
    for p in periods:
        print(f"    • {p['label']}: {p['test_start']} → {p['test_end']}")
    print(f"  DQN trained on PRE-CRISIS data only (no lookahead)")
    print("=" * 60)

    all_results = []

    for ticker in tickers:
        print(f"\n{'─'*60}")
        print(f"  TICKER: {ticker}")
        print(f"{'─'*60}")

        ticker_results = []
        for period in periods:
            result = _run_single_period(ticker, period, n_episodes=n_episodes)
            ticker_results.append(result)

            if result is not None:
                print_backtest_metrics(
                    result["metrics"],
                    f"{ticker} — {period['label']}"
                )

        all_results.extend(ticker_results)
        plot_stress_results(ticker_results, ticker)
        print_stress_summary(ticker_results, ticker)

    print("\n  All stress-test charts saved to the 'results/' directory.")
    return all_results
