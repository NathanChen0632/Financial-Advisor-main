from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stock_prediction.utils import OUTPUT_DIR, ensure_output_dir


# ---------------------------------------------------------------------------
# Rule-based strategies
# Each strategy takes a feature DataFrame and returns a signal array (0 or 1)
# These are deliberately simple — if DQN can't beat them, it isn't learning
# anything beyond what a basic rule already captures.
# ---------------------------------------------------------------------------

def ma_crossover_signals(feat_df: pd.DataFrame) -> np.ndarray:
    # Classic trend-following rule: hold when short MA is above long MA.
    # Uses the same MA ratio features already computed in features.py.
    # Buy signal when ma10 crosses above ma50, sell when it crosses below.
    ma10 = feat_df["ma10_ratio"].values if "ma10_ratio" in feat_df.columns else None
    ma50 = feat_df["ma50_ratio"].values if "ma50_ratio" in feat_df.columns else None

    if ma10 is None or ma50 is None:
        return np.zeros(len(feat_df), dtype=int)

    # Both ratios are price/MA — dividing them gives MA10/MA50 directly
    cross = ma10 / (ma50 + 1e-10)
    return (cross > 1.0).astype(int)


def rsi_signals(feat_df: pd.DataFrame, oversold: float = 35, overbought: float = 65) -> np.ndarray:
    # Mean-reversion rule: buy when RSI is oversold, sell when overbought.
    # Thresholds are slightly wider than the classic 30/70 to generate
    # more signals on the test period for a fairer comparison.
    if "rsi14" not in feat_df.columns:
        return np.zeros(len(feat_df), dtype=int)

    rsi      = feat_df["rsi14"].values
    signals  = np.zeros(len(rsi), dtype=int)
    position = 0

    for i, r in enumerate(rsi):
        if position == 0 and r < oversold:
            position = 1
        elif position == 1 and r > overbought:
            position = 0
        signals[i] = position

    return signals


def momentum_signals(feat_df: pd.DataFrame) -> np.ndarray:
    # Trend-continuation rule: hold when 20-day momentum is positive.
    # Simple but effective in trending markets — tests whether DQN adds
    # value beyond just riding the prevailing trend direction.
    if "momentum20" not in feat_df.columns:
        return np.zeros(len(feat_df), dtype=int)

    return (feat_df["momentum20"].values > 0).astype(int)


def combined_rule_signals(feat_df: pd.DataFrame) -> np.ndarray:
    # Majority vote across all three rule-based strategies.
    # Represents the best a pure rule-based system can do with the
    # same information available to the DQN.
    ma_sig  = ma_crossover_signals(feat_df)
    rsi_sig = rsi_signals(feat_df)
    mom_sig = momentum_signals(feat_df)
    votes   = ma_sig + rsi_sig + mom_sig
    return (votes >= 2).astype(int)


# ---------------------------------------------------------------------------
# Equity curve and metrics
# ---------------------------------------------------------------------------

def _equity_curve(signals: np.ndarray, daily_returns: np.ndarray, initial: float = 10_000.0) -> np.ndarray:
    strategy_ret = signals * daily_returns
    return initial * np.cumprod(1 + strategy_ret)


def _sharpe(daily_returns: np.ndarray) -> float:
    if daily_returns.std() == 0:
        return 0.0
    return float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252))


def _max_drawdown(equity: np.ndarray) -> float:
    roll_max = np.maximum.accumulate(equity)
    dd = (equity - roll_max) / roll_max
    return float(dd.min() * 100)


def _total_return(equity: np.ndarray, initial: float = 10_000.0) -> float:
    return float((equity[-1] / initial - 1) * 100)


def _annual_return(equity: np.ndarray, initial: float = 10_000.0) -> float:
    n_years = len(equity) / 252
    return float(((equity[-1] / initial) ** (1 / n_years) - 1) * 100)


def _metrics(signals, daily_returns, initial=10_000.0) -> dict:
    eq  = _equity_curve(signals, daily_returns, initial)
    ret = signals * daily_returns
    return {
        "total_return":  _total_return(eq, initial),
        "annual_return": _annual_return(eq, initial),
        "sharpe_ratio":  _sharpe(ret),
        "max_drawdown":  _max_drawdown(eq),
        "n_trades":      int((np.diff(np.concatenate([[0], signals])) > 0).sum()),
        "equity":        eq,
    }


# ---------------------------------------------------------------------------
# Main comparison runner
# ---------------------------------------------------------------------------

def run_benchmark_comparison(
    ticker:       str,
    dqn_model,
    test_df:      pd.DataFrame,
    feature_cols: list,
    prices:       np.ndarray,
    initial:      float = 10_000.0,
) -> dict:
    ensure_output_dir()

    daily_returns = test_df["daily_return"].values
    X_test        = test_df[feature_cols].values

    # DQN signals use the full trading environment (stops, R/R gating, etc.)
    dqn_signals = dqn_model.predict(X_test, prices=prices, feature_cols=feature_cols)

    # Rule-based signals operate only on the feature values — no learned policy
    strategies = {
        "DQN (RL Agent)":      dqn_signals,
        "MA Crossover":        ma_crossover_signals(test_df),
        "RSI (35/65)":         rsi_signals(test_df),
        "Momentum (20-day)":   momentum_signals(test_df),
        "Combined Rules":      combined_rule_signals(test_df),
        "Buy & Hold":          np.ones(len(test_df), dtype=int),
    }

    results = {}
    for name, signals in strategies.items():
        results[name] = _metrics(signals, daily_returns, initial)

    _print_comparison(ticker, results)
    _plot_comparison(ticker, results, test_df.index)

    return results


def _print_comparison(ticker: str, results: dict):
    print(f"\n{'='*75}")
    print(f"  STRATEGY COMPARISON — {ticker}")
    print(f"{'='*75}")
    print(f"  {'Strategy':<22} {'Total%':>8} {'Annual%':>9} {'Sharpe':>8} {'MaxDD%':>8} {'Trades':>7}")
    print(f"  {'-'*70}")

    # Sort by Sharpe ratio, DQN first regardless
    order = ["DQN (RL Agent)", "Combined Rules", "MA Crossover",
             "RSI (35/65)", "Momentum (20-day)", "Buy & Hold"]

    for name in order:
        if name not in results:
            continue
        m = results[name]
        marker = " ◀" if name == "DQN (RL Agent)" else ""
        print(
            f"  {name:<22} {m['total_return']:>7.1f}%  {m['annual_return']:>8.1f}%"
            f"  {m['sharpe_ratio']:>7.3f}  {m['max_drawdown']:>7.1f}%  {m['n_trades']:>6}{marker}"
        )

    print(f"{'='*75}")

    # Show DQN edge over each rule-based strategy
    dqn = results.get("DQN (RL Agent)")
    if dqn:
        print(f"\n  DQN edge over rule-based strategies (Sharpe):")
        for name, m in results.items():
            if name in ("DQN (RL Agent)", "Buy & Hold"):
                continue
            edge = dqn["sharpe_ratio"] - m["sharpe_ratio"]
            sign = "+" if edge >= 0 else ""
            print(f"    vs {name:<22} {sign}{edge:.3f}")
        print()


def _plot_comparison(ticker: str, results: dict, index: pd.Index):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Equity curves
    ax = axes[0]
    colors = {
        "DQN (RL Agent)":    "#2c7bb6",
        "Buy & Hold":        "#d7191c",
        "MA Crossover":      "#f4a742",
        "RSI (35/65)":       "#74c476",
        "Momentum (20-day)": "#9e9ac8",
        "Combined Rules":    "#fd8d3c",
    }
    styles = {
        "DQN (RL Agent)": (2.5, "-"),
        "Buy & Hold":     (1.5, "--"),
        "MA Crossover":   (1.2, "-."),
        "RSI (35/65)":    (1.2, ":"),
        "Momentum (20-day)": (1.2, "-."),
        "Combined Rules": (1.2, ":"),
    }
    for name, m in results.items():
        lw, ls = styles.get(name, (1.0, "-"))
        ax.plot(index, m["equity"], label=name,
                color=colors.get(name, "grey"), linewidth=lw, linestyle=ls)

    ax.set_title(f"Strategy Comparison — {ticker}")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    # Sharpe ratio bar chart
    ax2 = axes[1]
    names  = list(results.keys())
    sharpes = [results[n]["sharpe_ratio"] for n in names]
    bar_colors = [colors.get(n, "grey") for n in names]
    bars = ax2.bar(names, sharpes, color=bar_colors)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Sharpe Ratio Comparison")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    for bar, val in zip(bars, sharpes):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    safe = ticker.replace(" ", "_")
    path = os.path.join(OUTPUT_DIR, f"benchmark_{safe}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved benchmark chart → {path}")
