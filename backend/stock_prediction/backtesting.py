import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stock_prediction.utils import OUTPUT_DIR, ensure_output_dir
from stock_prediction.rl_agent import TradingConfig


def _reconstruct_exposure(signals: np.ndarray, test_df: pd.DataFrame, cfg: TradingConfig) -> np.ndarray:
    # Rebuild the per-day position size the DQN reward optimizes: volatility-based
    # sizing (risk_per_trade / stop-distance) fixed at entry and held for the life
    # of the trade, capped at cfg.max_position_size. Without this the backtest runs
    # flat 1x exposure while the agent was trained/rewarded on sized positions, so
    # the reported P&L wouldn't reflect what the policy actually optimizes.
    if "atr14_pct" not in test_df.columns:
        return signals.astype(float)   # can't size without ATR — fall back to 1x

    atr      = test_df["atr14_pct"].values
    exposure = np.zeros(len(signals), dtype=float)
    size     = 0.0
    for t in range(len(signals)):
        if signals[t] == 1:
            if t == 0 or signals[t - 1] == 0:  # entry bar — size uses ATR at entry
                risk_frac = cfg.stop_atr_mult * atr[t]
                size = min(cfg.risk_per_trade / risk_frac, cfg.max_position_size) if risk_frac > 0 else 1.0
            exposure[t] = size
        else:
            size = 0.0
    return exposure


def run_backtest(
    model,
    test_df: pd.DataFrame,
    feature_cols: list,
    initial_capital: float = 10_000.0,
    prices: np.ndarray | None = None,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
    config: TradingConfig | None = None,
) -> pd.DataFrame:
    X_test = test_df[feature_cols].values
    cfg    = config or TradingConfig()

    # Full disciplined simulation when prices are available (stops, R/R gating, etc.)
    if prices is not None and hasattr(model, "_predict_with_env"):
        signals = model.predict(X_test, prices=prices, feature_cols=feature_cols)
    else:
        signals = model.predict(X_test)

    daily_ret = test_df["daily_return"].values

    # Exposure = volatility-based position size (matches the DQN reward), not a
    # flat 1x signal. Turnover and cost scale with size too, so entering a 3x
    # position incurs 3x the friction — consistent with the training objective.
    exposure         = _reconstruct_exposure(signals, test_df, cfg)
    position_changes = np.abs(np.diff(exposure, prepend=exposure[0] if len(exposure) else 0.0))
    cost_per_turn    = transaction_cost + slippage   # total one-way friction
    execution_drag   = position_changes * cost_per_turn

    strategy_ret    = exposure * daily_ret - execution_drag
    strategy_equity = initial_capital * np.cumprod(1 + strategy_ret)
    bah_equity      = initial_capital * np.cumprod(1 + daily_ret)

    result = pd.DataFrame({
        "close_return":    daily_ret,
        "signal":          signals,
        "exposure":        exposure,
        "strategy_return": strategy_ret,
        "bah_return":      daily_ret,
        "strategy_equity": strategy_equity,
        "bah_equity":      bah_equity,
    }, index=test_df.index)

    return result


def _annualised_return(equity_series: pd.Series, trading_days: int = 252) -> float:
    total = equity_series.iloc[-1] / equity_series.iloc[0]
    n_years = len(equity_series) / trading_days
    return total ** (1 / n_years) - 1


def _sharpe_ratio(daily_returns: pd.Series, trading_days: int = 252) -> float:
    # Assumes risk-free rate ≈ 0 for simplicity.
    # Sharpe measures return per unit of volatility, higher is better.
    if daily_returns.std() == 0:
        return 0.0
    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(trading_days)


def _max_drawdown(equity_series: pd.Series) -> float:
    # Max drawdown is the largest peak-to-trough decline.
    # It captures downside risk that Sharpe ratio alone misses.
    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max
    return drawdown.min()


def compute_backtest_metrics(bt_df: pd.DataFrame, initial_capital: float = 10_000.0) -> dict:
    metrics = {}
    for label, eq_col, ret_col in [
        ("Strategy",   "strategy_equity", "strategy_return"),
        ("Buy & Hold", "bah_equity",      "bah_return"),
    ]:
        eq  = bt_df[eq_col]
        ret = bt_df[ret_col]
        metrics[label] = {
            "final_value":   eq.iloc[-1],
            "total_return":  (eq.iloc[-1] / initial_capital - 1) * 100,
            "annual_return": _annualised_return(eq) * 100,
            "sharpe_ratio":  _sharpe_ratio(ret),
            "max_drawdown":  _max_drawdown(eq) * 100,
        }
    return metrics


def print_backtest_metrics(metrics: dict, model_name: str):
    print(f"\n{'='*60}")
    print(f"  Backtest Results — {model_name}")
    print(f"{'='*60}")
    print(f"  {'Metric':<22} {'Strategy':>12} {'Buy & Hold':>12}")
    print(f"  {'-'*46}")
    keys = [
        ("final_value",   "Final Value ($)",   ",.2f"),
        ("total_return",  "Total Return (%)",  ".2f"),
        ("annual_return", "Annual Return (%)", ".2f"),
        ("sharpe_ratio",  "Sharpe Ratio",       ".3f"),
        ("max_drawdown",  "Max Drawdown (%)",   ".2f"),
    ]
    for key, label, fmt in keys:
        strat = metrics["Strategy"][key]
        bah   = metrics["Buy & Hold"][key]
        print(f"  {label:<22} {strat:>12{fmt}} {bah:>12{fmt}}")
    print(f"{'='*60}")


def plot_equity_curve(bt_df: pd.DataFrame, model_name: str):
    ensure_output_dir()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(bt_df.index, bt_df["strategy_equity"], label="Model Strategy", color="#2c7bb6", linewidth=1.5)
    ax.plot(bt_df.index, bt_df["bah_equity"],      label="Buy & Hold",     color="#d7191c", linewidth=1.5, linestyle="--")
    ax.set_title(f"Equity Curve — {model_name}")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.tight_layout()

    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")
    path = os.path.join(OUTPUT_DIR, f"equity_{safe_name}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved equity curve → {path}")
