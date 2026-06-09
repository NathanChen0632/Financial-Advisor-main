from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from stock_prediction.utils import OUTPUT_DIR, ensure_output_dir
from stock_prediction.data_collection import download_stock_data, download_spy_context
from stock_prediction.features import build_features, get_feature_columns
from stock_prediction.models import chronological_split, fit_feature_scaler
from stock_prediction.backtesting import run_backtest, compute_backtest_metrics
from stock_prediction.rl_agent import train_dqn_agent


def _run_ticker_pipeline(
    ticker: str,
    spy_df: pd.DataFrame | None,
    n_episodes: int = 150,
) -> dict | None:
    try:
        print(f"\n  [{ticker}] Downloading data...")
        df = download_stock_data(ticker)

        feat_df      = build_features(df, spy_df=spy_df)
        feature_cols = get_feature_columns(feat_df)

        splits        = chronological_split(feat_df, feature_cols)
        n_train_full  = len(splits["X_train"]) + len(splits["X_val"])
        X_train_full  = np.concatenate([splits["X_train"], splits["X_val"]])
        returns_train = feat_df["daily_return"].values[:n_train_full]

        scaler         = fit_feature_scaler(X_train_full)
        X_train_scaled = scaler.transform(X_train_full)

        prices_all   = df.loc[feat_df.index, "Close"].values.flatten()
        prices_train = prices_all[:n_train_full]

        print(f"  [{ticker}] Training DQN ({n_episodes} episodes)...")
        dqn = train_dqn_agent(
            X_train=X_train_full,
            daily_returns_train=returns_train,
            prices_train=prices_train,
            feature_cols=feature_cols,
            n_episodes=n_episodes,
            X_train_scaled=X_train_scaled,
        )
        dqn.scaler = scaler

        prices_test_full = prices_all[n_train_full:]
        bt_df = run_backtest(
            dqn,
            splits["test_df"],
            feature_cols,
            prices=prices_test_full[:len(splits["test_df"])],
        )
        metrics = compute_backtest_metrics(bt_df)

        # A ticker where the agent never entered a trade produces a flat
        # equity curve and a zero-variance return series, useless for
        # covariance estimation and meaningless in the portfolio.
        if bt_df["signal"].sum() == 0:
            print(f"  [{ticker}] Agent stayed in cash all period — skipping.")
            return None

        return {
            "ticker":            ticker,
            "sharpe":            metrics["Strategy"]["sharpe_ratio"],
            "annual_return":     metrics["Strategy"]["annual_return"],
            "total_return":      metrics["Strategy"]["total_return"],
            "max_drawdown":      metrics["Strategy"]["max_drawdown"],
            "bah_annual_return": metrics["Buy & Hold"]["annual_return"],
            # Strategy returns (not raw price returns) are passed to the optimizer.
            # This means Markowitz minimizes correlation between trading strategies,
            # not just between stocks — diversification at the strategy level.
            "strategy_returns":  bt_df["strategy_return"],
            "bah_returns":       bt_df["bah_return"],
        }

    except Exception as e:
        print(f"  [{ticker}] Failed: {e}")
        return None


def optimize_weights(strategy_returns_df: pd.DataFrame) -> np.ndarray:
    # Markowitz mean-variance optimization: find the weight vector that
    # maximizes Sharpe ratio subject to long-only and full-investment constraints.
    # Long-only (no short selling) keeps the result practically implementable.
    n        = strategy_returns_df.shape[1]
    mean_ret = strategy_returns_df.mean().values * 252
    cov_mat  = strategy_returns_df.cov().values  * 252

    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = float(w @ mean_ret)
        port_vol = float(np.sqrt(w @ cov_mat @ w))
        return -(port_ret / port_vol) if port_vol > 1e-9 else 0.0

    result = minimize(
        neg_sharpe,
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    # Clip tiny negatives from numerical noise, then renormalize
    weights = np.clip(result.x, 0.0, 1.0)
    weights /= weights.sum()
    return weights


def _simulate_portfolio(
    weights: np.ndarray,
    returns_df: pd.DataFrame,
    budget: float,
) -> pd.Series:
    # Weighted combination of daily strategy returns, compounded over time
    daily  = returns_df.values @ weights
    equity = budget * np.cumprod(1.0 + daily)
    return pd.Series(equity, index=returns_df.index)


def _portfolio_metrics(equity: pd.Series, daily_returns: np.ndarray) -> dict:
    n_years   = len(equity) / 252
    total_ret = (equity.iloc[-1] / equity.iloc[0] - 1.0) * 100
    annual    = ((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / n_years) - 1.0) * 100
    sharpe    = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)
                 if daily_returns.std() > 0 else 0.0)
    roll_max  = equity.cummax()
    max_dd    = ((equity - roll_max) / roll_max).min() * 100
    return {
        "total_return":  total_ret,
        "annual_return": annual,
        "sharpe_ratio":  sharpe,
        "max_drawdown":  max_dd,
    }


def _print_report(
    weights: np.ndarray,
    tickers: list,
    per_ticker: list[dict],
    portfolio_m: dict,
    eq_weight_m: dict,
    budget: float,
):
    ticker_map = {r["ticker"]: r for r in per_ticker}

    print(f"\n{'='*68}")
    print(f"  PORTFOLIO ALLOCATION  —  ${budget:,.0f} budget")
    print(f"{'='*68}")
    print(f"  {'Ticker':<8} {'Weight':>7} {'Allocation':>13} "
          f"{'Sharpe':>8} {'Annual%':>9} {'MaxDD%':>8}")
    print(f"  {'-'*63}")

    for ticker, w in zip(tickers, weights):
        if w < 0.005:
            continue
        m = ticker_map[ticker]
        print(
            f"  {ticker:<8} {w*100:>6.1f}%  ${w*budget:>11,.0f}"
            f"  {m['sharpe']:>8.2f}  {m['annual_return']:>8.1f}%"
            f"  {m['max_drawdown']:>7.1f}%"
        )

    print(f"\n  {'─'*63}")
    print(f"  Optimised Portfolio:")
    print(f"    Sharpe Ratio  : {portfolio_m['sharpe_ratio']:.3f}")
    print(f"    Annual Return : {portfolio_m['annual_return']:.1f}%")
    print(f"    Total Return  : {portfolio_m['total_return']:.1f}%")
    print(f"    Max Drawdown  : {portfolio_m['max_drawdown']:.1f}%")
    print(f"\n  Equal-Weight Baseline (same stocks):")
    print(f"    Sharpe Ratio  : {eq_weight_m['sharpe_ratio']:.3f}")
    print(f"    Annual Return : {eq_weight_m['annual_return']:.1f}%")
    print(f"    Total Return  : {eq_weight_m['total_return']:.1f}%")
    print(f"{'='*68}\n")


def _plot_results(
    portfolio_equity: pd.Series,
    eq_equity: pd.Series,
    spy_equity: pd.Series | None,
    weights: np.ndarray,
    tickers: list,
    budget: float,
):
    ensure_output_dir()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(portfolio_equity.index, portfolio_equity.values,
            label="Optimised Portfolio", color="#2c7bb6", linewidth=2)
    ax.plot(eq_equity.index, eq_equity.values,
            label="Equal Weight", color="#f4a742", linewidth=1.5, linestyle="--")
    if spy_equity is not None:
        ax.plot(spy_equity.index, spy_equity.values,
                label="SPY Buy & Hold", color="#d7191c", linewidth=1.5, linestyle=":")
    ax.set_title("Portfolio Equity Curve")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    # Pie chart shows final capital allocation across selected tickers.
    # Allocations below 0.5% are hidden to keep the chart readable.
    ax2 = axes[1]
    display_pairs = [(t, w) for t, w in zip(tickers, weights) if w >= 0.005]
    labels = [t for t, _ in display_pairs]
    sizes  = [w for _, w in display_pairs]
    colors = plt.cm.tab10.colors[:len(labels)]
    wedges, texts, autotexts = ax2.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        startangle=90, colors=colors,
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax2.set_title(f"Allocation  (${budget:,.0f})")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "portfolio_optimised.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved portfolio chart → {path}")


def run_portfolio_builder(
    tickers: list,
    budget: float = 10_000.0,
    top_n: int = 5,
    n_episodes: int = 150,
):
    print(f"\n{'#'*65}")
    print(f"  PORTFOLIO BUILDER")
    print(f"  Universe : {tickers}")
    print(f"  Budget   : ${budget:,.0f}  |  Top-N : {top_n}  |  Episodes : {n_episodes}")
    print(f"{'#'*65}")

    # SPY is downloaded once and shared across all tickers as a market context feature.
    # This is more efficient than re-downloading it for each ticker individually.
    print("\n[1/4] Downloading SPY market context...")
    spy_df = download_spy_context()

    # Stage 1: Train a DQN for each ticker and collect strategy performance metrics.
    # Tickers are ranked by Sharpe ratio, this filters for stocks where the agent
    # found a profitable strategy, not just the strongest raw price performers.
    print(f"\n[2/4] Screening {len(tickers)} tickers with DQN "
          f"({n_episodes} episodes each)...")

    all_results = []
    for ticker in tickers:
        if ticker.upper() == "SPY":
            continue
        result = _run_ticker_pipeline(ticker.upper(), spy_df, n_episodes)
        if result is not None:
            all_results.append(result)
            print(
                f"  ✓ [{result['ticker']}]  "
                f"Sharpe={result['sharpe']:+.3f}  "
                f"Annual={result['annual_return']:+.1f}%  "
                f"MaxDD={result['max_drawdown']:.1f}%"
            )

    if len(all_results) < 2:
        print("\n[ERROR] Fewer than 2 tickers passed screening. Widen the universe.")
        return

    all_results.sort(key=lambda r: r["sharpe"], reverse=True)
    selected    = all_results[:top_n]
    sel_tickers = [r["ticker"] for r in selected]

    print(f"\n  Top {len(selected)} selected: {sel_tickers}")
    print(f"  Eliminated : {[r['ticker'] for r in all_results[top_n:]]}")

    # Stage 2: Align strategy returns to shared trading days, then run Markowitz.
    # Date alignment is critical, tickers with different trading calendars
    # must share the same date index for the covariance matrix to be valid.
    print("\n[3/4] Optimising weights (Markowitz max-Sharpe)...")

    returns_df = pd.concat(
        [r["strategy_returns"].rename(r["ticker"]) for r in selected],
        axis=1,
    ).dropna()

    if returns_df.shape[0] < 30:
        print("[ERROR] Too few overlapping trading days after date alignment.")
        return

    print(f"  Aligned {returns_df.shape[0]} overlapping trading days "
          f"({returns_df.index[0].date()} – {returns_df.index[-1].date()})")

    weights    = optimize_weights(returns_df)
    eq_weights = np.ones(len(selected)) / len(selected)

    portfolio_equity = _simulate_portfolio(weights,    returns_df, budget)
    eq_equity        = _simulate_portfolio(eq_weights, returns_df, budget)

    portfolio_daily = returns_df.values @ weights
    eq_daily        = returns_df.values @ eq_weights
    portfolio_m     = _portfolio_metrics(portfolio_equity, portfolio_daily)
    eq_weight_m     = _portfolio_metrics(eq_equity,        eq_daily)

    # SPY benchmark over the same date window provides a market-wide reference point
    spy_equity = None
    try:
        spy_raw    = download_stock_data(
            "SPY",
            start=str(returns_df.index[0].date()),
            end=str((returns_df.index[-1] + pd.Timedelta(days=1)).date()),
        )
        spy_ret    = spy_raw["Close"].pct_change().dropna()
        spy_ret    = spy_ret.reindex(returns_df.index).ffill().fillna(0)
        spy_equity = pd.Series(
            budget * np.cumprod(1.0 + spy_ret.values),
            index=returns_df.index,
        )
    except Exception:
        pass

    print("\n[4/4] Generating report and charts...")
    _print_report(weights, sel_tickers, selected, portfolio_m, eq_weight_m, budget)
    _plot_results(portfolio_equity, eq_equity, spy_equity,
                  weights, sel_tickers, budget)
