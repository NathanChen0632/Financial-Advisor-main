import numpy as np
import pandas as pd


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def build_features(df: pd.DataFrame, spy_df: pd.DataFrame | None = None) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # Daily return is the primary signal the agent tries to predict.
    # H-L range and open-close diff capture intraday price behavior,
    # wide ranges and closes near the high suggest bullish conviction.
    feat["daily_return"]    = close.pct_change()
    feat["hl_range"]        = (high - low) / close
    feat["open_close_diff"] = (close - df["Open"]) / close

    # MA ratios express price as a position relative to its own trend.
    # Ratios above 1 mean price is above the moving average (bullish bias).
    # Using ratios instead of raw MA values makes the feature scale-free
    # so it generalizes across different price levels and time periods.
    for window in [10, 20, 50]:
        ma = close.rolling(window).mean()
        feat[f"ma{window}_ratio"] = close / ma

    # MA crossovers signal trend momentum shifts.
    # When the short MA crosses above the long MA, trend is strengthening.
    feat["ma10_ma20_cross"] = close.rolling(10).mean() / close.rolling(20).mean()
    feat["ma20_ma50_cross"] = close.rolling(20).mean() / close.rolling(50).mean()

    # RSI measures momentum by comparing recent gains to recent losses.
    # Readings below 30 suggest oversold (potential reversal up),
    # above 70 suggest overbought (potential reversal down).
    feat["rsi14"] = _compute_rsi(close, 14)

    # MACD captures the relationship between two trend-following EMAs.
    # The histogram (MACD - signal) is particularly useful, crossing zero
    # often precedes a momentum shift before price visibly confirms it.
    macd_line, signal_line = _compute_macd(close)
    feat["macd"]        = macd_line
    feat["macd_signal"] = signal_line
    feat["macd_hist"]   = macd_line - signal_line

    # Long-term trend context. Being above the 200-day MA is widely used
    # by institutional traders as a bull/bear regime filter.
    feat["ma100_ratio"] = close / close.rolling(100).mean()
    feat["ma200_ratio"] = close / close.rolling(200).mean()

    # 20-day price momentum tells the agent whether we're in a trending
    # move or a range-bound environment, which affects trade validity.
    feat["momentum20"] = close.pct_change(20)

    # Counts how consistently the stock is closing up over recent sessions.
    # High values indicate sustained buying pressure, not just a one-day spike.
    feat["up_day_ratio10"] = (close.diff() > 0).rolling(10).mean()

    # Rolling volatility over two windows. Higher volatility means wider
    # stops are needed and position sizes should be smaller, the agent
    # learns to factor this in through ATR-based reward shaping.
    feat["volatility10"] = feat["daily_return"].rolling(10).std()
    feat["volatility20"] = feat["daily_return"].rolling(20).std()

    # Volume relative to its 20-day average measures conviction.
    # A breakout on 2x normal volume is far more reliable than one on
    # 0.5x volume, the trading environment uses this for entry confirmation.
    vol_ma20 = vol.rolling(20).mean()
    feat["volume_ratio"] = vol / vol_ma20

    # ATR measures true price range accounting for overnight gaps.
    # Expressed as a percentage of close so it's comparable across tickers.
    # The trading environment uses ATR to dynamically size stop losses,
    # volatile stocks get wider stops so normal noise doesn't trigger them.
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    feat["atr14_pct"] = tr.rolling(14).mean() / close

    # Short-term momentum over 5 and 10 days captures near-term directional
    # bias without the smoothing lag of longer-period indicators.
    feat["momentum5"]  = close.pct_change(5)
    feat["momentum10"] = close.pct_change(10)

    # Bollinger %B places current price within the recent volatility envelope.
    # 0 = at the lower band, 1 = at the upper band, values outside [0,1]
    # indicate the price is beyond the bands, often a mean-reversion signal.
    bb_mid   = close.rolling(20).mean()
    bb_std   = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    feat["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

    # SPY features provide broad market context so the agent can distinguish
    # between a stock falling on its own fundamentals vs. a market-wide selloff.
    # relative_strength isolates the stock's alpha, positive means outperforming
    # the market, negative means underperforming regardless of market direction.
    if spy_df is not None:
        spy_close = spy_df["Close"] if "Close" in spy_df.columns else spy_df.iloc[:, 0]
        spy_close = spy_close.reindex(df.index).ffill()
        feat["spy_return"]        = spy_close.pct_change()
        feat["spy_momentum20"]    = spy_close.pct_change(20)
        feat["relative_strength"] = feat["daily_return"] - feat["spy_return"]

    # Supervised target: 1 if price closes higher tomorrow, 0 if lower.
    # Used only during training, the agent optimizes reward, not this label,
    # but classification metrics use it to benchmark prediction accuracy.
    feat["Target"] = (close.shift(-1) > close).astype(int)

    feat.dropna(inplace=True)
    return feat


def get_feature_columns(feat_df: pd.DataFrame) -> list:
    return [c for c in feat_df.columns if c != "Target"]
