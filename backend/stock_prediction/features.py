import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast    = series.ewm(span=fast, adjust=False).mean()
    ema_slow    = series.ewm(span=slow, adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index (0–100 → normalized to 0–1).
    Values above 0.25 indicate a trending market; below indicates chop.
    Uses Wilder's smoothing (EWM with alpha = 1/period) to match the original formula.
    """
    prev_close = close.shift(1)

    # True range
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Raw directional movement — only one of +DM or -DM can be positive per bar
    up_move = high.diff()
    dn_move = -low.diff()
    plus_dm_raw  = np.where((up_move > dn_move) & (up_move > 0), up_move.values, 0.0)
    minus_dm_raw = np.where((dn_move > up_move) & (dn_move > 0), dn_move.values, 0.0)

    alpha = 1.0 / period
    tr_s       = pd.Series(tr.values,         index=close.index).ewm(alpha=alpha, adjust=False).mean()
    plus_dm_s  = pd.Series(plus_dm_raw,        index=close.index).ewm(alpha=alpha, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm_raw,       index=close.index).ewm(alpha=alpha, adjust=False).mean()

    plus_di  = 100 * plus_dm_s  / (tr_s + 1e-10)
    minus_di = 100 * minus_dm_s / (tr_s + 1e-10)

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx / 100.0   # normalized to 0–1


def _rolling_vol_percentile(vol_series: pd.Series, window: int = 252) -> pd.Series:
    """Percentile rank of current volatility within a rolling lookback window.
    0 = historically calm, 1 = historically stressed.
    Computed with raw=True for speed (numpy array per window).
    """
    return vol_series.rolling(window, min_periods=60).apply(
        lambda x: float((x[:-1] < x[-1]).mean()), raw=True
    )


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, spy_df: pd.DataFrame | None = None) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # ------------------------------------------------------------------
    # Price structure
    # ------------------------------------------------------------------

    # Daily return and intraday structure
    feat["daily_return"]    = close.pct_change()
    feat["hl_range"]        = (high - low) / close
    feat["open_close_diff"] = (close - df["Open"]) / close

    # Multi-horizon returns give the agent context on short and medium-term
    # momentum without relying on lagged features that can leak.
    feat["ret_2d"]  = close.pct_change(2)
    feat["ret_3d"]  = close.pct_change(3)
    feat["ret_60d"] = close.pct_change(60)   # ~3-month momentum

    # Return z-score: how extreme is today's move vs recent history?
    # Extreme readings often precede mean-reversion; mild readings continue trending.
    feat["return_zscore"] = feat["daily_return"].rolling(20).apply(
        lambda x: (x[-1] - x[:-1].mean()) / (x[:-1].std() + 1e-10), raw=True
    )

    # ------------------------------------------------------------------
    # Trend / moving averages
    # ------------------------------------------------------------------

    for window in [10, 20, 50]:
        ma = close.rolling(window).mean()
        feat[f"ma{window}_ratio"] = close / ma

    feat["ma10_ma20_cross"] = close.rolling(10).mean() / close.rolling(20).mean()
    feat["ma20_ma50_cross"] = close.rolling(20).mean() / close.rolling(50).mean()
    feat["ma100_ratio"]     = close / close.rolling(100).mean()
    feat["ma200_ratio"]     = close / close.rolling(200).mean()

    # ------------------------------------------------------------------
    # Momentum
    # ------------------------------------------------------------------

    feat["momentum5"]  = close.pct_change(5)
    feat["momentum10"] = close.pct_change(10)
    feat["momentum20"] = close.pct_change(20)

    feat["up_day_ratio10"] = (close.diff() > 0).rolling(10).mean()

    # ------------------------------------------------------------------
    # RSI at multiple horizons — faster RSI is more sensitive to recent moves,
    # slower RSI filters noise; the spread between them is a divergence signal.
    # ------------------------------------------------------------------

    feat["rsi7"]           = _compute_rsi(close, 7)
    feat["rsi14"]          = _compute_rsi(close, 14)
    feat["rsi_divergence"] = feat["rsi7"] - feat["rsi14"]  # leading vs lagging signal

    # ------------------------------------------------------------------
    # MACD
    # ------------------------------------------------------------------

    macd_line, signal_line = _compute_macd(close)
    feat["macd"]        = macd_line
    feat["macd_signal"] = signal_line
    feat["macd_hist"]   = macd_line - signal_line

    # ------------------------------------------------------------------
    # Volatility
    # ------------------------------------------------------------------

    feat["volatility10"] = feat["daily_return"].rolling(10).std()
    feat["volatility20"] = feat["daily_return"].rolling(20).std()

    # ATR — stop sizing depends on this; expressed as % so it's cross-ticker comparable
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    feat["atr14_pct"] = tr.rolling(14).mean() / close

    # Bollinger %B
    bb_mid   = close.rolling(20).mean()
    bb_std   = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    feat["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

    # ------------------------------------------------------------------
    # Regime indicators — tell the model whether we're trending or choppy,
    # calm or stressed. These are typically the highest-value additions
    # because they let the model learn "when not to trade."
    # ------------------------------------------------------------------

    # ADX: trend strength irrespective of direction.
    # High ADX → the current move has momentum behind it (trending regime).
    # Low ADX  → mean-reverting / choppy — breakouts tend to fail here.
    feat["adx14"] = _compute_adx(high, low, close, 14)

    # Volatility regime: percentile rank of current vol vs trailing year.
    # 0 = historically calm (good for trend-following entries),
    # 1 = historically stressed (wider stops needed, often mean-reverting).
    feat["vol_regime"] = _rolling_vol_percentile(feat["volatility20"], window=252)

    # ------------------------------------------------------------------
    # Volume / price-volume relationships
    # ------------------------------------------------------------------

    vol_ma20 = vol.rolling(20).mean()
    feat["volume_ratio"] = vol / vol_ma20

    # OBV (On-Balance Volume) normalized as a z-score.
    # OBV rising ahead of price is a leading bullish signal; falling is bearish.
    price_dir = close.diff().apply(np.sign).fillna(0)
    obv       = (vol * price_dir).cumsum()
    obv_ma    = obv.rolling(20).mean()
    obv_std   = obv.rolling(20).std()
    feat["obv_zscore"] = (obv - obv_ma) / (obv_std + 1e-10)

    # VWAP ratio: where is price relative to the rolling volume-weighted price?
    # Price above VWAP is bullish (buyers in control); below is bearish.
    # Daily VWAP approximated using typical price * volume over a 20-day window.
    typical_price = (high + low + close) / 3
    rolling_vwap  = (typical_price * vol).rolling(20).sum() / (vol.rolling(20).sum() + 1e-10)
    feat["vwap_ratio"] = close / (rolling_vwap + 1e-10)

    # ------------------------------------------------------------------
    # SPY market context
    # ------------------------------------------------------------------

    if spy_df is not None:
        spy_close = spy_df["Close"] if "Close" in spy_df.columns else spy_df.iloc[:, 0]
        spy_close = spy_close.reindex(df.index).ffill()
        feat["spy_return"]        = spy_close.pct_change()
        feat["spy_momentum20"]    = spy_close.pct_change(20)
        feat["relative_strength"] = feat["daily_return"] - feat["spy_return"]

        # Beta-adjusted relative strength: removes market beta to isolate alpha.
        # A positive reading means the stock outperformed even after accounting
        # for how much of the market move it should have captured.
        spy_ret = spy_close.pct_change()
        cov     = feat["daily_return"].rolling(60).cov(spy_ret)
        var     = spy_ret.rolling(60).var()
        beta    = cov / (var + 1e-10)
        feat["alpha_return"] = feat["daily_return"] - beta * spy_ret

    # ------------------------------------------------------------------
    # Supervised target (not used by DQN; classification metrics only)
    # ------------------------------------------------------------------

    feat["Target"] = (close.shift(-1) > close).astype(int)

    feat.dropna(inplace=True)
    return feat


def get_feature_columns(feat_df: pd.DataFrame) -> list:
    return [c for c in feat_df.columns if c not in ("Target", "daily_return")]
