from __future__ import annotations

# Claude API research agent — screens the broad stock universe using technical
# metrics, then asks Claude to identify the most promising trading candidates.
# This splits the workload: yfinance handles data, Claude handles stock selection.

import os
import json
import textwrap
from datetime import datetime, timedelta

from stock_prediction.utils import GREEN, RED, YELLOW, BOLD, RESET

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

import numpy as np
import pandas as pd
import yfinance as yf

from stock_prediction.data_collection import BROAD_UNIVERSE, PORTFOLIO_UNIVERSE


# Combined universe — portfolio candidates + broad validation universe
RESEARCH_UNIVERSE = list(dict.fromkeys(PORTFOLIO_UNIVERSE + BROAD_UNIVERSE))


def _compute_metrics(ticker: str, df: pd.DataFrame, spy_df: pd.DataFrame) -> dict | None:
    # Compute momentum, volume, RSI, relative strength, and 52w position.
    # Returns None if insufficient data.
    if len(df) < 60:
        return None

    close  = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    # Momentum — percentage return over the last 20 and 60 trading days
    mom_20  = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 21 else None
    mom_60  = (close.iloc[-1] / close.iloc[-61] - 1) * 100 if len(close) > 61 else None

    # Volume ratio — recent average vs 20-day baseline (>1 = elevated activity)
    vol_ratio = float(volume.iloc[-5:].mean() / volume.iloc[-20:].mean()) if len(volume) >= 20 else None

    # RSI 14 — standard overbought/oversold oscillator
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = float(100 - 100 / (1 + rs.iloc[-1])) if not rs.iloc[-1:].isna().all() else None

    # Relative strength — stock 20d return minus SPY 20d return
    rel_strength = None
    if spy_df is not None and len(spy_df) > 21:
        spy_close = spy_df["Close"].squeeze()
        spy_ret   = (spy_close.iloc[-1] / spy_close.iloc[-21] - 1) * 100
        rel_strength = mom_20 - spy_ret if mom_20 is not None else None

    # Position within the 52-week range — 100% = at high, 0% = at low
    high_52 = close.iloc[-252:].max() if len(close) >= 252 else close.max()
    low_52  = close.iloc[-252:].min() if len(close) >= 252 else close.min()
    rng     = high_52 - low_52
    pos_52w = float((close.iloc[-1] - low_52) / rng * 100) if rng > 0 else None

    # ATR 14 — average true range as a % of price (volatility proxy)
    high = df["High"].squeeze()
    low  = df["Low"].squeeze()
    tr   = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_pct = float(tr.rolling(14).mean().iloc[-1] / close.iloc[-1] * 100)

    return {
        "ticker":       ticker,
        "price":        round(float(close.iloc[-1]), 2),
        "mom_20d":      round(mom_20,      1) if mom_20      is not None else None,
        "mom_60d":      round(mom_60,      1) if mom_60      is not None else None,
        "vol_ratio":    round(vol_ratio,   2) if vol_ratio   is not None else None,
        "rsi_14":       round(rsi,         1) if rsi         is not None else None,
        "rel_strength": round(rel_strength,1) if rel_strength is not None else None,
        "pos_52w_pct":  round(pos_52w,     1) if pos_52w     is not None else None,
        "atr_pct":      round(atr_pct,     2),
    }


def screen_universe(
    universe: list[str] = None,
    lookback_days: int = 300,
) -> list[dict]:
    # Download price data for every ticker in the universe and compute
    # screening metrics. Returns a list of metric dicts, skipping failures.

    universe = universe or RESEARCH_UNIVERSE
    end      = datetime.today()
    start    = end - timedelta(days=lookback_days)

    print(f"\n{BOLD}[RESEARCH] Screening {len(universe)} stocks...{RESET}")

    # Download SPY for relative-strength calculation
    spy_raw = yf.download("SPY", start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
    spy_df  = spy_raw[["Close"]] if not spy_raw.empty else None

    results = []
    for ticker in universe:
        try:
            raw = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
            )
            if raw.empty or len(raw) < 40:
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            metrics = _compute_metrics(ticker, raw, spy_df)
            if metrics:
                results.append(metrics)
                print(f"  {ticker:<6}  price=${metrics['price']:.2f}  "
                      f"mom20={metrics['mom_20d']:+.1f}%  "
                      f"RSI={metrics['rsi_14']:.0f}  "
                      f"rel={metrics['rel_strength']:+.1f}%" if metrics['rel_strength'] else
                      f"  {ticker:<6}  price=${metrics['price']:.2f}  "
                      f"mom20={metrics['mom_20d']:+.1f}%  "
                      f"RSI={metrics['rsi_14']:.0f}")
        except Exception as e:
            print(f"  {YELLOW}[RESEARCH] Skipped {ticker}: {e}{RESET}")

    print(f"  [RESEARCH] Screened {len(results)}/{len(universe)} tickers successfully")
    return results


def _build_prompt(metrics: list[dict]) -> str:
    # Format the screening data into a concise table for Claude to reason over.
    today = datetime.today().strftime("%Y-%m-%d")

    header = (
        f"Today is {today}. You are a quantitative analyst reviewing technical screening "
        f"metrics for {len(metrics)} stocks. Your job is to identify the 5-8 most "
        f"promising candidates for a short-to-medium-term momentum strategy (1–4 week holds).\n\n"
        f"Screening metrics:\n"
        f"{'Ticker':<7} {'Price':>7} {'Mom20':>7} {'Mom60':>7} {'RSI':>5} "
        f"{'RelStr':>7} {'52wPos':>7} {'VolRatio':>9} {'ATR%':>5}\n"
        f"{'-'*70}\n"
    )

    rows = []
    for m in sorted(metrics, key=lambda x: x.get("rel_strength") or -999, reverse=True):
        rows.append(
            f"{m['ticker']:<7} "
            f"{m['price']:>7.2f} "
            f"{(str(m['mom_20d'])+' %') if m['mom_20d'] is not None else 'N/A':>7} "
            f"{(str(m['mom_60d'])+' %') if m['mom_60d'] is not None else 'N/A':>7} "
            f"{m['rsi_14'] if m['rsi_14'] is not None else 'N/A':>5} "
            f"{(str(m['rel_strength'])+' %') if m['rel_strength'] is not None else 'N/A':>7} "
            f"{(str(m['pos_52w_pct'])+'%') if m['pos_52w_pct'] is not None else 'N/A':>7} "
            f"{m['vol_ratio'] if m['vol_ratio'] is not None else 'N/A':>9} "
            f"{m['atr_pct']:>5.2f}"
        )

    footer = textwrap.dedent("""

        Column definitions:
        - Mom20/60: price momentum over 20 and 60 trading days
        - RSI: 14-day RSI (30=oversold, 70=overbought)
        - RelStr: stock 20d return minus SPY 20d return (outperformance)
        - 52wPos: position within 52-week high/low range (100%=at high)
        - VolRatio: 5d avg volume / 20d avg volume (>1 = elevated)
        - ATR%: 14-day average true range as % of price (volatility)

        Select 5-8 tickers that show:
        1. Positive relative strength (outperforming SPY)
        2. RSI between 40 and 68 (momentum without being overbought)
        3. Elevated volume (VolRatio > 1.0 preferred)
        4. Reasonable ATR% (not so volatile it's uncontrollable)
        5. Positive 20-day and 60-day momentum

        Respond ONLY with valid JSON in this exact format:
        {
          "tickers": ["AAPL", "MSFT", ...],
          "reasoning": "Brief explanation of why these were selected."
        }
    """)

    return header + "\n".join(rows) + footer


def ask_claude(metrics: list[dict], model: str = "claude-opus-4-8") -> tuple[list[str], str]:
    # Send the screening table to Claude and parse its ticker recommendations.
    # Returns (tickers, reasoning). Falls back to rule-based selection if API fails.

    if not _ANTHROPIC_AVAILABLE:
        raise ImportError("anthropic not installed. Run: pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in environment or .env file")

    client = anthropic.Anthropic(api_key=api_key)
    prompt = _build_prompt(metrics)

    print(f"\n{BOLD}[RESEARCH] Asking Claude to identify top candidates...{RESET}")

    message = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()

    # Strip markdown code fences if Claude wraps the JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed    = json.loads(raw)
        tickers   = [t.upper().strip() for t in parsed.get("tickers", [])]
        reasoning = parsed.get("reasoning", "")
        return tickers, reasoning
    except json.JSONDecodeError:
        print(f"  {YELLOW}[RESEARCH] Could not parse Claude response — falling back to rule-based selection.{RESET}")
        return _rule_based_fallback(metrics), "Rule-based fallback (JSON parse failed)"


def _rule_based_fallback(metrics: list[dict], top_n: int = 7) -> list[str]:
    # Score-based selection used when Claude API is unavailable or fails.
    # Ranks stocks by a composite of relative strength, momentum, and RSI quality.
    scored = []
    for m in metrics:
        score = 0.0
        if m.get("rel_strength") is not None:
            score += m["rel_strength"] * 2.0     # outperformance is the strongest signal
        if m.get("mom_20d") is not None:
            score += m["mom_20d"] * 1.0
        if m.get("rsi_14") is not None:
            rsi = m["rsi_14"]
            # Prefer RSI in the sweet spot (45–65), penalise extremes
            if 45 <= rsi <= 65:
                score += 10
            elif rsi > 70 or rsi < 35:
                score -= 10
        if m.get("vol_ratio") is not None and m["vol_ratio"] > 1.1:
            score += 5                            # elevated volume confirms momentum
        scored.append((m["ticker"], score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in scored[:top_n]]


def run_research_agent(
    universe:  list[str] | None = None,
    use_claude: bool = True,
) -> list[str]:
    # Main entry point. Downloads data, computes metrics, asks Claude (or falls
    # back to rule-based scoring), and returns a ranked list of ticker symbols.

    metrics = screen_universe(universe)

    if not metrics:
        print(f"  {RED}[RESEARCH] No metrics computed — returning default tickers.{RESET}")
        from stock_prediction.data_collection import TICKERS
        return TICKERS

    tickers, reasoning = [], ""

    if use_claude:
        try:
            tickers, reasoning = ask_claude(metrics)
        except Exception as e:
            print(f"  {YELLOW}[RESEARCH] Claude unavailable ({e}) — using rule-based fallback.{RESET}")
            tickers   = _rule_based_fallback(metrics)
            reasoning = "Rule-based fallback"
    else:
        tickers   = _rule_based_fallback(metrics)
        reasoning = "Rule-based selection (Claude disabled)"

    # Filter to tickers that actually had valid metrics
    valid = {m["ticker"] for m in metrics}
    tickers = [t for t in tickers if t in valid]

    print(f"\n{GREEN}{BOLD}[RESEARCH] Selected tickers: {', '.join(tickers)}{RESET}")
    print(f"  Reasoning: {reasoning}\n")

    return tickers


if __name__ == "__main__":
    selected = run_research_agent()
    print("Recommended tickers:", selected)
