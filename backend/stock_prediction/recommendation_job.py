from __future__ import annotations

# Daily recommendation job — run once a day (e.g. before market open).
#
#   python -m stock_prediction.recommendation_job
#
# It writes two things to Supabase for the React dashboard:
#   1. recommendations  — market-wide AI buy picks (Claude research agent)
#   2. ticker_signals   — a rule-based HOLD/SELL/BUY for every ticker that
#                         appears in any user's holdings or watchlist
#
# Reuses screen_universe / ask_claude / _compute_metrics from research_agent
# so there is no duplicated technical-analysis logic.

from datetime import datetime, timezone

import yfinance as yf

from stock_prediction.utils import GREEN, YELLOW, BOLD, RESET
from stock_prediction.research_agent import (
    screen_universe,
    ask_claude,
    _rule_based_fallback,
)
from stock_prediction import supabase_bridge as sb

# Always pull news for these majors so the News page is never empty, on top of
# whatever the user tracks or the agent recommends.
DEFAULT_NEWS_TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL", "META", "AMZN"]


def _composite_score(m: dict) -> float:
    # Same shape as research_agent._rule_based_fallback scoring, returned as a
    # single number so the dashboard can rank picks.
    score = 0.0
    if m.get("rel_strength") is not None:
        score += m["rel_strength"] * 2.0
    if m.get("mom_20d") is not None:
        score += m["mom_20d"] * 1.0
    if m.get("rsi_14") is not None:
        rsi = m["rsi_14"]
        if 45 <= rsi <= 65:
            score += 10
        elif rsi > 70 or rsi < 35:
            score -= 10
    if m.get("vol_ratio") is not None and m["vol_ratio"] > 1.1:
        score += 5
    return score


def _advisory(m: dict) -> tuple[str, str]:
    # Turn a screening metric dict into a HOLD / SELL / BUY suggestion with a
    # short human-readable rationale. Mirrors the thresholds used for picks.
    rsi = m.get("rsi_14")
    mom = m.get("mom_20d")
    rel = m.get("rel_strength")

    bits = []
    if rsi is not None: bits.append(f"RSI {rsi:.0f}")
    if mom is not None: bits.append(f"20d momentum {mom:+.1f}%")
    if rel is not None: bits.append(f"{rel:+.1f}% vs SPY")
    summary = ", ".join(bits) or "limited data"

    # SELL: overbought, or clearly rolling over.
    if (rsi is not None and rsi >= 72) or (mom is not None and mom <= -8):
        return "SELL", f"Consider trimming — {summary}."
    # BUY: momentum sweet spot, outperforming, not overbought.
    if (
        rsi is not None and 45 <= rsi <= 68
        and mom is not None and mom > 0
        and (rel is None or rel > 0)
    ):
        return "BUY", f"Constructive setup — {summary}."
    return "HOLD", f"No strong signal — {summary}."


def _news_field(content: dict, *keys):
    # Pull the first present key from a yfinance news 'content' dict.
    for k in keys:
        v = content.get(k)
        if isinstance(v, dict):
            # provider → displayName; canonicalUrl/clickThroughUrl → url
            v = v.get("displayName") or v.get("url")
        if v:
            return v
    return None


def store_news(tickers: list[str], per_ticker: int = 6) -> None:
    # Pull recent Yahoo Finance headlines for each ticker and write the new
    # ones to news_items. De-dupes on URL so daily re-runs don't pile up.
    db = sb._db()
    seen: set[str] = set()
    if db is not None:
        try:
            res = db.table("news_items").select("url").order("published_at", desc=True).limit(2000).execute()
            seen = {r["url"] for r in (res.data or []) if r.get("url")}
        except Exception:
            pass

    written = 0
    for t in tickers:
        try:
            items = yf.Ticker(t).news or []
        except Exception as e:
            print(f"  {YELLOW}[NEWS] {t}: could not fetch ({e}){RESET}")
            continue

        for it in items[:per_ticker]:
            # yfinance >=1.x nests everything under 'content'; older versions are flat.
            c       = it.get("content") or it
            url     = _news_field(c, "canonicalUrl", "clickThroughUrl", "link")
            title   = c.get("title")
            if not url or not title or url in seen:
                continue
            summary = c.get("summary") or c.get("description") or None
            source  = _news_field(c, "provider") or c.get("publisher")

            # published_at: ISO string on new format, epoch seconds on old.
            pub = c.get("pubDate") or c.get("displayTime") or c.get("providerPublishTime")
            if isinstance(pub, (int, float)):
                pub = datetime.fromtimestamp(pub, tz=timezone.utc).isoformat()

            sb.write_news_item(
                headline=title, ticker=t, summary=summary,
                source=source, url=url, published_at=pub,
            )
            seen.add(url)
            written += 1

    print(f"  {GREEN}[NEWS] Stored {written} new headline(s) across {len(tickers)} ticker(s).{RESET}")


def run_recommendation_job() -> None:
    print(f"\n{BOLD}[JOB] Generating daily recommendations...{RESET}")

    # ---- 1. Market-wide AI buy picks -------------------------------------
    metrics   = screen_universe()
    by_ticker = {m["ticker"]: m for m in metrics}

    if metrics:
        try:
            tickers, reasoning = ask_claude(metrics)
        except Exception as e:
            print(f"  {YELLOW}[JOB] Claude unavailable ({e}) — using rule-based picks.{RESET}")
            tickers, reasoning = _rule_based_fallback(metrics), "Rule-based selection"

        picks = [t for t in tickers if t in by_ticker]
        for t in picks:
            m = by_ticker[t]
            sb.upsert_recommendation(
                ticker=t,
                rationale=reasoning,
                score=_composite_score(m),
                price=m.get("price"),
                rsi_14=m.get("rsi_14"),
                mom_20d=m.get("mom_20d"),
                rel_strength=m.get("rel_strength"),
            )
        print(f"  {GREEN}[JOB] Wrote {len(picks)} recommendation(s): {', '.join(picks)}{RESET}")
    else:
        print(f"  {YELLOW}[JOB] No screening metrics — skipping recommendations.{RESET}")

    # ---- 2. Advisory signals for every user-tracked ticker ---------------
    tracked = sb.fetch_tracked_tickers()
    if tracked:
        print(f"  [JOB] Computing advisory signals for {len(tracked)} tracked ticker(s)...")
        tmetrics = screen_universe(universe=tracked)
        for m in tmetrics:
            action, rationale = _advisory(m)
            sb.upsert_ticker_signal(
                ticker=m["ticker"],
                action=action,
                rationale=rationale,
                rsi=m.get("rsi_14"),
                momentum=m.get("mom_20d"),
                price=m.get("price"),
            )
        print(f"  {GREEN}[JOB] Wrote advisory signals for {len(tmetrics)} ticker(s).{RESET}")
    else:
        print(f"  [JOB] No user holdings/watchlist tickers to advise on yet.")

    # ---- 3. Yahoo Finance news for majors + everything users track --------
    news_tickers = sorted(set(DEFAULT_NEWS_TICKERS) | set(tracked))
    print(f"  [JOB] Fetching Yahoo Finance news for {len(news_tickers)} ticker(s)...")
    store_news(news_tickers)

    print(f"{GREEN}{BOLD}[JOB] Done.{RESET}\n")


if __name__ == "__main__":
    run_recommendation_job()
