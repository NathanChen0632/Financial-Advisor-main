from __future__ import annotations

# Thin wrapper around the Supabase REST API.
# The Python monitor calls these functions to write signals, positions,
# trades, and performance snapshots so the React frontend stays in sync.
#
# Requires:
#   pip install supabase
#   SUPABASE_URL and SUPABASE_SERVICE_KEY in your .env
#
# The service-role key (not the anon key) is used here because the
# frontend's RLS policies block anon writes — only the backend can insert.

import os
from datetime import date
from stock_prediction.utils import YELLOW, RESET

try:
    from supabase import create_client, Client
    _SUPABASE_AVAILABLE = True
except ImportError:
    _SUPABASE_AVAILABLE = False


def _get_client() -> "Client | None":
    if not _SUPABASE_AVAILABLE:
        return None
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


_client: "Client | None" = None

def _db() -> "Client | None":
    global _client
    if _client is None:
        _client = _get_client()
    return _client


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def write_signal(
    ticker:       str,
    action:       str,
    price:        float,
    stop_price:   float  = 0.0,
    target_price: float  = 0.0,
    atr_pct:      float  = 0.0,
    pnl_pct:      float | None = None,
    exit_reason:  str | None   = None,
) -> None:
    db = _db()
    if db is None:
        return
    try:
        row = {
            "ticker":       ticker,
            "action":       action,
            "price":        round(price, 4),
            "stop_price":   round(stop_price,   4) if stop_price   else None,
            "target_price": round(target_price, 4) if target_price else None,
            "atr_pct":      round(atr_pct, 6)      if atr_pct      else None,
            "pnl_pct":      round(pnl_pct, 4)      if pnl_pct is not None else None,
            "exit_reason":  exit_reason,
        }
        db.table("signals").insert(row).execute()
    except Exception as e:
        print(f"  {YELLOW}[SUPABASE] write_signal failed: {e}{RESET}")


# ---------------------------------------------------------------------------
# Positions — upsert keeps exactly one row per ticker (the current state)
# ---------------------------------------------------------------------------

def upsert_position(
    ticker:          str,
    qty:             int,
    entry_price:     float,
    current_price:   float,
    stop_price:      float = 0.0,
    target_price:    float = 0.0,
    strategy_type:   str   = "DQN Equity",
) -> None:
    db = _db()
    if db is None:
        return
    try:
        market_value      = round(qty * current_price, 2)
        unrealized_pnl    = round(qty * (current_price - entry_price), 2)
        unrealized_pnl_pct = round((current_price - entry_price) / entry_price * 100, 4) if entry_price else 0
        row = {
            "ticker":              ticker,
            "strategy_type":       strategy_type,
            "qty":                 qty,
            "entry_price":         round(entry_price,   4),
            "current_price":       round(current_price, 4),
            "stop_price":          round(stop_price,    4) if stop_price   else None,
            "target_price":        round(target_price,  4) if target_price else None,
            "market_value":        market_value,
            "unrealized_pnl":      unrealized_pnl,
            "unrealized_pnl_pct":  unrealized_pnl_pct,
            "updated_at":          "now()",
        }
        db.table("positions").upsert(row, on_conflict="ticker").execute()
    except Exception as e:
        print(f"  {YELLOW}[SUPABASE] upsert_position failed: {e}{RESET}")


def delete_position(ticker: str) -> None:
    db = _db()
    if db is None:
        return
    try:
        db.table("positions").delete().eq("ticker", ticker).execute()
    except Exception as e:
        print(f"  {YELLOW}[SUPABASE] delete_position failed: {e}{RESET}")


# ---------------------------------------------------------------------------
# Trades — append-only history
# ---------------------------------------------------------------------------

def write_trade(
    ticker:        str,
    qty:           int,
    entry_price:   float,
    exit_price:    float,
    entry_time:    str | None = None,
    exit_reason:   str        = "signal",
    strategy_type: str        = "DQN Equity",
) -> None:
    db = _db()
    if db is None:
        return
    try:
        pnl     = round(qty * (exit_price - entry_price), 2)
        pnl_pct = round((exit_price - entry_price) / entry_price * 100, 4) if entry_price else 0
        row = {
            "ticker":        ticker,
            "strategy_type": strategy_type,
            "qty":           qty,
            "entry_price":   round(entry_price, 4),
            "exit_price":    round(exit_price,  4),
            "pnl":           pnl,
            "pnl_pct":       pnl_pct,
            "exit_reason":   exit_reason,
            "entry_time":    entry_time,
        }
        db.table("trades").insert(row).execute()
    except Exception as e:
        print(f"  {YELLOW}[SUPABASE] write_trade failed: {e}{RESET}")


# ---------------------------------------------------------------------------
# Performance snapshots — one per day, upserted on snapshot_date
# ---------------------------------------------------------------------------

def write_performance_snapshot(
    equity:       float,
    bah_equity:   float | None = None,
    daily_return: float | None = None,
    sharpe_ytd:   float | None = None,
    max_drawdown: float | None = None,
    snapshot_date: str | None  = None,
) -> None:
    db = _db()
    if db is None:
        return
    try:
        today = snapshot_date or date.today().isoformat()
        row   = {
            "snapshot_date": today,
            "equity":        round(equity, 2),
            "bah_equity":    round(bah_equity,   2) if bah_equity   is not None else None,
            "daily_return":  round(daily_return,  6) if daily_return is not None else None,
            "sharpe_ytd":    round(sharpe_ytd,    4) if sharpe_ytd   is not None else None,
            "max_drawdown":  round(max_drawdown,  4) if max_drawdown is not None else None,
        }
        db.table("performance_snapshots").upsert(row, on_conflict="snapshot_date").execute()
    except Exception as e:
        print(f"  {YELLOW}[SUPABASE] write_performance_snapshot failed: {e}{RESET}")


# ---------------------------------------------------------------------------
# News items — written by research_agent.py
# ---------------------------------------------------------------------------

def write_news_item(
    headline:     str,
    ticker:       str | None = None,
    summary:      str | None = None,
    source:       str | None = None,
    url:          str | None = None,
    sentiment:    str | None = None,
    published_at: str | None = None,
) -> None:
    db = _db()
    if db is None:
        return
    try:
        row = {
            "ticker":       ticker,
            "headline":     headline,
            "summary":      summary,
            "source":       source,
            "url":          url,
            "sentiment":    sentiment,
            "published_at": published_at,
        }
        db.table("news_items").insert(row).execute()
    except Exception as e:
        print(f"  {YELLOW}[SUPABASE] write_news_item failed: {e}{RESET}")


# ---------------------------------------------------------------------------
# Recommendations — daily market-wide AI buy picks (one row per ticker/day)
# ---------------------------------------------------------------------------

def upsert_recommendation(
    ticker:       str,
    rationale:    str | None   = None,
    score:        float | None = None,
    price:        float | None = None,
    rsi_14:       float | None = None,
    mom_20d:      float | None = None,
    rel_strength: float | None = None,
    batch_date:   str | None   = None,
) -> None:
    db = _db()
    if db is None:
        return
    try:
        row = {
            "ticker":       ticker,
            "rationale":    rationale,
            "score":        round(score, 2)        if score        is not None else None,
            "price":        round(price, 4)        if price        is not None else None,
            "rsi_14":       round(rsi_14, 2)       if rsi_14       is not None else None,
            "mom_20d":      round(mom_20d, 2)      if mom_20d      is not None else None,
            "rel_strength": round(rel_strength, 2) if rel_strength is not None else None,
            "batch_date":   batch_date or date.today().isoformat(),
        }
        db.table("recommendations").upsert(row, on_conflict="ticker,batch_date").execute()
    except Exception as e:
        print(f"  {YELLOW}[SUPABASE] upsert_recommendation failed: {e}{RESET}")


# ---------------------------------------------------------------------------
# Ticker signals — advisory HOLD/SELL/BUY per ticker (one current row each)
# ---------------------------------------------------------------------------

def upsert_ticker_signal(
    ticker:    str,
    action:    str,
    rationale: str | None   = None,
    rsi:       float | None = None,
    momentum:  float | None = None,
    price:     float | None = None,
) -> None:
    db = _db()
    if db is None:
        return
    try:
        row = {
            "ticker":     ticker,
            "action":     action,
            "rationale":  rationale,
            "rsi":        round(rsi, 2)      if rsi      is not None else None,
            "momentum":   round(momentum, 2) if momentum is not None else None,
            "price":      round(price, 4)    if price    is not None else None,
            "updated_at": "now()",
        }
        db.table("ticker_signals").upsert(row, on_conflict="ticker").execute()
    except Exception as e:
        print(f"  {YELLOW}[SUPABASE] upsert_ticker_signal failed: {e}{RESET}")


def fetch_tracked_tickers() -> list[str]:
    # Union of every ticker that appears in any user's holdings or watchlist,
    # so the daily job knows which symbols need an advisory signal.
    db = _db()
    if db is None:
        return []
    tickers: set[str] = set()
    try:
        for table in ("holdings", "watchlist"):
            res = db.table(table).select("ticker").execute()
            for row in (res.data or []):
                if row.get("ticker"):
                    tickers.add(row["ticker"].upper())
    except Exception as e:
        print(f"  {YELLOW}[SUPABASE] fetch_tracked_tickers failed: {e}{RESET}")
    return sorted(tickers)


# ---------------------------------------------------------------------------
# Order requests — UI-initiated trades the monitor executes (master only)
# ---------------------------------------------------------------------------

def fetch_pending_orders() -> list[dict]:
    db = _db()
    if db is None:
        return []
    try:
        res = (
            db.table("order_requests")
            .select("*")
            .eq("status", "pending")
            .order("created_at", desc=False)
            .execute()
        )
        return res.data or []
    except Exception as e:
        print(f"  {YELLOW}[SUPABASE] fetch_pending_orders failed: {e}{RESET}")
        return []


def update_order_status(order_id, status: str, error: str | None = None) -> None:
    db = _db()
    if db is None:
        return
    try:
        row = {"status": status, "processed_at": "now()"}
        if error is not None:
            row["error"] = error[:500]
        db.table("order_requests").update(row).eq("id", order_id).execute()
    except Exception as e:
        print(f"  {YELLOW}[SUPABASE] update_order_status failed: {e}{RESET}")


# ---------------------------------------------------------------------------
# Sell approvals — algo sells the master must approve before they execute
# ---------------------------------------------------------------------------

def create_sell_approval(ticker: str, reason: str, price: float | None = None) -> bool:
    # Returns True if a NEW pending approval was created; False if one is
    # already pending for this ticker (so we don't email on every poll).
    db = _db()
    if db is None:
        return False
    try:
        existing = (
            db.table("sell_approvals")
            .select("id")
            .eq("ticker", ticker)
            .eq("status", "pending")
            .execute()
        )
        if existing.data:
            return False
        db.table("sell_approvals").insert({
            "ticker":          ticker,
            "reason":          reason,
            "suggested_price": round(price, 4) if price is not None else None,
        }).execute()
        return True
    except Exception as e:
        print(f"  {YELLOW}[SUPABASE] create_sell_approval failed: {e}{RESET}")
        return False


def fetch_approved_sells() -> list[dict]:
    db = _db()
    if db is None:
        return []
    try:
        res = (
            db.table("sell_approvals")
            .select("*")
            .eq("status", "approved")
            .execute()
        )
        return res.data or []
    except Exception as e:
        print(f"  {YELLOW}[SUPABASE] fetch_approved_sells failed: {e}{RESET}")
        return []


def mark_sell_executed(approval_id) -> None:
    db = _db()
    if db is None:
        return
    try:
        db.table("sell_approvals").update(
            {"status": "executed", "decided_at": "now()"}
        ).eq("id", approval_id).execute()
    except Exception as e:
        print(f"  {YELLOW}[SUPABASE] mark_sell_executed failed: {e}{RESET}")
