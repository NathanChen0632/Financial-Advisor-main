from __future__ import annotations

import argparse
import os
import smtplib
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

from stock_prediction.utils import GREEN, BLUE, RED, YELLOW, GREY, BOLD, RESET
from stock_prediction.data_collection import download_stock_data, download_spy_context, TICKERS
from stock_prediction.features import build_features, get_feature_columns
from stock_prediction.models import fit_feature_scaler
from stock_prediction.rl_agent import train_dqn_agent

try:
    from stock_prediction.alpaca_trader import AlpacaConfig, AlpacaTrader
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False

try:
    from stock_prediction.straddle import StraddleMonitor
    _STRADDLE_AVAILABLE = True
except ImportError:
    _STRADDLE_AVAILABLE = False

# Bridge to Supabase so the React dashboard mirrors the live Alpaca account.
# Every function here is a no-op when SUPABASE_URL / SUPABASE_SERVICE_KEY are
# unset, so importing it unconditionally is safe even without Supabase.
from stock_prediction import supabase_bridge as sb


# ---------------------------------- Market hours ---------------------------


MARKET_TZ    = pytz.timezone("America/New_York")
MARKET_OPEN  = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)


def is_market_open() -> bool:
    now_et = datetime.now(MARKET_TZ)
    if now_et.weekday() >= 5:
        return False
    t = now_et.time()
    return MARKET_OPEN <= t <= MARKET_CLOSE


def seconds_until_open() -> float:
    now_et     = datetime.now(MARKET_TZ)
    today_open = MARKET_TZ.localize(
        datetime(now_et.year, now_et.month, now_et.day, 9, 30)
    )
    if now_et < today_open and now_et.weekday() < 5:
        return (today_open - now_et).total_seconds()
    days_ahead = 1
    while True:
        candidate = today_open + pd.Timedelta(days=days_ahead)
        if candidate.weekday() < 5:
            return (candidate - now_et).total_seconds()
        days_ahead += 1


# ------------------------------ Strategy state ---------------------------------


class Position(Enum):
    FLAT = "FLAT"   # not holding — looking to buy
    LONG = "LONG"   # holding     — looking to sell


@dataclass
class StrategyState:

    ticker:       str
    position:     Position = Position.FLAT
    entry_price:  float    = 0.0
    stop_price:   float    = 0.0
    target_price: float    = 0.0
    entry_time:   str      = ""
    days_held:    int      = 0
    trade_log:    list     = field(default_factory=list)



# ----------------------------------- Email alerts -----------------------------------


@dataclass
class EmailConfig:

    sender:   str
    password: str
    to:       str
    host:     str = "smtp.gmail.com"
    port:     int = 587

    @classmethod
    def from_env(cls) -> "EmailConfig | None":

        sender   = os.environ.get("SMTP_USER")
        password = os.environ.get("SMTP_PASSWORD")
        to       = os.environ.get("ALERT_TO")

        if not all([sender, password, to]):
            missing = [v for v, val in [
                ("SMTP_USER", sender), ("SMTP_PASSWORD", password), ("ALERT_TO", to)
            ] if not val]
            print(f"{YELLOW}[EMAIL] Missing env vars: {', '.join(missing)} — alerts disabled.{RESET}")
            print(f"{YELLOW}[EMAIL] Set them in a .env file or export before running.{RESET}")
            return None

        return cls(
            sender=sender,
            password=password,
            to=to,
            host=os.environ.get("SMTP_HOST", "smtp.gmail.com"),
            port=int(os.environ.get("SMTP_PORT", 587)),
        )


def _build_email_body(
    action:      str,
    ticker:      str,
    price:       float,
    state:       "StrategyState",
    votes:       dict[str, int],
) -> tuple[str, str]:
    """Return (subject, html_body) for a BUY or SELL alert."""
    ts       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    emoji    = "🟢" if action == "BUY" else "🔴"
    subject  = f"{emoji} {action} {ticker} @ ${price:.2f} — Trading Alert"

    if action == "BUY":
        action_line = f"<b style='color:green'>BUY {ticker} NOW @ ${price:.2f}</b>"
        detail_line = f"New position opened. Watching for SELL signal."
    else:
        pnl      = (price - state.entry_price) / state.entry_price * 100
        sign     = "+" if pnl >= 0 else ""
        colour   = "green" if pnl >= 0 else "red"
        action_line = f"<b style='color:red'>SELL {ticker} NOW @ ${price:.2f}</b>"
        detail_line = (
            f"Entry: ${state.entry_price:.2f} &nbsp;|&nbsp; "
            f"P&amp;L: <b style='color:{colour}'>{sign}{pnl:.2f}%</b>"
        )

    html = f"""
    <html><body style="font-family:Arial,sans-serif;max-width:480px">
      <h2>{emoji} DQN Trading Signal — {ticker}</h2>
      <p style="font-size:1.3em">{action_line}</p>
      <p>{detail_line}</p>
      <p style="color:#888">Time: {ts}</p>
      <hr/>
      <p style="color:#aaa;font-size:0.8em">
        Generated by a Deep Q-Network trained on 10 years of market data.
        This is a research tool, not financial advice.
      </p>
    </body></html>
    """
    return subject, html


def send_email_alert(
    cfg:    "EmailConfig",
    action: str,
    ticker: str,
    price:  float,
    state:  "StrategyState",
    votes:  dict[str, int],
) -> None:
    """Send a BUY or SELL alert email. Silently logs on failure."""
    try:
        subject, html = _build_email_body(action, ticker, price, state, votes)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = cfg.sender
        msg["To"]      = cfg.to
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(cfg.host, cfg.port) as server:
            server.ehlo()
            server.starttls()
            server.login(cfg.sender, cfg.password)
            server.sendmail(cfg.sender, cfg.to, msg.as_string())

        print(f"  {GREEN}[EMAIL] Alert sent to {cfg.to}{RESET}")
    except Exception as e:
        print(f"  {YELLOW}[EMAIL] Failed to send alert: {e}{RESET}")


# --------------------------------- Data helpers ------------------------------------


def fetch_latest_bar(ticker: str) -> pd.Series | None:
    """Fetch the most recent 1-minute bar for the current trading session."""
    try:
        raw = yf.download(
            ticker,
            period="1d",
            interval="1m",
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return raw[["Open", "High", "Low", "Close", "Volume"]].iloc[-1]
    except Exception as e:
        print(f"  [WARN] Could not fetch intraday bar for {ticker}: {e}")
        return None


def build_live_feature_row(
    history_df: pd.DataFrame,
    live_bar: pd.Series,
) -> tuple[np.ndarray, list[str]] | tuple[None, None]:
    """
    Append the live bar to daily history and return the feature vector
    for that bar so all rolling-window indicators are correctly computed.
    """
    now_ts   = pd.Timestamp.now().normalize()
    live_row = pd.DataFrame(
        {
            "Open":   [float(live_bar["Open"])],
            "High":   [float(live_bar["High"])],
            "Low":    [float(live_bar["Low"])],
            "Close":  [float(live_bar["Close"])],
            "Volume": [float(live_bar["Volume"])],
        },
        index=[now_ts],
    )

    combined = history_df.copy()
    if now_ts in combined.index:
        combined = combined.drop(index=now_ts)
    combined = pd.concat([combined, live_row]).sort_index()

    try:
        # Add a dummy next-day row so build_features doesn't drop the live bar
        dummy_ts  = now_ts + pd.Timedelta(days=1)
        dummy_row = pd.DataFrame(
            {col: [combined[col].iloc[-1]] for col in combined.columns},
            index=[dummy_ts],
        )
        feat_df      = build_features(pd.concat([combined, dummy_row]))
        feature_cols = get_feature_columns(feat_df)
        return feat_df.iloc[-2][feature_cols].values.reshape(1, -1), feature_cols
    except Exception as e:
        print(f"  [WARN] Feature computation failed: {e}")
        return None, None


# -------------------------- Model training ------------------------


def train_models_on_history(ticker: str, history_df: pd.DataFrame, spy_df=None, n_episodes: int = 500) -> dict:
    print(f"  [{ticker}] Building features...")

    # Include SPY market context so the live model matches backtest feature space.
    # Without this, the agent is missing spy_return, spy_momentum20, relative_strength
    # and gets a different state vector than what it was designed for.
    feat_df      = build_features(history_df, spy_df=spy_df)
    feature_cols = get_feature_columns(feat_df)
    X            = feat_df[feature_cols].values
    prices       = history_df.loc[feat_df.index, "Close"].values.flatten()

    # Scaler must be fit here so the DQN receives normalized inputs.
    # Without normalization, RSI (0-100) dominates ATR (0.01) in the Q-network
    # and the agent makes poor decisions regardless of training length.
    scaler        = fit_feature_scaler(X)
    X_scaled      = scaler.transform(X)

    print(f"  [{ticker}] Training DQN — {n_episodes} episodes on {len(X)} days of history...")
    dqn = train_dqn_agent(
        X_train=X,
        daily_returns_train=feat_df["daily_return"].values,
        prices_train=prices,
        feature_cols=feature_cols,
        n_episodes=n_episodes,
        X_train_scaled=X_scaled,
    )
    dqn.scaler = scaler

    print(f"  [{ticker}] Done.\n")
    return {
        "DQN":          dqn,
        "feature_cols": feature_cols,
    }


# ------------------------- Strategy signal resolution -------------------------


def get_consensus(trained: dict, X_live: np.ndarray) -> tuple[int, dict[str, int]]:
    """
    Run the DQN agent and return its signal.
    Returns (signal: 0|1, votes dict for display).
    """
    pred  = int(trained["DQN"].predict(X_live)[0])
    votes = {"DQN Agent": pred}
    return pred, votes


def resolve_strategy_action(
    state:        StrategyState,
    signal:       int,
    price:        float,
    features:     np.ndarray,
    feature_cols: list,
) -> str:
    """
    Apply DQN signal to current position state.
    On BUY: computes ATR-based stop and 2:1 target and stores them in state.
    Updates state in place. Returns action string.
    """
    from stock_prediction.rl_agent import TradingConfig
    cfg = TradingConfig()

    if state.position == Position.FLAT:
        if signal == 1:
            # Compute stop and target at entry using ATR
            atr_idx    = feature_cols.index("atr14_pct") if "atr14_pct" in feature_cols else None
            atr_pct    = float(features[atr_idx]) if atr_idx is not None else 0.015
            stop_dist  = cfg.stop_atr_mult * atr_pct * price
            stop_price = price - stop_dist
            tgt_price  = price + cfg.min_rr_ratio * stop_dist

            state.position     = Position.LONG
            state.entry_price  = price
            state.stop_price   = stop_price
            state.target_price = tgt_price
            state.entry_time   = datetime.now().strftime("%H:%M:%S")
            state.days_held    = 0
            return "BUY"
        return "WAIT"

    else:  # Position.LONG
        state.days_held += 1

        # Check if stop or target was hit regardless of DQN signal
        if price <= state.stop_price:
            signal = 0   # force exit, stop hit
        elif price >= state.target_price:
            signal = 0   # force exit, target hit
        elif state.days_held >= cfg.max_holding_days:
            signal = 0   # force exit, time stop

        if signal == 0:
            pnl = (price - state.entry_price) / state.entry_price * 100
            reason = "stop hit" if price <= state.stop_price else (
                     "target hit" if price >= state.target_price else
                     "time stop" if state.days_held >= cfg.max_holding_days else "signal")
            state.trade_log.append({
                "entry":  state.entry_price,
                "exit":   price,
                "pnl%":   pnl,
                "time":   datetime.now().strftime("%H:%M:%S"),
                "reason": reason,
            })
            state.position     = Position.FLAT
            state.entry_price  = 0.0
            state.stop_price   = 0.0
            state.target_price = 0.0
            state.days_held    = 0
            state.entry_time   = ""
            return "SELL"
        return "HOLD"


# ----------------------------------- Poll display ----------------------------------------



def print_strategy_signal(
    action:     str,
    ticker:     str,
    price:      float,
    state:      StrategyState,
    votes:      dict[str, int],
    consensus:  int,
) -> None:
    ts = datetime.now().strftime("%H:%M:%S")

    # --- colour and banner by action ---
    if action == "BUY":
        colour  = GREEN
        banner  = f"  {BOLD}{GREEN}>>> BUY {ticker} NOW  @  ${price:.2f} <<<{RESET}"
        summary = (
            f"  {GREEN}Entry: ${price:.2f}  |  "
            f"Stop: ${state.stop_price:.2f}  |  "
            f"Target: ${state.target_price:.2f}  |  "
            f"R/R: 2:1{RESET}"
        )
    elif action == "SELL":
        colour  = RED
        pnl     = (price - state.entry_price) / state.entry_price * 100
        sign    = "+" if pnl >= 0 else ""
        reason  = state.trade_log[-1]["reason"] if state.trade_log else "signal"
        banner  = f"  {BOLD}{RED}>>> SELL {ticker} NOW  @  ${price:.2f} <<<{RESET}"
        summary = (
            f"  {RED}Reason: {reason}  |  Entry: ${state.entry_price:.2f}  "
            f"|  P&L: {sign}{pnl:.2f}%{RESET}"
        )
    elif action == "HOLD":
        colour  = BLUE
        pnl     = (price - state.entry_price) / state.entry_price * 100
        sign    = "+" if pnl >= 0 else ""
        banner  = f"  {BOLD}{BLUE}HOLD {ticker}  @  ${price:.2f}{RESET}"
        summary = (
            f"  {BLUE}Day {state.days_held}  |  "
            f"Entry: ${state.entry_price:.2f}  Stop: ${state.stop_price:.2f}  "
            f"Target: ${state.target_price:.2f}  |  "
            f"Unrealised P&L: {sign}{pnl:.2f}%{RESET}"
        )
    else:  # WAIT
        colour  = GREY
        banner  = f"  {GREY}WAIT — no position in {ticker}  (${price:.2f}){RESET}"
        summary = f"  {GREY}Watching for a valid BUY setup (R/R ≥ 2:1, volume confirmed)...{RESET}"

    dqn_signal = votes.get("DQN Agent", consensus)
    dot        = f"{GREEN}●{RESET}" if dqn_signal == 1 else f"{RED}●{RESET}"

    print(f"\n  {'═'*58}")
    print(f"  [{ts}]  {ticker}  ${price:.2f}")
    print(f"  {'═'*58}")
    print(banner)
    print(summary)
    print(f"  {'─'*58}")
    print(f"  DQN Signal:  {dot}  {'UP — expects price to rise' if dqn_signal == 1 else 'DOWN — expects price to fall'}")

    # Session trade log
    if state.trade_log:
        print(f"  {'─'*58}")
        print(f"  Session trades:")
        for t in state.trade_log[-3:]:
            sign = "+" if t["pnl%"] >= 0 else ""
            col  = GREEN if t["pnl%"] >= 0 else RED
            print(f"    {t['time']}  entry ${t['entry']:.2f}  "
                  f"exit ${t['exit']:.2f}  {col}{sign}{t['pnl%']:.2f}%{RESET}")

    print(f"  {'═'*58}")


# ---------------------------------------------------------------------------
# Supabase sync — mirror the live Alpaca account onto the dashboard
# ---------------------------------------------------------------------------

def _sync_alpaca_to_supabase(
    trader:   "AlpacaTrader",
    states:   dict,
    prev_open: dict,
) -> dict:
    # Mirror Alpaca's live state into Supabase so the website reflects the
    # real account. Alpaca is the single source of truth here:
    #   - every open position is upserted with a fresh price / P&L
    #   - any ticker that was open last loop but is gone now (monitor SELL,
    #     trailing-stop exit, or a manual close in the Alpaca UI) is recorded
    #     as a completed trade and removed from the positions table
    #   - a daily equity snapshot is written for the performance page
    #
    # Returns the updated {ticker: {...}} map to feed into the next loop.
    try:
        positions = trader.client.get_all_positions()
    except Exception as e:
        print(f"  {YELLOW}[SUPABASE] Could not read Alpaca positions: {e}{RESET}")
        return prev_open

    open_now = {}
    for p in positions:
        sym   = p.symbol
        qty   = int(float(p.qty))
        entry = float(p.avg_entry_price)
        curr  = float(p.current_price)
        st    = states.get(sym)
        sb.upsert_position(
            ticker=sym,
            qty=qty,
            entry_price=entry,
            current_price=curr,
            stop_price=st.stop_price   if st else 0.0,
            target_price=st.target_price if st else 0.0,
        )
        open_now[sym] = {
            "qty":        qty,
            "entry":      entry,
            "last_price": curr,
            # Preserve the first time we saw this position so the trade
            # record gets a sensible entry_time; default to now for new ones.
            "first_seen": prev_open.get(sym, {}).get("first_seen") or datetime.now().isoformat(),
        }

    # Positions open last loop but gone now → log the completed trade & drop it.
    # last_price is the most recent price we saw, used as the exit proxy.
    for sym, info in prev_open.items():
        if sym not in open_now:
            sb.write_trade(
                ticker=sym,
                qty=info["qty"],
                entry_price=info["entry"],
                exit_price=info["last_price"],
                entry_time=info.get("first_seen"),
                exit_reason="closed",
            )
            sb.delete_position(sym)

    # Daily equity snapshot for the performance page.
    try:
        acct      = trader.client.get_account()
        equity    = float(acct.equity)
        last_eq   = float(acct.last_equity)
        sb.write_performance_snapshot(
            equity=equity,
            daily_return=(equity - last_eq) / last_eq if last_eq else None,
        )
    except Exception:
        pass

    return open_now


# ---------------------------------------------------------------------------
# UI-initiated trades — process the order_requests queue (master)
# ---------------------------------------------------------------------------

def _estimate_stop(ticker: str, entry: float, history: dict, spy_df) -> tuple[float, float]:
    # Reuse the same ATR-based stop sizing the restore path uses, so a
    # UI-bought ticker gets a sensible stop/target for the DQN to manage.
    from stock_prediction.rl_agent import TradingConfig
    cfg       = TradingConfig()
    atr_guess = 0.015
    try:
        feat_df      = build_features(history[ticker], spy_df=spy_df)
        feature_cols = get_feature_columns(feat_df)
        if "atr14_pct" in feature_cols:
            atr_guess = float(feat_df["atr14_pct"].iloc[-1])
    except Exception:
        pass
    return cfg.stop_atr_mult * atr_guess * entry, cfg.min_rr_ratio


def _ensure_tracked(ticker, history, trained, states, spy_df, today_str) -> bool:
    # Make sure the DQN strategy is tracking `ticker`. Trains a model on first
    # sighting of a UI-bought symbol so the algorithm can decide when to sell.
    if ticker in trained:
        return True
    try:
        print(f"  [ALPACA] New ticker {ticker} — downloading history & training model...")
        history[ticker] = download_stock_data(ticker, start="2015-01-01", end=today_str)
        trained[ticker] = train_models_on_history(ticker, history[ticker], spy_df=spy_df)
        states[ticker]  = StrategyState(ticker=ticker)
        return True
    except Exception as e:
        print(f"  {RED}[ALPACA] Could not start tracking {ticker}: {e}{RESET}")
        return False


def _process_approved_sells(trader, states) -> None:
    # Execute master sells that have been approved in the dashboard, then
    # mark them executed so they aren't picked up again.
    for a in sb.fetch_approved_sells():
        ticker = (a.get("ticker") or "").upper()
        try:
            if trader.submit_sell(ticker, reason="approved (UI)") and ticker in states:
                states[ticker].position = Position.FLAT
        except Exception as e:
            print(f"  {RED}[ALPACA] Approved sell {ticker} failed: {e}{RESET}")
        finally:
            sb.mark_sell_executed(a.get("id"))


def _process_order_requests(trader, history, trained, states, spy_df, today_str) -> None:
    # Execute pending UI trades from Supabase and report status back.
    for o in sb.fetch_pending_orders():
        oid    = o.get("id")
        ticker = (o.get("ticker") or "").upper()
        side   = o.get("side")
        qty    = o.get("qty")
        sb.update_order_status(oid, "executing")
        try:
            if side == "buy":
                if not _ensure_tracked(ticker, history, trained, states, spy_df, today_str):
                    sb.update_order_status(oid, "error", "could not initialise ticker")
                    continue
                bar   = fetch_latest_bar(ticker)
                price = float(bar["Close"]) if bar is not None else float(history[ticker]["Close"].iloc[-1])
                stop_dist, rr = _estimate_stop(ticker, price, history, spy_df)
                stop_price    = price - stop_dist
                if trader.submit_buy(ticker, price, stop_price, qty=qty):
                    st = states[ticker]
                    st.position     = Position.LONG
                    st.entry_price  = price
                    st.stop_price   = stop_price
                    st.target_price = price + rr * stop_dist
                    st.entry_time   = "ui"
                    sb.update_order_status(oid, "filled")
                else:
                    sb.update_order_status(oid, "rejected", "already in position or size 0")
            elif side == "sell":
                if trader.submit_sell(ticker, reason="manual (UI)"):
                    if ticker in states:
                        states[ticker].position = Position.FLAT
                    sb.update_order_status(oid, "filled")
                else:
                    sb.update_order_status(oid, "rejected", "no position to sell")
            else:
                sb.update_order_status(oid, "error", f"unknown side {side}")
        except Exception as e:
            sb.update_order_status(oid, "error", str(e))
            print(f"  {RED}[ALPACA] Order {oid} ({side} {ticker}) failed: {e}{RESET}")


# ---------------------------------------------------------------------------
# Single poll
# ---------------------------------------------------------------------------

def poll_once(
    ticker:     str,
    history_df: pd.DataFrame,
    trained:    dict,
    state:      StrategyState,
    email_cfg:  EmailConfig | None = None,
    trader:     "AlpacaTrader | None" = None,
) -> None:
    live_bar = fetch_latest_bar(ticker)
    if live_bar is None:
        print(f"  [{ticker}] No intraday data — market may not have opened yet.")
        return

    X_live, _ = build_live_feature_row(history_df, live_bar)
    if X_live is None:
        print(f"  [{ticker}] Feature computation failed.")
        return

    price = float(live_bar["Close"])

    # Use predict_step so the agent has full trade context (stop, target, days held)
    pred = trained["DQN"].predict_step(
        features=X_live[0],
        position=1 if state.position == Position.LONG else 0,
        days_held=state.days_held,
        entry_price=state.entry_price,
        stop_price=state.stop_price,
        target_price=state.target_price,
        current_price=price,
    )

    votes  = {"DQN Agent": pred}
    action = resolve_strategy_action(state, pred, price, X_live[0], trained["feature_cols"])

    print_strategy_signal(action, ticker, price, state, votes, pred)

    # Send email on actionable signals (not HOLD or WAIT). For a master with
    # live trading, the SELL email is sent once when the approval is created
    # (below) — not here — so a lingering pending sell doesn't re-email.
    if email_cfg and (action == "BUY" or (action == "SELL" and trader is None)):
        send_email_alert(email_cfg, action, ticker, price, state, votes)

    # Submit live/paper order to Alpaca on actionable signals.
    # On success, write the decision to Supabase as a signal so the
    # dashboard's recommendations feed reflects what the strategy did.
    # (Positions/trades/equity are synced separately from Alpaca itself —
    # see _sync_alpaca_to_supabase — to keep Alpaca the single source of truth.)
    if trader is not None:
        if action == "BUY":
            if trader.submit_buy(ticker, price, state.stop_price):
                sb.write_signal(
                    ticker, "BUY", price,
                    stop_price=state.stop_price,
                    target_price=state.target_price,
                )
        elif action == "SELL":
            reason = state.trade_log[-1]["reason"] if state.trade_log else "signal"
            # Approval-gated: do NOT sell here. Queue a pending approval and let
            # the master approve it in the dashboard; _process_approved_sells
            # executes it on a later tick. create_sell_approval returns False if
            # one is already pending, so we only write the signal/email once.
            pos     = trader.get_position(ticker)
            pnl_pct = float(pos.unrealized_plpc) * 100 if pos else None
            if sb.create_sell_approval(ticker, reason, price):
                sb.write_signal(
                    ticker, "SELL", price,
                    pnl_pct=pnl_pct,
                    exit_reason=reason,
                )
                if email_cfg:
                    send_email_alert(email_cfg, "SELL", ticker, price, state, votes)
                print(f"  {YELLOW}[ALPACA] SELL pending approval for {ticker} — awaiting master OK.{RESET}")


# ----------------------- Main monitor loop ---------------------------



def run_monitor(
    tickers:           list[str],
    interval_minutes:  int  = 5,
    skip_market_check: bool = False,
    email_alerts:      bool = False,
    alpaca_trading:    bool = False,
    alpaca_live:       bool = False,
    straddle_trading:  bool = False,
) -> None:

    # Load email config from env vars if alerts are requested
    email_cfg = EmailConfig.from_env() if email_alerts else None

    # Connect to Alpaca if trading is enabled
    trader = None
    if alpaca_trading:
        if not _ALPACA_AVAILABLE:
            print(f"{YELLOW}[ALPACA] alpaca-py not installed. Run: pip install alpaca-py{RESET}")
        else:
            cfg = AlpacaConfig.from_env()
            if cfg is not None:
                # Respect the --live flag — default is always paper for safety
                if alpaca_live:
                    cfg.paper = False
                    print(f"{RED}{BOLD}[ALPACA] LIVE trading mode enabled — real money at risk!{RESET}")
                trader = AlpacaTrader(cfg)

    # Straddle monitor runs alongside the equity strategy — separate capital pool,
    # separate positions. Only available when Alpaca is connected.
    straddle_monitor = None
    if straddle_trading:
        if not _STRADDLE_AVAILABLE:
            print(f"{YELLOW}[STRADDLE] straddle module unavailable — check scipy/alpaca-py install.{RESET}")
        elif trader is None:
            print(f"{YELLOW}[STRADDLE] --straddle requires --alpaca to execute orders.{RESET}")
        else:
            api_key    = os.environ.get("ALPACA_API_KEY", "")
            secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
            paper      = not alpaca_live
            try:
                straddle_monitor = StraddleMonitor(api_key, secret_key, paper=paper)
            except Exception as e:
                print(f"{YELLOW}[STRADDLE] Could not initialise: {e}{RESET}")

    print("\n" + "="*60)
    print(f"  {BOLD}TRADING STRATEGY MONITOR{RESET}")
    print(f"  Tickers  : {', '.join(tickers)}")
    print(f"  Interval : every {interval_minutes} minute(s)")
    print(f"  Strategy : FLAT → BUY → HOLD → SELL → FLAT")
    print(f"  Market   : {'always run' if skip_market_check else 'NYSE/NASDAQ hours only'}")
    email_status = f"{GREEN}enabled → {email_cfg.to}{RESET}" if email_cfg else f"{GREY}disabled{RESET}"
    print(f"  Email    : {email_status}")
    if trader is not None:
        mode = f"{RED}LIVE{RESET}" if not trader.paper else f"{YELLOW}PAPER{RESET}"
        print(f"  Alpaca   : {GREEN}enabled ({mode}){RESET}")
    else:
        print(f"  Alpaca   : {GREY}disabled (signal-only){RESET}")
    if straddle_monitor is not None:
        print(f"  Straddle : {GREEN}enabled — ATM call+put, 40% profit target{RESET}")
    else:
        print(f"  Straddle : {GREY}disabled{RESET}")
    print("  Press Ctrl-C to stop.")
    print("="*60 + "\n")

    from datetime import date
    today_str = date.today().isoformat()

    history = {}
    trained = {}
    states  = {}

    # Download SPY once and share across all tickers so the live model
    # has the same market context features as the backtest model.
    print("[STARTUP] Downloading SPY market context...")
    try:
        spy_df = download_spy_context(start="2015-01-01", end=today_str)
    except Exception:
        spy_df = None
        print("[WARN] Could not download SPY — market context features disabled.")

    for ticker in tickers:
        print(f"[STARTUP] Downloading history for {ticker}...")
        try:
            history[ticker] = download_stock_data(ticker, start="2015-01-01", end=today_str)
            print(f"[STARTUP] Training models for {ticker}...")
            trained[ticker] = train_models_on_history(ticker, history[ticker], spy_df=spy_df)
            states[ticker]  = StrategyState(ticker=ticker)
        except Exception as e:
            print(f"[ERROR] Could not initialise {ticker}: {e}")

    # Sync open positions from Alpaca so a restart doesn't cause double-entries.
    # If we were holding AAPL before the script stopped, Alpaca still has the
    # position — this restores StrategyState so we don't buy it again.
    if trader is not None and trained:
        print("\n[STARTUP] Syncing open positions from Alpaca...")
        restored = trader.sync_positions_from_alpaca(list(trained.keys()))
        for ticker, info in restored.items():
            if ticker in states:
                from stock_prediction.rl_agent import TradingConfig
                cfg       = TradingConfig()
                entry     = info["entry_price"]
                atr_guess = 0.015  # conservative fallback if feature data unavailable
                try:
                    feat_df      = build_features(history[ticker], spy_df=spy_df)
                    feature_cols = get_feature_columns(feat_df)
                    if "atr14_pct" in feature_cols:
                        atr_guess = float(feat_df["atr14_pct"].iloc[-1])
                except Exception:
                    pass
                stop_dist  = cfg.stop_atr_mult * atr_guess * entry
                states[ticker].position     = Position.LONG
                states[ticker].entry_price  = entry
                states[ticker].stop_price   = entry - stop_dist
                states[ticker].target_price = entry + cfg.min_rr_ratio * stop_dist
                states[ticker].entry_time   = "restored"
                print(f"  Restored state for {ticker}: entry=${entry:.2f}  stop=${states[ticker].stop_price:.2f}")

    if not trained:
        print("[ERROR] No tickers could be initialised. Exiting.")
        return

    print(f"\n{GREEN}[READY] Strategy is live. Watching: {', '.join(trained.keys())}{RESET}\n")

    # Tracks which tickers were open in Alpaca on the previous loop, so the
    # reconciler can detect positions that closed (incl. trailing-stop exits).
    synced_open: dict = {}

    try:
        while True:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if not skip_market_check and not is_market_open():
                secs = seconds_until_open()
                hrs  = int(secs // 3600)
                mins = int((secs % 3600) // 60)
                print(f"[{now_str}] Market closed — next open in {hrs}h {mins}m. Sleeping...")
                for _ in range(min(int(secs), 300)):
                    time.sleep(1)
                continue

            if trader is not None:
                trader.print_account_summary()
                # Execute any UI-initiated trades queued in Supabase, then
                # process master sell approvals (see Phase 5).
                _process_order_requests(trader, history, trained, states, spy_df, today_str)
                _process_approved_sells(trader, states)

            for ticker in trained:
                try:
                    poll_once(ticker, history[ticker], trained[ticker], states[ticker], email_cfg, trader)
                except Exception as e:
                    print(f"  [{ticker}] Poll error: {e}")

            # Straddle monitor: screen for new entries and manage open positions.
            # Runs after the equity strategy so both strategies can act on the same tick.
            if straddle_monitor is not None:
                try:
                    price_history = {
                        t: history[t]["Close"].squeeze()
                        for t in trained if t in history
                    }
                    straddle_monitor.poll(list(trained.keys()), price_history)
                    straddle_monitor.print_status()
                except Exception as e:
                    print(f"  [STRADDLE] Poll error: {e}")

            # Mirror the live Alpaca account to Supabase for the dashboard.
            if trader is not None:
                synced_open = _sync_alpaca_to_supabase(trader, states, synced_open)

            print(f"\n  Next check in {interval_minutes} minute(s)...  [{now_str}]")
            time.sleep(interval_minutes * 60)

    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}[MONITOR] Stopped by user.{RESET}")
        # Print final session summary
        print(f"\n{'='*60}")
        print("  SESSION SUMMARY")
        print(f"{'='*60}")
        for ticker, state in states.items():
            print(f"\n  {ticker}")
            if state.trade_log:
                total_pnl = sum(t["pnl%"] for t in state.trade_log)
                wins      = sum(1 for t in state.trade_log if t["pnl%"] > 0)
                print(f"  Completed trades : {len(state.trade_log)}")
                print(f"  Win rate         : {wins}/{len(state.trade_log)}")
                print(f"  Total P&L        : {'+' if total_pnl >= 0 else ''}{total_pnl:.2f}%")
            else:
                status = "LONG (open position)" if state.position == Position.LONG else "FLAT"
                print(f"  No completed trades this session  |  Status: {status}")
        print(f"{'='*60}\n")


# ----------------------------------- Entry point ----------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Live trading strategy monitor.")
    parser.add_argument("--ticker", nargs="+", default=TICKERS, metavar="TICKER")
    parser.add_argument("--interval", type=int, default=5, metavar="MINUTES")
    parser.add_argument("--no-market-check", action="store_true")
    parser.add_argument(
        "--email",
        action="store_true",
        help="Send email alerts on BUY/SELL signals. "
             "Requires SMTP_USER, SMTP_PASSWORD, and ALERT_TO env vars.",
    )
    parser.add_argument(
        "--alpaca",
        action="store_true",
        help="Submit orders to Alpaca on BUY/SELL signals. "
             "Requires ALPACA_API_KEY and ALPACA_SECRET_KEY in .env. "
             "Defaults to paper trading.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live Alpaca account instead of paper. "
             "Only valid with --alpaca. Real money will be traded.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_monitor(
        tickers=[t.upper() for t in args.ticker],
        interval_minutes=args.interval,
        skip_market_check=args.no_market_check,
        email_alerts=args.email,
        alpaca_trading=args.alpaca,
        alpaca_live=args.live,
    )


if __name__ == "__main__":
    main()
