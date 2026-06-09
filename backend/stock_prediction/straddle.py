from __future__ import annotations

# Long straddle options strategy.
#
# Buys an ATM call + ATM put at the same strike and expiration.
# Profits when the underlying makes a large move in either direction.
# Ideal entry: IV is relatively low (cheap options) and a big move is expected.
#
# Entry filters
#   - 14–42 DTE (enough time for the move, not too much theta burn)
#   - IV 30–65%  (affordable premiums; avoid buying after a vol spike)
#   - |delta| 0.10–0.50 (near-the-money, not deep ITM/OTM)
#   - Open interest > 100 (liquid contracts only)
#   - Total premium ≤ 5% of account buying power
#
# Exit rules
#   - Target profit : 40% gain on total premium paid
#   - Stop loss     : 45% loss on total premium paid
#   - Time stop     : close when ≤ 5 DTE (theta decay accelerates)

import os
import math
import time
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from stock_prediction.utils import GREEN, RED, YELLOW, BOLD, RESET

try:
    from scipy.stats import norm
    from scipy.optimize import brentq
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.historical.stock import StockHistoricalDataClient, StockLatestTradeRequest
    from alpaca.data.requests import OptionLatestQuoteRequest
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        GetOptionContractsRequest,
        MarketOrderRequest,
        OptionLegRequest,
    )
    from alpaca.trading.enums import (
        AssetStatus,
        ContractType,
        OrderClass,
        OrderSide,
        OrderStatus,
        TimeInForce,
    )
    _ALPACA_OPTIONS_AVAILABLE = True
except ImportError:
    _ALPACA_OPTIONS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Black-Scholes: implied volatility and Greeks
# ---------------------------------------------------------------------------

def _bs_price(S: float, K: float, T: float, r: float, sigma: float, opt_type: str) -> float:
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt_type == "call":
        return S * float(norm.cdf(d1)) - K * math.exp(-r * T) * float(norm.cdf(d2))
    return K * math.exp(-r * T) * float(norm.cdf(-d2)) - S * float(norm.cdf(-d1))


def calculate_iv(option_price: float, S: float, K: float, T: float,
                 r: float, opt_type: str) -> float | None:
    # Numerical root-finding (Brent's method) to invert the Black-Scholes formula.
    # Returns None when IV cannot be determined reliably.
    if not _SCIPY_AVAILABLE:
        return None
    if T <= 0:
        return None

    intrinsic = max(0.0, (S - K) if opt_type == "call" else (K - S))
    if option_price <= intrinsic + 1e-6:
        return 0.0

    try:
        iv = brentq(lambda s: _bs_price(S, K, T, r, s, opt_type) - option_price,
                    1e-6, 5.0, xtol=1e-6)
        return float(iv)
    except (ValueError, RuntimeError):
        return None


def calculate_greeks(option_price: float, S: float, K: float,
                     T: float, r: float, opt_type: str) -> dict | None:
    # Returns delta, gamma, theta (daily), vega (per 1% IV move).
    # Uses BS IV internally so all greeks are IV-consistent.
    if not _SCIPY_AVAILABLE or T <= 0:
        return None

    iv = calculate_iv(option_price, S, K, T, r, opt_type)
    if not iv:
        return None

    d1 = (math.log(S / K) + (r + 0.5 * iv**2) * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)

    delta = float(norm.cdf(d1))  if opt_type == "call" else float(-norm.cdf(-d1))
    gamma = float(norm.pdf(d1)) / (S * iv * math.sqrt(T))
    vega  = S * math.sqrt(T) * float(norm.pdf(d1)) / 100

    if opt_type == "call":
        theta = (
            -(S * float(norm.pdf(d1)) * iv) / (2 * math.sqrt(T))
            - r * K * math.exp(-r * T) * float(norm.cdf(d2))
        ) / 365
    else:
        theta = (
            -(S * float(norm.pdf(d1)) * iv) / (2 * math.sqrt(T))
            + r * K * math.exp(-r * T) * float(norm.cdf(-d2))
        ) / 365

    return {"iv": iv, "delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


# ---------------------------------------------------------------------------
# Volatility screening — Bollinger Bands + RVI
# These decide whether conditions favour entering a new straddle.
# ---------------------------------------------------------------------------

def _bollinger_position(prices: pd.Series, period: int = 14, mult: float = 2.0) -> float:
    # Returns how far price is from the band midpoint, normalised 0–1.
    # Values near 0 or 1 mean price is at the lower or upper band — elevated volatility.
    sma = prices.rolling(period).mean().iloc[-1]
    std = prices.rolling(period).std().iloc[-1]
    upper = sma + mult * std
    lower = sma - mult * std
    rng   = upper - lower
    return float((prices.iloc[-1] - lower) / rng) if rng > 0 else 0.5


def _rvi(prices: pd.Series, period: int = 21) -> float:
    # Relative Volatility Index — RSI applied to standard deviations rather
    # than price changes. 40–60 signals neutral-to-rising volatility, a
    # good regime for entering a long straddle.
    chg  = prices.diff()
    up   = chg.where(chg > 0, 0.0).rolling(period).std()
    down = (-chg.where(chg < 0, 0.0)).rolling(period).std()
    rs   = up / down.replace(0, float("nan"))
    rvi  = (100 - 100 / (1 + rs)).rolling(4).mean()
    val  = rvi.iloc[-1]
    return float(val) if not math.isnan(val) else 50.0


# ---------------------------------------------------------------------------
# Alpaca option chain helpers
# ---------------------------------------------------------------------------

def _get_underlying_price(symbol: str, stock_client: "StockHistoricalDataClient") -> float:
    req  = StockLatestTradeRequest(symbol_or_symbols=symbol)
    resp = stock_client.get_stock_latest_trade(req)
    return float(resp[symbol].price)


def _option_mid(symbol: str, opt_client: "OptionHistoricalDataClient") -> float | None:
    try:
        req   = OptionLatestQuoteRequest(symbol_or_symbols=symbol)
        quote = opt_client.get_option_latest_quote(req)[symbol]
        bid, ask = float(quote.bid_price), float(quote.ask_price)
        return (bid + ask) / 2 if bid > 0 and ask > 0 else None
    except Exception:
        return None


def _fetch_option_chain(
    symbol: str,
    trade_client: "TradingClient",
    min_expiry: date,
    max_expiry: date,
    min_strike: float,
    max_strike: float,
    contract_type: "ContractType",
) -> list:
    req = GetOptionContractsRequest(
        underlying_symbols=[symbol],
        status=AssetStatus.ACTIVE,
        type=contract_type,
        strike_price_gte=str(min_strike),
        strike_price_lte=str(max_strike),
        expiration_date_gte=min_expiry,
        expiration_date_lte=max_expiry,
    )
    return trade_client.get_option_contracts(req).option_contracts


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

def _build_candidate(
    contract,
    underlying_price: float,
    opt_client: "OptionHistoricalDataClient",
    risk_free_rate: float,
    oi_threshold: int,
) -> dict | None:
    # Validate open interest
    if contract.open_interest is None or float(contract.open_interest) < oi_threshold:
        return None

    mid = _option_mid(contract.symbol, opt_client)
    if mid is None or mid <= 0:
        return None

    K   = float(contract.strike_price)
    exp = pd.Timestamp(contract.expiration_date)
    T   = max((exp - pd.Timestamp.now()).days / 365, 1e-6)
    otype = contract.type.value  # "call" or "put"

    greeks = calculate_greeks(mid, underlying_price, K, T, risk_free_rate, otype)
    if greeks is None:
        return None

    return {
        "symbol":       contract.symbol,
        "strike":       K,
        "expiration":   exp,
        "dte":          (exp - pd.Timestamp.now()).days,
        "option_type":  otype,
        "price":        mid,
        "oi":           int(float(contract.open_interest)),
        "size":         float(contract.size) if contract.size else 100.0,
        **greeks,
    }


def _passes_filters(cand: dict, iv_range: tuple, delta_range: tuple,
                    dte_range: tuple, theta_range: tuple) -> bool:
    return (
        dte_range[0]   <= cand["dte"]    <= dte_range[1]
        and iv_range[0]    <= cand["iv"]     <= iv_range[1]
        and delta_range[0] <= cand["delta"]  <= delta_range[1]
        and theta_range[0] <= cand["theta"]  <= theta_range[1]
    )


# ---------------------------------------------------------------------------
# Core straddle selection
# ---------------------------------------------------------------------------

ENTRY_CRITERIA = {
    # (dte_range, iv_range, delta_range, theta_range)
    "call": ((14, 42), (0.20, 0.65), (0.10,  0.50), (-0.10, -0.001)),
    "put":  ((14, 42), (0.20, 0.65), (-0.50, -0.10), (-0.10, -0.001)),
}


def find_straddle(
    symbol: str,
    trade_client: "TradingClient",
    opt_client: "OptionHistoricalDataClient",
    stock_client: "StockHistoricalDataClient",
    buying_power: float,
    risk_free_rate: float = 0.01,
    strike_range_pct: float = 0.05,
    bp_limit_pct: float = 0.05,
    oi_threshold: int = 100,
) -> tuple[dict, dict] | None:
    # Returns (put_leg, call_leg) dicts or None if no valid straddle exists.

    underlying = _get_underlying_price(symbol, stock_client)
    min_strike = underlying * (1 - strike_range_pct)
    max_strike = underlying * (1 + strike_range_pct)
    today      = date.today()
    min_exp    = today + timedelta(days=14)
    max_exp    = today + timedelta(days=60)

    calls_raw = _fetch_option_chain(symbol, trade_client, min_exp, max_exp,
                                    min_strike, max_strike, ContractType.CALL)
    puts_raw  = _fetch_option_chain(symbol, trade_client, min_exp, max_exp,
                                    min_strike, max_strike, ContractType.PUT)

    # Build and filter candidates for each leg
    calls, puts = [], []
    for c in calls_raw:
        cand = _build_candidate(c, underlying, opt_client, risk_free_rate, oi_threshold)
        if cand and _passes_filters(cand, *ENTRY_CRITERIA["call"]):
            calls.append(cand)
    for p in puts_raw:
        cand = _build_candidate(p, underlying, opt_client, risk_free_rate, oi_threshold)
        if cand and _passes_filters(cand, *ENTRY_CRITERIA["put"]):
            puts.append(cand)

    if not calls or not puts:
        return None

    # Group by expiration and pick the expiration with the most candidates on both sides
    call_by_exp = {}
    for c in calls:
        call_by_exp.setdefault(c["expiration"], []).append(c)
    put_by_exp = {}
    for p in puts:
        put_by_exp.setdefault(p["expiration"], []).append(p)

    common = set(call_by_exp) & set(put_by_exp)
    if not common:
        return None

    best_exp = max(common, key=lambda e: len(call_by_exp[e]) + len(put_by_exp[e]))

    # Within the chosen expiration, find a common strike closest to ATM
    call_strikes = {c["strike"] for c in call_by_exp[best_exp]}
    put_strikes  = {p["strike"] for p in put_by_exp[best_exp]}
    common_strikes = call_strikes & put_strikes
    if not common_strikes:
        return None

    atm_strike = min(common_strikes, key=lambda k: abs(k - underlying))

    chosen_call = next(c for c in call_by_exp[best_exp] if c["strike"] == atm_strike)
    chosen_put  = next(p for p in put_by_exp[best_exp]  if p["strike"] == atm_strike)

    # Buying power check — never risk more than bp_limit_pct of account
    contract_size = chosen_call["size"]
    total_cost    = (chosen_call["price"] + chosen_put["price"]) * contract_size
    if total_cost > buying_power * bp_limit_pct:
        print(f"  {YELLOW}[STRADDLE] {symbol}: premium ${total_cost:.2f} exceeds "
              f"${buying_power * bp_limit_pct:.2f} BP limit — skipping.{RESET}")
        return None

    return chosen_put, chosen_call


# ---------------------------------------------------------------------------
# Order execution
# ---------------------------------------------------------------------------

def place_straddle(
    symbol: str,
    put_leg: dict,
    call_leg: dict,
    trade_client: "TradingClient",
    qty: int = 1,
) -> dict | None:
    # Submits a multi-leg market order with one put and one call.
    legs = [
        OptionLegRequest(symbol=put_leg["symbol"],  side=OrderSide.BUY, ratio_qty=1),
        OptionLegRequest(symbol=call_leg["symbol"], side=OrderSide.BUY, ratio_qty=1),
    ]
    req = MarketOrderRequest(
        qty=qty,
        order_class=OrderClass.MLEG,
        time_in_force=TimeInForce.DAY,
        legs=legs,
    )
    try:
        order = trade_client.submit_order(req)
        total_debit = (put_leg["price"] + call_leg["price"]) * put_leg["size"] * qty
        print(
            f"  {GREEN}[STRADDLE] Opened — {symbol}  "
            f"strike ${put_leg['strike']:.2f}  "
            f"exp {put_leg['expiration'].date()}  "
            f"debit ~${total_debit:.2f}  "
            f"order {order.id}{RESET}"
        )
        return {
            "order_id":     str(order.id),
            "symbol":       symbol,
            "put_symbol":   put_leg["symbol"],
            "call_symbol":  call_leg["symbol"],
            "strike":       put_leg["strike"],
            "expiration":   put_leg["expiration"],
            "entry_debit":  total_debit,
            "entry_time":   datetime.now().isoformat(),
            "qty":          qty,
            "contract_size": put_leg["size"],
        }
    except Exception as e:
        print(f"  {RED}[STRADDLE] Order failed for {symbol}: {e}{RESET}")
        return None


# ---------------------------------------------------------------------------
# Position monitoring and exit
# ---------------------------------------------------------------------------

@dataclass
class StraddlePosition:
    order_id:      str
    symbol:        str
    put_symbol:    str
    call_symbol:   str
    strike:        float
    expiration:    pd.Timestamp
    entry_debit:   float
    entry_time:    str
    qty:           int   = 1
    contract_size: float = 100.0

    @classmethod
    def from_dict(cls, d: dict) -> "StraddlePosition":
        return cls(
            order_id      = d["order_id"],
            symbol        = d["symbol"],
            put_symbol    = d["put_symbol"],
            call_symbol   = d["call_symbol"],
            strike        = d["strike"],
            expiration    = pd.Timestamp(d["expiration"]),
            entry_debit   = d["entry_debit"],
            entry_time    = d["entry_time"],
            qty           = d.get("qty", 1),
            contract_size = d.get("contract_size", 100.0),
        )

    @property
    def dte(self) -> int:
        return max((self.expiration - pd.Timestamp.now()).days, 0)


def _current_straddle_value(pos: "StraddlePosition",
                             opt_client: "OptionHistoricalDataClient") -> float | None:
    # Current mark-to-market value = sum of midpoints × contract_size × qty
    put_mid  = _option_mid(pos.put_symbol,  opt_client)
    call_mid = _option_mid(pos.call_symbol, opt_client)
    if put_mid is None or call_mid is None:
        return None
    return (put_mid + call_mid) * pos.contract_size * pos.qty


def check_exit(
    pos: "StraddlePosition",
    opt_client: "OptionHistoricalDataClient",
    target_pct: float = 0.40,
    stop_pct: float   = 0.45,
    min_dte: int      = 5,
) -> tuple[bool, str]:
    # Returns (should_exit, reason).
    if pos.dte <= min_dte:
        return True, f"DTE={pos.dte} ≤ {min_dte} (theta accelerates near expiry)"

    current = _current_straddle_value(pos, opt_client)
    if current is None:
        return False, ""

    pnl_pct = (current - pos.entry_debit) / pos.entry_debit

    if pnl_pct >= target_pct:
        return True, f"target hit +{pnl_pct*100:.1f}% (≥ {target_pct*100:.0f}%)"
    if pnl_pct <= -stop_pct:
        return True, f"stop loss {pnl_pct*100:.1f}% (≤ -{stop_pct*100:.0f}%)"
    return False, ""


def close_straddle(pos: "StraddlePosition", trade_client: "TradingClient") -> bool:
    # Closes both legs by liquidating the Alpaca options positions.
    closed = 0
    for sym in [pos.put_symbol, pos.call_symbol]:
        try:
            trade_client.close_position(sym)
            closed += 1
        except Exception as e:
            print(f"  {YELLOW}[STRADDLE] Could not close {sym}: {e}{RESET}")
    if closed > 0:
        print(f"  {RED}[STRADDLE] Closed {pos.symbol} straddle ({closed}/2 legs){RESET}")
    return closed == 2


# ---------------------------------------------------------------------------
# High-level monitor interface
# ---------------------------------------------------------------------------

class StraddleMonitor:
    # Manages the full lifecycle: screening → entry → monitoring → exit.
    # Call poll() on every monitoring interval.

    TARGET_PROFIT = 0.40   # close when straddle value is up 40%
    STOP_LOSS     = 0.45   # close when straddle value is down 45%
    MIN_DTE       = 5      # close when ≤ 5 days remain
    BP_LIMIT_PCT  = 0.05   # max 5% of buying power per straddle
    RVI_LOW       = 35     # only enter when RVI is below this (volatility not already spiked)
    RVI_HIGH      = 65

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        if not _ALPACA_OPTIONS_AVAILABLE:
            raise ImportError("alpaca-py not installed or missing options support. "
                              "Run: pip install alpaca-py")
        if not _SCIPY_AVAILABLE:
            raise ImportError("scipy not installed. Run: pip install scipy")

        self.trade_client = TradingClient(api_key=api_key, secret_key=secret_key, paper=paper)
        self.opt_client   = OptionHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        self.stock_client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        self.positions: list[StraddlePosition] = []

        mode = f"{YELLOW}PAPER{RESET}" if paper else f"{RED}{BOLD}LIVE{RESET}"
        print(f"  {GREEN}[STRADDLE] Monitor initialised — {mode}{RESET}")

    def _buying_power(self) -> float:
        return float(self.trade_client.get_account().buying_power)

    def _volatility_ok(self, symbol: str, prices: pd.Series) -> bool:
        # Only enter a straddle when IV environment is not already elevated.
        # We use RVI as a cheap proxy: avoid entering after a vol spike.
        rvi_val = _rvi(prices)
        bp_pos  = _bollinger_position(prices)
        print(f"    {symbol}: RVI={rvi_val:.1f}  BB_pos={bp_pos:.2f}")
        return (self.RVI_LOW <= rvi_val <= self.RVI_HIGH) or (bp_pos < 0.15 or bp_pos > 0.85)

    def poll(self, tickers: list[str], price_history: dict[str, pd.Series]):
        # 1. Monitor existing positions first
        for pos in list(self.positions):
            should_exit, reason = check_exit(
                pos, self.opt_client,
                target_pct=self.TARGET_PROFIT,
                stop_pct=self.STOP_LOSS,
                min_dte=self.MIN_DTE,
            )
            if should_exit:
                close_straddle(pos, self.trade_client)
                self.positions.remove(pos)
                print(f"  [STRADDLE] Exited {pos.symbol} — {reason}")

        # 2. Screen for new entries
        open_symbols = {p.symbol for p in self.positions}
        bp = self._buying_power()

        for symbol in tickers:
            if symbol in open_symbols:
                continue   # already have a straddle on this one

            prices = price_history.get(symbol)
            if prices is None or len(prices) < 30:
                continue

            # Skip if volatility is already elevated — premiums too expensive
            if not self._volatility_ok(symbol, prices):
                print(f"  [STRADDLE] {symbol}: volatility regime unfavourable — skipping")
                continue

            print(f"  [STRADDLE] Screening {symbol} for straddle entry...")
            result = find_straddle(
                symbol=symbol,
                trade_client=self.trade_client,
                opt_client=self.opt_client,
                stock_client=self.stock_client,
                buying_power=bp,
                bp_limit_pct=self.BP_LIMIT_PCT,
            )
            if result is None:
                print(f"  [STRADDLE] {symbol}: no suitable options found")
                continue

            put_leg, call_leg = result
            pos_dict = place_straddle(symbol, put_leg, call_leg, self.trade_client)
            if pos_dict:
                self.positions.append(StraddlePosition.from_dict(pos_dict))
                open_symbols.add(symbol)

    def print_status(self):
        if not self.positions:
            print(f"  [STRADDLE] No open straddle positions")
            return
        print(f"  [STRADDLE] Open straddles:")
        for pos in self.positions:
            current = _current_straddle_value(pos, self.opt_client)
            if current is not None:
                pnl     = current - pos.entry_debit
                pnl_pct = pnl / pos.entry_debit * 100
                sign    = "+" if pnl >= 0 else ""
                col     = GREEN if pnl >= 0 else RED
                print(
                    f"    {pos.symbol:<6}  strike ${pos.strike:.2f}  "
                    f"exp {pos.expiration.date()}  DTE={pos.dte}  "
                    f"{col}{sign}${pnl:.2f} ({sign}{pnl_pct:.1f}%){RESET}"
                )
            else:
                print(f"    {pos.symbol:<6}  strike ${pos.strike:.2f}  "
                      f"exp {pos.expiration.date()}  DTE={pos.dte}  (quote unavailable)")
