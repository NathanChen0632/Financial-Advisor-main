from __future__ import annotations

import os
import math
from dataclasses import dataclass
from datetime import datetime

from stock_prediction.utils import GREEN, RED, YELLOW, BOLD, RESET  # noqa: F401 (re-exported)

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        TrailingStopOrderRequest,
    )
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.models import Position as AlpacaPosition
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False



@dataclass
class AlpacaConfig:
    api_key:    str
    secret_key: str
    paper:      bool = True

    @classmethod
    def from_env(cls) -> "AlpacaConfig | None":
        api_key    = os.environ.get("ALPACA_API_KEY")
        secret_key = os.environ.get("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            missing = [k for k, v in [("ALPACA_API_KEY", api_key), ("ALPACA_SECRET_KEY", secret_key)] if not v]
            print(f"{YELLOW}[ALPACA] Missing env vars: {', '.join(missing)} — trading disabled.{RESET}")
            return None

        paper = os.environ.get("ALPACA_PAPER", "true").lower() != "false"
        return cls(api_key=api_key, secret_key=secret_key, paper=paper)


class AlpacaTrader:
    # Wraps the Alpaca API for order execution and account management.
    #
    # Key improvements over v1:
    #   - risk_pct raised to 2% so positions are meaningful relative to account size
    #   - Trailing stop replaces fixed stop — lets winners run instead of cutting at ATR target
    #   - sync_positions_from_alpaca() restores state after a restart so no double-entries
    #   - Open positions printed on startup for full visibility

    def __init__(
        self,
        config:            AlpacaConfig,
        risk_pct:          float = 0.02,   # 2% of equity risked per trade (up from 1%)
        max_position_pct:  float = 0.25,   # max 25% of equity in any one stock
        trail_pct:         float = 2.5,    # trailing stop trails 2.5% below running high
    ):
        if not _ALPACA_AVAILABLE:
            raise ImportError("alpaca-py not installed. Run: pip install alpaca-py")

        self.cfg             = config
        self.risk_pct        = risk_pct
        self.max_position_pct = max_position_pct
        self.trail_pct       = trail_pct
        self.paper           = config.paper

        self.client = TradingClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
            paper=config.paper,
        )

        mode = f"{YELLOW}PAPER{RESET}" if config.paper else f"{RED}{BOLD}LIVE{RESET}"
        print(f"\n  {GREEN}[ALPACA] Connected — {mode} trading{RESET}")
        acct = self.client.get_account()
        print(f"  [ALPACA] Equity       : ${float(acct.equity):,.2f}")
        print(f"  [ALPACA] Buying Power : ${float(acct.buying_power):,.2f}")
        print(f"  [ALPACA] Risk/trade   : {risk_pct*100:.1f}%  |  Max position: {max_position_pct*100:.0f}%  |  Trail: {trail_pct:.1f}%")
        self._print_open_positions()

    def _equity(self) -> float:
        return float(self.client.get_account().equity)

    def _buying_power(self) -> float:
        return float(self.client.get_account().buying_power)

    def _print_open_positions(self):
        try:
            positions = self.client.get_all_positions()
            if not positions:
                print(f"  [ALPACA] No open positions")
                return
            print(f"  [ALPACA] Open positions:")
            for p in positions:
                pnl     = float(p.unrealized_pl)
                pnl_pct = float(p.unrealized_plpc) * 100
                sign    = "+" if pnl >= 0 else ""
                col     = GREEN if pnl >= 0 else RED
                print(
                    f"    {p.symbol:<6}  {p.qty} shares @ ${float(p.avg_entry_price):.2f}"
                    f"  current ${float(p.current_price):.2f}"
                    f"  {col}{sign}${pnl:.2f} ({sign}{pnl_pct:.1f}%){RESET}"
                )
        except Exception:
            pass

    def _calc_shares(self, price: float, stop_price: float) -> int:
        stop_dist = price - stop_price

        if stop_dist <= 0:
            print(f"  {YELLOW}[ALPACA] Invalid stop distance — order skipped.{RESET}")
            return 0

        # One account fetch covers both equity and buying power.
        acct          = self.client.get_account()
        equity        = float(acct.equity)
        buying_power  = float(acct.buying_power)

        # Size so exactly risk_pct of equity is lost if stop is hit
        risk_shares = math.floor((equity * self.risk_pct) / stop_dist)

        # Hard cap: never put more than max_position_pct of equity in one name
        cap_shares  = math.floor((equity * self.max_position_pct) / price)

        shares = min(risk_shares, cap_shares)
        shares = min(shares, math.floor(buying_power / price))
        return max(shares, 0)

    def get_position(self, ticker: str) -> "AlpacaPosition | None":
        try:
            return self.client.get_open_position(ticker)
        except Exception:
            return None

    def has_position(self, ticker: str) -> bool:
        return self.get_position(ticker) is not None

    def sync_positions_from_alpaca(self, tickers: list[str]) -> dict:
        # On startup or after a restart, pull open positions from Alpaca and
        # return a dict of ticker → position info so the monitor can restore
        # StrategyState without re-entering trades that are already open.
        restored = {}
        try:
            positions = self.client.get_all_positions()
            pos_map   = {p.symbol: p for p in positions}

            for ticker in tickers:
                if ticker in pos_map:
                    p = pos_map[ticker]
                    entry = float(p.avg_entry_price)
                    curr  = float(p.current_price)
                    pnl   = float(p.unrealized_plpc) * 100
                    sign  = "+" if pnl >= 0 else ""

                    restored[ticker] = {
                        "entry_price": entry,
                        "qty":         int(float(p.qty)),
                        "current":     curr,
                        "pnl_pct":     pnl,
                    }
                    print(
                        f"  {YELLOW}[ALPACA] Restored {ticker}: "
                        f"{p.qty} shares @ ${entry:.2f}  "
                        f"({sign}{pnl:.1f}%){RESET}"
                    )
        except Exception as e:
            print(f"  {YELLOW}[ALPACA] Could not sync positions: {e}{RESET}")

        return restored

    def submit_buy(
        self,
        ticker:     str,
        price:      float,
        stop_price: float,
        qty:        int | None = None,
    ) -> bool:
        if self.has_position(ticker):
            print(f"  {YELLOW}[ALPACA] Already in {ticker} — skipping buy.{RESET}")
            return False

        # Clear any orphaned orders before entering — prevents wash trade rejection
        try:
            self.client.cancel_orders_for_symbol(ticker)
        except Exception:
            pass

        # An explicit qty (e.g. a UI order for "10 shares") overrides the risk
        # model, but is still capped by available buying power. With no qty,
        # size the position by risk_pct as usual.
        if qty and qty > 0:
            shares = min(int(qty), math.floor(self._buying_power() / price))
        else:
            shares = self._calc_shares(price, stop_price)
        if shares == 0:
            print(f"  {YELLOW}[ALPACA] Position size 0 shares — skipping.{RESET}")
            return False

        try:
            # Market buy
            order = self.client.submit_order(MarketOrderRequest(
                symbol=ticker,
                qty=shares,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            ))
            print(f"  {GREEN}[ALPACA] BUY — {shares} shares {ticker} @ ~${price:.2f}{RESET}")
            print(f"  [ALPACA] Order ID : {order.id}")

            # Trailing stop replaces the fixed stop loss.
            # It starts at trail_pct% below current price and moves up
            # as price rises, locking in gains rather than exiting at a
            # fixed target — lets winners run while still capping losses.
            trail_order = self.client.submit_order(TrailingStopOrderRequest(
                symbol=ticker,
                qty=shares,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                trail_percent=self.trail_pct,
            ))
            print(f"  [ALPACA] Trailing stop set at {self.trail_pct:.1f}% below running high (order {trail_order.id})")

            dollar_risk = shares * (price - stop_price)
            print(f"  [ALPACA] Max risk: ${dollar_risk:.2f}  ({self.risk_pct*100:.1f}% of equity)")
            return True

        except Exception as e:
            print(f"  {RED}[ALPACA] Buy order failed: {e}{RESET}")
            return False

    def submit_sell(self, ticker: str, reason: str = "signal") -> bool:
        if not self.has_position(ticker):
            print(f"  {YELLOW}[ALPACA] No position in {ticker} — nothing to sell.{RESET}")
            return False

        try:
            # Capture P&L *before* closing — once the position is closed
            # get_position() returns None and the P&L is lost.
            pos = self.get_position(ticker)

            self.client.cancel_orders_for_symbol(ticker)
            self.client.close_position(ticker)

            if pos:
                pnl     = float(pos.unrealized_pl)
                pnl_pct = float(pos.unrealized_plpc) * 100
                sign    = "+" if pnl >= 0 else ""
                col     = GREEN if pnl >= 0 else RED
                print(
                    f"  {col}[ALPACA] SELL — {ticker}  "
                    f"P&L: {sign}${pnl:.2f} ({sign}{pnl_pct:.2f}%)  "
                    f"Reason: {reason}{RESET}"
                )
            else:
                print(f"  {RED}[ALPACA] SELL submitted — {ticker}  Reason: {reason}{RESET}")
            return True

        except Exception as e:
            print(f"  {RED}[ALPACA] Sell failed: {e}{RESET}")
            return False

    def account_summary(self) -> dict:
        acct = self.client.get_account()
        return {
            "equity":       float(acct.equity),
            "cash":         float(acct.cash),
            "buying_power": float(acct.buying_power),
            "pnl_today":    float(acct.equity) - float(acct.last_equity),
        }

    def print_account_summary(self):
        s    = self.account_summary()
        sign = "+" if s["pnl_today"] >= 0 else ""
        col  = GREEN if s["pnl_today"] >= 0 else RED
        print(f"\n  [ALPACA] Equity: ${s['equity']:,.2f}  "
              f"Cash: ${s['cash']:,.2f}  "
              f"Today P&L: {col}{sign}${s['pnl_today']:.2f}{RESET}")
