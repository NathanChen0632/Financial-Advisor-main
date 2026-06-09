"""
Shared fixtures for the DQN trading system test suite.

All fixtures use fully synthetic, deterministic data so the tests run
offline without yfinance downloads and produce the same results every time.
"""
from __future__ import annotations

import sys
import os

# Ensure backend/ is on the path so stock_prediction is importable
# regardless of which directory pytest is invoked from.
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import numpy as np
import pytest

from stock_prediction.rl_agent import TradingEnv, TradingConfig, DQNAgent, train_dqn_agent


# ---------------------------------------------------------------------------
# Minimal feature column set that matches what TradingEnv looks for
# ---------------------------------------------------------------------------

# TradingEnv only cares about three specific column names.
# The remaining columns can be anything — we pad with generic names.
_REQUIRED_COLS = ["atr14_pct", "volume_ratio", "ma20_ratio"]
N_FEATURES = 10  # small, for fast tests

FEATURE_COLS: list[str] = _REQUIRED_COLS + [f"feat_{i}" for i in range(N_FEATURES - len(_REQUIRED_COLS))]


def make_synthetic_data(
    n_steps: int = 100,
    start_price: float = 100.0,
    drift: float = 0.0005,
    vol: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (features, daily_returns, prices) with controllable drift/vol.

    features[:, 0]  = atr14_pct  — fixed at 0.015 (realistic ATR)
    features[:, 1]  = volume_ratio — fixed at 1.0  (average volume)
    features[:, 2]  = ma20_ratio   — fixed at 1.0  (at MA20)
    features[:, 3:] = random noise
    """
    rng = np.random.default_rng(seed)

    # Build a realistic price series
    log_returns = rng.normal(drift, vol, n_steps)
    prices = start_price * np.cumprod(np.exp(log_returns))
    daily_returns = np.exp(log_returns) - 1.0

    # Feature matrix
    features = rng.standard_normal((n_steps, N_FEATURES)).astype(np.float32) * 0.1
    features[:, 0] = 0.015   # atr14_pct — ~1.5% ATR (2% stop distance with mult=2)
    features[:, 1] = 1.0     # volume_ratio — at average
    features[:, 2] = 1.0     # ma20_ratio   — at MA20 support

    return features, daily_returns, prices


@pytest.fixture
def base_data():
    """Canonical 100-step synthetic dataset."""
    return make_synthetic_data(n_steps=100, seed=42)


@pytest.fixture
def env(base_data):
    """Fresh TradingEnv reset and ready for each test."""
    features, daily_returns, prices = base_data
    cfg = TradingConfig(max_holding_days=10)
    e = TradingEnv(features, daily_returns, prices, FEATURE_COLS, config=cfg)
    e.reset()
    return e


@pytest.fixture
def long_env():
    """
    Env with 500 steps and an upward-trending price series.
    Used for integration tests where we need enough room to enter + exit multiple trades.
    """
    features, daily_returns, prices = make_synthetic_data(n_steps=500, drift=0.001, seed=7)
    cfg = TradingConfig(max_holding_days=10)
    e = TradingEnv(features, daily_returns, prices, FEATURE_COLS, config=cfg)
    e.reset()
    return e


@pytest.fixture
def agent(base_data):
    """Freshly initialised (untrained) DQNAgent with correct state size."""
    features, _, _ = base_data
    state_size = features.shape[1] + 5  # N_FEATURES + 5 context dims
    return DQNAgent(state_size=state_size, hidden_dim=32, random_state=0)


@pytest.fixture
def trained_agent(base_data):
    """
    DQNAgent trained for a small number of episodes on synthetic data.
    Fast enough for unit tests (~3s); enough to produce a learned policy.
    """
    features, daily_returns, prices = make_synthetic_data(n_steps=200, seed=99)
    agent = train_dqn_agent(
        X_train=features,
        daily_returns_train=daily_returns,
        prices_train=prices,
        feature_cols=FEATURE_COLS,
        n_episodes=30,
        hidden_dim=32,
        random_state=0,
    )
    return agent, features, daily_returns, prices
