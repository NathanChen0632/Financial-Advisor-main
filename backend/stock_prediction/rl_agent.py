from __future__ import annotations

import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass
from typing import Tuple

# Optional Gymnasium integration.
# When available the environments inherit from gymnasium.Env and expose standard
# observation_space / action_space attributes, making them compatible with
# third-party RL libraries and testable via the Gymnasium API checker.
try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:
    gym = None          # type: ignore[assignment]
    spaces = None       # type: ignore[assignment]
    _GYM_AVAILABLE = False

_EnvBase: type = gym.Env if _GYM_AVAILABLE else object


@dataclass
class TradingConfig:
    # All trading strategy parameters in one place.
    account_size:         float = 10_000.0  # starting capital ($)
    risk_per_trade:       float = 0.01      # max 1% of account risked per trade
    min_rr_ratio:         float = 2.0       # only enter if target is 2x the stop distance
    max_holding_days:     int   = 10        # forced exit after N days to prevent dead money
    stop_atr_mult:        float = 2.0       # stop placed at 2x ATR below entry
    min_volume_ratio:     float = 0.8       # require at least 80% of average volume to confirm
    ma_proximity_bonus:   float = 0.001     # bonus for entering near MA20 (pullback setup)
    time_decay_start:     int   = 5         # holding penalty begins after this many days
    time_decay_rate:      float = 0.0001    # reward reduction per extra day held
    transaction_cost:     float = 0.001     # 0.10% per trade (realistic commission + spread)
    slippage:             float = 0.0005    # 0.05% slippage per side on execution
    max_position_size:    float = 5.0       # cap on volatility-based sizing (leverage ceiling)
    high_vol_regime_cut:  float = 0.70      # vol_regime percentile above which regime penalty kicks in
    vol_regime_penalty:   float = 0.003     # per-day penalty for holding in a high-vol regime


class TradingEnv(_EnvBase):
    # Simulates a real trading account with disciplined rules baked into the reward.
    # The agent can't just buy and hope, it has to respect stops, targets, volume,
    # and position-sizing constraints to earn positive rewards.
    #
    # Implements the Gymnasium Env API when gymnasium is installed:
    #   reset() → (obs, info)
    #   step()  → (obs, reward, terminated, truncated, info)

    metadata = {"render_modes": []}

    def __init__(
        self,
        feature_array:   np.ndarray,
        daily_returns:   np.ndarray,
        prices:          np.ndarray,
        feature_cols:    list,
        config:          TradingConfig | None = None,
        scaled_features: np.ndarray | None   = None,
    ):
        super().__init__()

        # Unscaled features are used for trading logic (ATR math, volume checks)
        # where original values matter. Scaled features go into the Q-network.
        self.features        = feature_array
        self.scaled_features = scaled_features if scaled_features is not None else feature_array
        self.daily_returns   = daily_returns
        self.prices          = prices
        self.feature_cols    = feature_cols
        self.cfg             = config or TradingConfig()

        # Pre-compute column indices so step() can look up specific raw features
        # (ATR for stop sizing, volume for entry confirmation, regime for reward shaping)
        # without passing extra arrays around.
        self._atr_idx         = feature_cols.index("atr14_pct")   if "atr14_pct"    in feature_cols else None
        self._vol_idx         = feature_cols.index("volume_ratio") if "volume_ratio" in feature_cols else None
        self._ma20_idx        = feature_cols.index("ma20_ratio")   if "ma20_ratio"   in feature_cols else None
        self._vol_regime_idx  = feature_cols.index("vol_regime")   if "vol_regime"   in feature_cols else None
        self._adx_idx         = feature_cols.index("adx14")        if "adx14"        in feature_cols else None

        self.n_steps    = len(feature_array)
        self.n_features = feature_array.shape[1]
        # 5 extra context dims: position, days held, unrealized P&L, stop distance, target distance
        self.state_size = self.n_features + 5

        # Gymnasium spaces — enables compatibility with RL libraries and the
        # gymnasium.utils.env_checker validation tool.
        if _GYM_AVAILABLE:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.state_size,),
                dtype=np.float32,
            )
            self.action_space = spaces.Discrete(2)

        # Initialise episode state directly (no self.reset() call) so __init__
        # is side-effect-free and gymnasium env_checker is happy.
        self.t            = 0
        self.position     = 0
        self.entry_price  = 0.0
        self.stop_price   = 0.0
        self.target_price = 0.0
        self.days_held    = 0
        self.position_size = 0.0   # volatility-based size, fixed at entry

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """Reset the episode and return (observation, info) per Gymnasium API."""
        if _GYM_AVAILABLE and seed is not None:
            super().reset(seed=seed)
        self.t            = 0
        self.position     = 0
        self.entry_price  = 0.0
        self.stop_price   = 0.0
        self.target_price = 0.0
        self.days_held    = 0
        self.position_size = 0.0
        return self._build_state(), {}

    def _build_state(self) -> np.ndarray:
        # The state gives the agent both market context and trade context.
        # Market features tell it what the stock is doing;
        # trade context (unrealized P&L, stop/target distances) tells it
        # how the current position is performing so it can decide when to exit.
        days_norm = self.days_held / self.cfg.max_holding_days

        if self.position == 1 and self.entry_price > 0:
            curr         = self.prices[self.t]
            unrealized   = (curr - self.entry_price) / self.entry_price
            risk_dist    = self.entry_price - self.stop_price
            stop_dist    = (curr - self.stop_price)   / risk_dist  if risk_dist > 0 else 1.0
            target_range = self.target_price - self.entry_price
            target_dist  = (self.target_price - curr) / target_range if target_range > 0 else 0.0
        else:
            unrealized  = 0.0
            stop_dist   = 1.0
            target_dist = 1.0

        context = np.array([
            float(self.position),
            days_norm,
            unrealized,
            np.clip(stop_dist,   -2.0, 3.0),
            np.clip(target_dist, -1.0, 2.0),
        ])
        return np.concatenate([self.scaled_features[self.t], context])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        cfg        = self.cfg
        curr_price = self.prices[self.t]
        daily_ret  = self.daily_returns[self.t]
        next_price = curr_price * (1.0 + daily_ret)
        reward     = 0.0


        # ------------------------------------------------------------------
        # Regime context — read once per step for reward adjustments.
        # vol_regime is a 0–1 percentile: 1.0 = historically stressed market.
        # adx is 0–1 trend strength: low = choppy, high = trending.
        # Both are raw (unscaled) features because the reward math uses them
        # in their natural units.
        # ------------------------------------------------------------------
        vol_regime = (
            float(self.features[self.t, self._vol_regime_idx])
            if self._vol_regime_idx is not None else 0.5
        )
        adx = (
            float(self.features[self.t, self._adx_idx])
            if self._adx_idx is not None else 0.25
        )

        execution_cost = cfg.transaction_cost + cfg.slippage  # one-way execution friction

        # FLAT — agent wants to BUY
        if self.position == 0 and action == 1:

            # Refuse entry in high-vol, low-trend regimes: these are the market
            # conditions where most false breakouts and stop-hunts occur.
            # The 25-bonus floor on ADX (weak trend) makes the check more lenient
            # when the model is uncertain about trend — we still allow entries but
            # penalise harder when both conditions are unfavourable.
            if vol_regime > cfg.high_vol_regime_cut and adx < 0.25:
                reward -= 0.002  # extra penalty for trying to enter in bad conditions
                # Still allow the entry — let the R/R gate reject truly bad setups

            atr_pct      = self.features[self.t, self._atr_idx] if self._atr_idx is not None else 0.015
            stop_dist    = cfg.stop_atr_mult * atr_pct * curr_price
            stop_price   = curr_price - stop_dist
            target_price = curr_price + cfg.min_rr_ratio * stop_dist
            rr           = (target_price - curr_price) / stop_dist if stop_dist > 0 else 0.0

            if rr >= cfg.min_rr_ratio:
                self.position     = 1
                self.entry_price  = curr_price
                self.stop_price   = stop_price
                self.target_price = target_price
                self.days_held    = 0
                # Volatility-based position size, fixed for the life of the trade.
                # Same formula the backtest uses so training and evaluation agree.
                risk_frac          = (self.entry_price - self.stop_price) / self.entry_price
                self.position_size = min(cfg.risk_per_trade / risk_frac, cfg.max_position_size) if risk_frac > 0 else 1.0
                reward            -= execution_cost * self.position_size  # sized commission + slippage on entry

                if self._vol_idx is not None:
                    vol = self.features[self.t, self._vol_idx]
                    if vol < cfg.min_volume_ratio:
                        reward -= 0.001
                    elif vol > 1.5:
                        reward += 0.0005

                if self._ma20_idx is not None:
                    ma20_ratio = self.features[self.t, self._ma20_idx]
                    if 0.98 <= ma20_ratio <= 1.03:
                        reward += cfg.ma_proximity_bonus
            else:
                reward -= 0.0015


        # LONG, agent wants to SELL (voluntary exit)
        elif self.position == 1 and action == 0:
            pnl    = (curr_price - self.entry_price) / self.entry_price
            reward = (pnl - execution_cost) * self.position_size
            self._reset_position()


        # LONG, agent wants to HOLD
        elif self.position == 1 and action == 1:
            self.days_held += 1
            size = self.position_size

            if next_price <= self.stop_price:
                pnl    = (self.stop_price - self.entry_price) / self.entry_price
                reward = (pnl - execution_cost) * size - 0.002
                self._reset_position()

            elif next_price >= self.target_price:
                pnl    = (self.target_price - self.entry_price) / self.entry_price
                reward = (pnl - execution_cost) * size + 0.002
                self._reset_position()

            elif self.days_held >= cfg.max_holding_days:
                pnl    = (curr_price - self.entry_price) / self.entry_price
                reward = (pnl - execution_cost) * size - 0.001
                self._reset_position()

            else:
                # Normal hold — risk-adjusted daily return.
                # A positive return in a high-vol regime is less impressive than
                # the same return in a calm regime (the Sharpe is lower), so we
                # apply a regime penalty that grows with vol stress.  This teaches
                # the agent to prefer holding through calm, trending conditions.
                reward = daily_ret * size
                if self.days_held > cfg.time_decay_start:
                    reward -= cfg.time_decay_rate * (self.days_held - cfg.time_decay_start)
                # Regime penalty: discourages sitting through high-vol, choppy periods
                if vol_regime > cfg.high_vol_regime_cut:
                    severity = (vol_regime - cfg.high_vol_regime_cut) / (1.0 - cfg.high_vol_regime_cut)
                    reward  -= cfg.vol_regime_penalty * severity

        # FLAT — opportunity cost nudge (unchanged)
        else:
            if daily_ret > 0:
                reward = -0.0003 * daily_ret

        self.t       += 1
        terminated    = self.t >= self.n_steps - 1
        next_state    = self._build_state() if not terminated else np.zeros(self.state_size)
        return next_state, float(reward), terminated, False, {}

    def _reset_position(self):
        self.position     = 0
        self.entry_price  = 0.0
        self.stop_price   = 0.0
        self.target_price = 0.0
        self.days_held    = 0
        self.position_size = 0.0


class _QNetwork:
    # Two-layer fully-connected network with ReLU, implemented in pure NumPy.
    # Xavier initialization keeps activations from exploding or vanishing
    # early in training when weights are large relative to the input scale.

    def __init__(self, input_dim: int, hidden_dim: int, n_actions: int, lr: float):
        self.lr = lr
        s1 = np.sqrt(2.0 / input_dim)
        s2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = np.random.randn(input_dim, hidden_dim) * s1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, n_actions) * s2
        self.b2 = np.zeros(n_actions)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        self._h = np.maximum(0.0, x @ self.W1 + self.b1)
        return self._h @ self.W2 + self.b2

    def backward(self, grad_out: np.ndarray) -> None:
        dW2 = self._h.T @ grad_out
        db2 = grad_out.sum(axis=0)
        dh  = (grad_out @ self.W2.T) * (self._h > 0)
        dW1 = self._x.T @ dh
        db1 = dh.sum(axis=0)
        self.W1 -= self.lr * dW1;  self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2;  self.b2 -= self.lr * db2

    def get_weights(self):
        return (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())

    def set_weights(self, w):
        self.W1, self.b1, self.W2, self.b2 = [x.copy() for x in w]


class DQNAgent:
    # Deep Q-Network with experience replay and a separate target network.
    
    # Experience replay breaks the temporal correlation between consecutive
    # training samples, without it, the network overfits to recent experience
    # and forgets earlier patterns (catastrophic forgetting).
    
    # The target network provides stable Q-value targets during gradient updates.
    # Without it, the network chases a moving target and training diverges.

    N_ACTIONS = 2  # 0 = sell/stay cash, 1 = buy/hold long

    def __init__(
        self,
        state_size:         int,
        hidden_dim:         int   = 128,
        lr:                 float = 1e-3,
        gamma:              float = 0.99,
        epsilon_start:      float = 1.0,
        epsilon_end:        float = 0.05,
        epsilon_decay:      float = 0.99,
        batch_size:         int   = 64,
        replay_capacity:    int   = 10_000,
        target_update_freq: int   = 50,
        random_state:       int   = 42,
    ):
        np.random.seed(random_state)
        self.gamma              = gamma
        self.epsilon            = epsilon_start
        self.epsilon_end        = epsilon_end
        self.epsilon_decay      = epsilon_decay
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self._step              = 0
        self.scaler             = None  # set after training; applied internally during inference
        self.q_net      = _QNetwork(state_size, hidden_dim, self.N_ACTIONS, lr)
        self.target_net = _QNetwork(state_size, hidden_dim, self.N_ACTIONS, lr)
        self.target_net.set_weights(self.q_net.get_weights())
        self.memory: deque = deque(maxlen=replay_capacity)

    def act(self, state: np.ndarray) -> int:
        # Epsilon-greedy exploration: random action with probability epsilon,
        # greedy (best Q-value) otherwise. Epsilon decays over training so the
        # agent explores broadly early on and exploits learned policy later.
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.N_ACTIONS)
        return int(np.argmax(self.q_net.forward(state.reshape(1, -1))))

    def remember(self, s, a, r, ns, done):
        self.memory.append((s, a, r, ns, done))

    def replay(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None
        idx         = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch       = [self.memory[i] for i in idx]
        states      = np.array([b[0] for b in batch])
        actions     = np.array([b[1] for b in batch], dtype=int)
        rewards     = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones       = np.array([b[4] for b in batch], dtype=float)

        q_current = self.q_net.forward(states)
        q_next    = self.target_net.forward(next_states)
        q_target  = q_current.copy()

        # Bellman update: Q(s,a) = r + gamma * max Q(s') for non-terminal states
        for i in range(self.batch_size):
            td = rewards[i] if dones[i] else rewards[i] + self.gamma * np.max(q_next[i])
            q_target[i, actions[i]] = td

        grad = 2.0 * (q_current - q_target) / self.batch_size
        self.q_net.backward(grad)
        loss = float(np.mean((q_current - q_target) ** 2))

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self._step  += 1
        if self._step % self.target_update_freq == 0:
            self.target_net.set_weights(self.q_net.get_weights())
        return loss

    def predict(
        self,
        X:                np.ndarray,
        prices:           np.ndarray | None = None,
        feature_cols:     list | None       = None,
        config:           TradingConfig | None = None,
        initial_position: int = 0,
    ) -> np.ndarray:
        if prices is not None and feature_cols is not None:
            return self._predict_with_env(X, prices, feature_cols, config, initial_position)
        return self._predict_simple(X, initial_position)

    def _predict_with_env(self, X, prices, feature_cols, config, initial_position):
        # Full environment simulation at inference time so the same trading rules
        # that shaped training rewards also govern backtest position decisions.
        # Scaler is applied here, X must be raw (unscaled) on entry.
        cfg = config or TradingConfig()
        X_scaled = self.scaler.transform(X) if self.scaler is not None else X
        dummy_returns = np.zeros(len(X))
        env = TradingEnv(X, dummy_returns, prices, feature_cols, cfg, scaled_features=X_scaled)
        env.position = initial_position

        signals  = np.zeros(len(X), dtype=int)
        state    = env._build_state()
        for t in range(len(X) - 1):
            action                         = int(np.argmax(self.q_net.forward(state.reshape(1, -1))))
            next_state, _, _, _, _         = env.step(action)
            signals[t]                     = env.position
            state                          = next_state
        signals[-1] = env.position
        return signals

    def _predict_simple(self, X, initial_position):
        # Fallback when prices aren't available, greedy argmax with position tracking
        # but without stop/target/volume rules. Used for classification metrics only.
        X_scaled = self.scaler.transform(X) if self.scaler is not None else X
        signals  = np.zeros(len(X), dtype=int)
        position = initial_position
        for t, features in enumerate(X_scaled):
            state      = np.append(features, [float(position), 0.0, 0.0, 1.0, 1.0])
            action     = int(np.argmax(self.q_net.forward(state.reshape(1, -1))))
            signals[t] = action
            position   = action
        return signals

    def predict_step(
        self,
        features:      np.ndarray,
        position:      int,
        days_held:     int,
        entry_price:   float,
        stop_price:    float,
        target_price:  float,
        current_price: float,
        config:        TradingConfig | None = None,
    ) -> int:
        # Single-step inference for live trading.
        # The caller provides full trade context so the agent can decide
        # whether to hold, exit, or enter with the same information it had
        # during training — not just raw features.
        cfg       = config or TradingConfig()
        days_norm = days_held / cfg.max_holding_days
        if self.scaler is not None:
            features = self.scaler.transform(features.reshape(1, -1))[0]

        if position == 1 and entry_price > 0:
            unrealized  = (current_price - entry_price) / entry_price
            risk_dist   = entry_price - stop_price
            stop_dist   = (current_price - stop_price) / risk_dist if risk_dist > 0 else 1.0
            target_rng  = target_price - entry_price
            target_dist = (target_price - current_price) / target_rng if target_rng > 0 else 0.0
        else:
            unrealized = stop_dist = 0.0
            target_dist = 1.0

        state = np.append(features, [
            float(position), days_norm,
            unrealized,
            np.clip(stop_dist,   -2.0, 3.0),
            np.clip(target_dist, -1.0, 2.0),
        ])
        return int(np.argmax(self.q_net.forward(state.reshape(1, -1))))


def train_dqn_agent(
    X_train:             np.ndarray,
    daily_returns_train: np.ndarray,
    prices_train:        np.ndarray | None  = None,
    feature_cols:        list | None        = None,
    config:              TradingConfig | None = None,
    n_episodes:          int   = 1000,
    hidden_dim:          int   = 128,
    lr:                  float = 1e-3,
    transaction_cost:    float = 0.0005,
    random_state:        int   = 42,
    X_train_scaled:      np.ndarray | None  = None,
) -> DQNAgent:
    cfg = config or TradingConfig()

    if prices_train is not None and feature_cols is not None:
        # Full disciplined environment, all trading rules active during training
        env  = TradingEnv(X_train, daily_returns_train, prices_train, feature_cols, cfg,
                          scaled_features=X_train_scaled)
        mode = "disciplined (R/R + stops + volume + MA)"
    else:
        # Basic environment used as fallback when price data is unavailable
        env  = _BasicTradingEnv(X_train, daily_returns_train, transaction_cost)
        mode = "basic (daily return only)"

    agent = DQNAgent(
        state_size=env.state_size,
        hidden_dim=hidden_dim,
        lr=lr,
        random_state=random_state,
    )

    print(f"  Training DQN — {n_episodes} episodes | mode: {mode}")
    print(f"  State size: {env.state_size} | Hidden: {hidden_dim}")

    for ep in range(n_episodes):
        state, _     = env.reset()
        total_reward = 0.0
        losses       = []

        while True:
            action                                  = agent.act(state)
            next_state, reward, terminated, _, _    = env.step(action)
            done = terminated
            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()
            if loss is not None:
                losses.append(loss)
            total_reward += reward
            state         = next_state
            if done:
                break

        avg_loss = float(np.mean(losses)) if losses else float("nan")
        print(
            f"    ep {ep+1:>2}/{n_episodes}  "
            f"reward={total_reward:+.4f}  "
            f"ε={agent.epsilon:.3f}  "
            f"loss={avg_loss:.6f}"
        )

    return agent


class _BasicTradingEnv(_EnvBase):
    # Simplified environment used as fallback when price data isn't provided.
    # No stops, no R/R gating, reward is simply the daily return times position.
    # The full TradingEnv should always be preferred in practice.

    def __init__(self, feature_array, daily_returns, transaction_cost=0.0005):
        super().__init__()
        self.features         = feature_array
        self.daily_returns    = daily_returns
        self.transaction_cost = transaction_cost
        self.n_steps          = len(feature_array)
        self.n_features       = feature_array.shape[1]
        self.state_size       = self.n_features + 5  # matches TradingEnv state size

        if _GYM_AVAILABLE:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.state_size,),
                dtype=np.float32,
            )
            self.action_space = spaces.Discrete(2)

        self.t        = 0
        self.position = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        if _GYM_AVAILABLE and seed is not None:
            super().reset(seed=seed)
        self.t        = 0
        self.position = 0
        return self._state(), {}

    def _state(self):
        return np.append(self.features[self.t], [float(self.position), 0.0, 0.0, 1.0, 1.0])

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        daily_ret     = self.daily_returns[self.t]
        tx            = self.transaction_cost * abs(action - self.position)
        reward        = action * daily_ret - tx
        self.position = action
        self.t       += 1
        terminated    = self.t >= self.n_steps - 1
        next_state    = self._state() if not terminated else np.zeros(self.state_size)
        return next_state, float(reward), terminated, False, {}
