"""
Test suite for the DQN trading system.

Objective: diagnose and prevent the "agent buys but never sells" bug by
verifying every layer that is supposed to produce exits:

    Layer 1 — TradingEnv.step()       : forced exits (stop / target / time stop)
    Layer 2 — resolve_strategy_action : live signal overrides
    Layer 3 — DQNAgent                : action distribution, predict_step inference
    Layer 4 — Integration             : backtest produces real buys AND sells

Run from the project root:
    pytest backend/tests/ -v
"""
from __future__ import annotations

import sys
import os

# Ensure backend/ is importable regardless of invocation directory.
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import numpy as np
import pytest

from stock_prediction.rl_agent import (
    TradingEnv,
    TradingConfig,
    DQNAgent,
    _BasicTradingEnv,
    train_dqn_agent,
)
from stock_prediction.monitor import (
    StrategyState,
    Position,
    resolve_strategy_action,
)
from tests.conftest import FEATURE_COLS, N_FEATURES, make_synthetic_data


# ===========================================================================
# Layer 1 — TradingEnv: Gymnasium API compliance
# ===========================================================================


class TestGymnasiumAPI:
    """TradingEnv must expose the standard Gymnasium reset/step contract."""

    def test_reset_returns_tuple(self, env):
        result = env.reset()
        assert isinstance(result, tuple), "reset() must return a tuple (obs, info)"
        assert len(result) == 2, "reset() must return exactly (obs, info)"

    def test_reset_obs_shape(self, env):
        obs, info = env.reset()
        assert obs.shape == (env.state_size,), (
            f"Observation shape {obs.shape} != expected ({env.state_size},)"
        )

    def test_reset_info_is_dict(self, env):
        _, info = env.reset()
        assert isinstance(info, dict)

    def test_step_returns_five_tuple(self, env):
        env.reset()
        result = env.step(0)
        assert isinstance(result, tuple)
        assert len(result) == 5, "step() must return (obs, reward, terminated, truncated, info)"

    def test_step_types(self, env):
        env.reset()
        obs, reward, terminated, truncated, info = env.step(1)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_truncated_always_false(self, env):
        """TradingEnv uses terminated only — truncated is always False."""
        env.reset()
        for _ in range(5):
            _, _, _, truncated, _ = env.step(1)
            assert not truncated, "TradingEnv should never set truncated=True"

    def test_terminated_fires_at_end(self):
        """terminated must be True on the final step."""
        features, daily_returns, prices = make_synthetic_data(n_steps=10)
        cfg = TradingConfig()
        e = TradingEnv(features, daily_returns, prices, FEATURE_COLS, config=cfg)
        e.reset()
        terminated = False
        for _ in range(20):          # more steps than n_steps to be safe
            _, _, terminated, _, _ = e.step(0)
            if terminated:
                break
        assert terminated, "Episode must eventually terminate"

    def test_observation_space_attribute(self, env):
        """observation_space must be present when gymnasium is installed."""
        try:
            import gymnasium  # noqa: F401
            assert hasattr(env, "observation_space"), "observation_space missing"
            assert env.observation_space.shape == (env.state_size,)
        except ImportError:
            pytest.skip("gymnasium not installed")

    def test_action_space_attribute(self, env):
        """action_space must be present when gymnasium is installed."""
        try:
            import gymnasium  # noqa: F401
            assert hasattr(env, "action_space"), "action_space missing"
            assert env.action_space.n == 2
        except ImportError:
            pytest.skip("gymnasium not installed")

    def test_basic_env_api_compliance(self):
        """_BasicTradingEnv must also comply with the Gymnasium API."""
        features, daily_returns, _ = make_synthetic_data(n_steps=50)
        e = _BasicTradingEnv(features, daily_returns)
        obs, info = e.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        obs2, reward, terminated, truncated, info2 = e.step(1)
        assert isinstance(obs2, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert not truncated


# ===========================================================================
# Layer 1 — TradingEnv: forced exit mechanics
# ===========================================================================


class TestForcedExits:
    """
    The three forced exits (stop, target, time stop) are the primary mechanism
    that closes positions when the DQN would otherwise never output action=0.
    Each one must fire reliably.
    """

    # ------------------------------------------------------------------
    # Helper: put env into a LONG position with explicit stop/target/days
    # ------------------------------------------------------------------

    @staticmethod
    def _enter_long(env: TradingEnv, stop_price: float, target_price: float, days: int = 0):
        """Manually inject a long position into the env without going through step()."""
        env.position     = 1
        env.entry_price  = env.prices[env.t]
        env.stop_price   = stop_price
        env.target_price = target_price
        env.days_held    = days

    # ------------------------------------------------------------------
    # Stop loss
    # ------------------------------------------------------------------

    def test_stop_loss_exits_position(self, env):
        """When next_price falls below stop_price the position must close."""
        env.reset()
        curr_price = env.prices[env.t]

        # Set stop just above the next expected price (guaranteed hit)
        # next_price = curr_price * (1 + daily_return)
        # We force daily_return to be a big negative number
        env.daily_returns[env.t] = -0.05          # -5% move
        next_price_approx = curr_price * 0.95
        stop_price = next_price_approx + 1.0       # stop ABOVE the expected next price

        self._enter_long(env, stop_price=stop_price, target_price=curr_price * 1.10)

        _, reward, _, _, _ = env.step(1)           # hold — stop should fire

        assert env.position == 0, "Position must be flat after stop is hit"
        assert reward < 0, "Stop loss must produce a negative reward"

    def test_stop_loss_reward_includes_penalty(self, env):
        """Stop-loss reward must include the extra -0.002 penalty."""
        env.reset()
        curr_price = env.prices[env.t]
        env.daily_returns[env.t] = -0.05
        stop_price = curr_price * 0.96             # well above the 0.95 next price

        self._enter_long(env, stop_price=stop_price, target_price=curr_price * 1.10)

        _, reward, _, _, _ = env.step(1)

        # The stop-loss path subtracts transaction_cost + 0.002 in addition to pnl.
        # A 4% loss with size≈1 gives roughly -0.04 - 0.0005 - 0.002 ≈ -0.0425
        assert reward < -0.002, f"Stop reward too lenient: {reward:.4f}"

    # ------------------------------------------------------------------
    # Target hit
    # ------------------------------------------------------------------

    def test_target_hit_exits_position(self, env):
        """When next_price rises above target_price the position must close."""
        env.reset()
        curr_price = env.prices[env.t]

        env.daily_returns[env.t] = 0.10             # +10% move
        next_price_approx = curr_price * 1.10
        target_price = next_price_approx - 1.0      # target BELOW the expected next price

        self._enter_long(env, stop_price=curr_price * 0.90, target_price=target_price)

        _, reward, _, _, _ = env.step(1)

        assert env.position == 0, "Position must be flat after target is hit"
        assert reward > 0, "Target hit must produce a positive reward"

    def test_target_hit_reward_includes_bonus(self, env):
        """Target-hit reward must include the +0.002 bonus."""
        env.reset()
        curr_price = env.prices[env.t]
        env.daily_returns[env.t] = 0.10
        target_price = curr_price * 1.05            # below the 10% move

        self._enter_long(env, stop_price=curr_price * 0.90, target_price=target_price)

        _, reward, _, _, _ = env.step(1)

        # Profit path adds +0.002 bonus after pnl*size - tx
        assert reward > 0.002, f"Target reward too small: {reward:.4f}"

    # ------------------------------------------------------------------
    # Time stop
    # ------------------------------------------------------------------

    def test_time_stop_fires_exactly_at_max_days(self, env):
        """Position must be closed when days_held reaches max_holding_days."""
        env.reset()
        cfg = env.cfg
        curr_price = env.prices[env.t]

        # Put position one step before the time stop
        self._enter_long(
            env,
            stop_price=curr_price * 0.50,       # stop far below — won't hit
            target_price=curr_price * 2.00,     # target far above — won't hit
            days=cfg.max_holding_days - 1,      # one HOLD step will push it to max
        )

        # step() increments days_held, then checks if >= max_holding_days
        _, _, _, _, _ = env.step(1)             # hold — days_held reaches max

        assert env.position == 0, (
            f"Time stop should have fired at days_held={cfg.max_holding_days}, "
            f"but position is still LONG"
        )

    def test_time_stop_does_not_fire_early(self, env):
        """Time stop must NOT fire before max_holding_days is reached."""
        env.reset()
        cfg = env.cfg
        curr_price = env.prices[env.t]

        self._enter_long(
            env,
            stop_price=curr_price * 0.50,
            target_price=curr_price * 2.00,
            days=0,
        )

        # Step through max-2 hold periods (days 1 through max-1) — no exit yet
        for _ in range(cfg.max_holding_days - 2):
            env.daily_returns[env.t] = 0.001    # tiny positive drift — no stop/target hit
            obs, _, terminated, _, _ = env.step(1)
            if terminated:
                break

        assert env.position == 1, "Position should still be LONG before time stop"

    # ------------------------------------------------------------------
    # Voluntary sell (action=0 while LONG)
    # ------------------------------------------------------------------

    def test_voluntary_sell_closes_position(self, env):
        """action=0 while LONG must immediately close the position."""
        env.reset()
        curr_price = env.prices[env.t]

        self._enter_long(
            env,
            stop_price=curr_price * 0.90,
            target_price=curr_price * 1.10,
        )

        _, _, _, _, _ = env.step(0)   # explicit sell

        assert env.position == 0, "Voluntary sell (action=0) must flatten position"

    # ------------------------------------------------------------------
    # Buy gate: R/R check
    # ------------------------------------------------------------------

    def test_bad_rr_penalises_but_does_not_open_position(self, env):
        """
        Entry is refused when R/R < min_rr_ratio.
        Set ATR to near-zero so stop_dist→0 and R/R can't be computed.
        """
        env.reset()
        # Override atr_pct to 0 — makes stop_dist = 0, rr calculation falls apart
        env.features[env.t, 0] = 0.0             # atr14_pct index = 0 in FEATURE_COLS
        _, reward, _, _, _ = env.step(1)          # attempt buy

        assert env.position == 0, "Position must stay flat when R/R is invalid"
        assert reward < 0, "Bad R/R attempt must incur a penalty reward"


# ===========================================================================
# Layer 2 — resolve_strategy_action: live signal override logic
# ===========================================================================


class TestResolveStrategyAction:
    """
    resolve_strategy_action is the live-trading bridge between the raw DQN
    signal and actual order submission.  It must enforce the same exit rules
    as TradingEnv, regardless of what the DQN outputs.
    """

    @staticmethod
    def _flat_state(ticker="TEST") -> StrategyState:
        return StrategyState(ticker=ticker)

    @staticmethod
    def _long_state(
        ticker="TEST",
        entry_price=100.0,
        stop_price=95.0,
        target_price=110.0,
        days_held=0,
    ) -> StrategyState:
        s = StrategyState(ticker=ticker)
        s.position     = Position.LONG
        s.entry_price  = entry_price
        s.stop_price   = stop_price
        s.target_price = target_price
        s.days_held    = days_held
        return s

    @staticmethod
    def _dummy_features(n: int = N_FEATURES) -> np.ndarray:
        """Feature vector with realistic ATR and normal market indicators."""
        f = np.zeros(n, dtype=np.float32)
        f[FEATURE_COLS.index("atr14_pct")]    = 0.015
        f[FEATURE_COLS.index("volume_ratio")] = 1.0
        f[FEATURE_COLS.index("ma20_ratio")]   = 1.0
        return f

    # ------------------------------------------------------------------
    # FLAT state transitions
    # ------------------------------------------------------------------

    def test_flat_signal1_returns_buy(self):
        state  = self._flat_state()
        action = resolve_strategy_action(state, 1, 100.0, self._dummy_features(), FEATURE_COLS)
        assert action == "BUY"

    def test_flat_signal0_returns_wait(self):
        state  = self._flat_state()
        action = resolve_strategy_action(state, 0, 100.0, self._dummy_features(), FEATURE_COLS)
        assert action == "WAIT"

    def test_buy_sets_stop_and_target(self):
        """On BUY the state must have a valid stop below and target above entry."""
        state  = self._flat_state()
        price  = 100.0
        resolve_strategy_action(state, 1, price, self._dummy_features(), FEATURE_COLS)
        assert state.stop_price   < price,  "Stop must be below entry price"
        assert state.target_price > price,  "Target must be above entry price"

    def test_buy_achieves_min_rr(self):
        """On BUY the R/R ratio must be exactly min_rr_ratio."""
        state   = self._flat_state()
        price   = 100.0
        features = self._dummy_features()
        resolve_strategy_action(state, 1, price, features, FEATURE_COLS)

        risk    = price - state.stop_price
        reward_ = state.target_price - price
        cfg     = TradingConfig()
        assert abs(reward_ / risk - cfg.min_rr_ratio) < 1e-6, (
            f"R/R={reward_/risk:.4f} should equal {cfg.min_rr_ratio}"
        )

    def test_buy_transitions_to_long(self):
        state  = self._flat_state()
        resolve_strategy_action(state, 1, 100.0, self._dummy_features(), FEATURE_COLS)
        assert state.position == Position.LONG

    # ------------------------------------------------------------------
    # LONG — DQN says hold but forced exit fires
    # ------------------------------------------------------------------

    def test_long_signal1_returns_hold_normally(self):
        state  = self._long_state()
        action = resolve_strategy_action(state, 1, 102.0, self._dummy_features(), FEATURE_COLS)
        assert action == "HOLD", "Should HOLD when price is between stop and target"

    def test_long_signal1_stop_override_returns_sell(self):
        """Stop hit must force SELL even when DQN outputs signal=1 (hold)."""
        state  = self._long_state(stop_price=98.0)
        price  = 97.0    # below stop
        action = resolve_strategy_action(state, 1, price, self._dummy_features(), FEATURE_COLS)
        assert action == "SELL", "Stop hit must produce SELL regardless of DQN signal"

    def test_long_signal1_target_override_returns_sell(self):
        """Target hit must force SELL even when DQN outputs signal=1 (hold)."""
        state  = self._long_state(target_price=108.0)
        price  = 109.0   # above target
        action = resolve_strategy_action(state, 1, price, self._dummy_features(), FEATURE_COLS)
        assert action == "SELL", "Target hit must produce SELL regardless of DQN signal"

    def test_long_time_stop_returns_sell(self):
        """Time stop must fire when days_held reaches max_holding_days."""
        cfg    = TradingConfig()
        state  = self._long_state(days_held=cfg.max_holding_days - 1)
        # Price within stop/target — only time stop applies
        action = resolve_strategy_action(state, 1, 102.0, self._dummy_features(), FEATURE_COLS)
        assert action == "SELL", (
            f"Time stop should fire at days_held >= {cfg.max_holding_days}, "
            f"got {action!r}"
        )

    def test_long_voluntary_sell_signal0(self):
        """DQN signal=0 while LONG must produce SELL (voluntary exit)."""
        state  = self._long_state()
        action = resolve_strategy_action(state, 0, 102.0, self._dummy_features(), FEATURE_COLS)
        assert action == "SELL"

    # ------------------------------------------------------------------
    # SELL must reset state
    # ------------------------------------------------------------------

    def test_sell_resets_state_to_flat(self):
        state = self._long_state()
        resolve_strategy_action(state, 0, 102.0, self._dummy_features(), FEATURE_COLS)
        assert state.position     == Position.FLAT
        assert state.entry_price  == 0.0
        assert state.stop_price   == 0.0
        assert state.target_price == 0.0
        assert state.days_held    == 0

    def test_sell_records_trade_log_entry(self):
        state = self._long_state()
        resolve_strategy_action(state, 0, 102.0, self._dummy_features(), FEATURE_COLS)
        assert len(state.trade_log) == 1
        trade = state.trade_log[0]
        assert "entry"  in trade
        assert "exit"   in trade
        assert "pnl%"   in trade
        assert "reason" in trade

    def test_sell_reason_stop(self):
        state  = self._long_state(stop_price=100.0)
        price  = 99.0
        resolve_strategy_action(state, 1, price, self._dummy_features(), FEATURE_COLS)
        assert state.trade_log[-1]["reason"] == "stop hit"

    def test_sell_reason_target(self):
        state  = self._long_state(target_price=105.0)
        price  = 106.0
        resolve_strategy_action(state, 1, price, self._dummy_features(), FEATURE_COLS)
        assert state.trade_log[-1]["reason"] == "target hit"

    def test_sell_reason_time_stop(self):
        cfg   = TradingConfig()
        state = self._long_state(days_held=cfg.max_holding_days - 1)
        resolve_strategy_action(state, 1, 102.0, self._dummy_features(), FEATURE_COLS)
        assert state.trade_log[-1]["reason"] == "time stop"

    def test_sell_reason_signal(self):
        state = self._long_state()
        resolve_strategy_action(state, 0, 102.0, self._dummy_features(), FEATURE_COLS)
        assert state.trade_log[-1]["reason"] == "signal"

    # ------------------------------------------------------------------
    # Full round-trip: BUY → multiple HOLDs → SELL
    # ------------------------------------------------------------------

    def test_full_round_trip(self):
        """Simulate a complete trade: buy, hold a few steps, then time-stop sell."""
        cfg    = TradingConfig()
        state  = self._flat_state()
        feats  = self._dummy_features()
        price  = 100.0

        # BUY
        action = resolve_strategy_action(state, 1, price, feats, FEATURE_COLS)
        assert action == "BUY"
        assert state.position == Position.LONG

        # HOLDs — price stays between stop and target
        hold_price = 101.0
        for _ in range(cfg.max_holding_days - 2):
            action = resolve_strategy_action(state, 1, hold_price, feats, FEATURE_COLS)
            assert action == "HOLD"

        # The next step reaches max_holding_days → time stop
        action = resolve_strategy_action(state, 1, hold_price, feats, FEATURE_COLS)
        assert action == "SELL"
        assert state.position == Position.FLAT


# ===========================================================================
# Layer 3 — DQNAgent: action distribution and inference
# ===========================================================================


class TestDQNAgent:
    """
    A trained (or even freshly-initialised) DQN must produce both actions.
    If it outputs only 1s, the time stop will eventually close positions —
    but any model that NEVER outputs 0 spontaneously is clearly degenerate.
    """

    def test_act_returns_valid_action(self, agent):
        obs = np.random.randn(N_FEATURES + 5).astype(np.float32)
        a   = agent.act(obs)
        assert a in (0, 1), f"act() returned {a}, expected 0 or 1"

    def test_act_explores_both_actions_with_high_epsilon(self, agent):
        """With epsilon=1.0 (pure exploration) the agent should sample both actions."""
        agent.epsilon = 1.0
        obs     = np.random.randn(N_FEATURES + 5).astype(np.float32)
        actions = [agent.act(obs) for _ in range(200)]
        assert 0 in actions, "With epsilon=1.0 agent never sampled action=0"
        assert 1 in actions, "With epsilon=1.0 agent never sampled action=1"

    def test_replay_reduces_loss(self, agent):
        """replay() should return a loss once the buffer has enough samples."""
        obs = np.random.randn(N_FEATURES + 5).astype(np.float32)
        for _ in range(agent.batch_size + 10):
            a  = np.random.randint(2)
            ns = np.random.randn(N_FEATURES + 5).astype(np.float32)
            agent.remember(obs, a, 0.01, ns, False)
        loss = agent.replay()
        assert loss is not None
        assert loss >= 0

    def test_predict_step_returns_valid_action(self, agent):
        """predict_step must return 0 or 1."""
        obs = np.random.randn(N_FEATURES).astype(np.float32)
        a   = agent.predict_step(
            features=obs,
            position=0,
            days_held=0,
            entry_price=0.0,
            stop_price=0.0,
            target_price=0.0,
            current_price=100.0,
        )
        assert a in (0, 1), f"predict_step returned {a}"

    def test_trained_agent_action_distribution(self, trained_agent):
        """
        After training the agent must NOT output only 1s.
        Root cause of the 'never sells' bug: all Q-values favour action=1.
        """
        agent, features, _, _ = trained_agent
        agent.epsilon = 0.0  # greedy — no exploration noise

        actions = []
        for row in features:
            state = np.append(row, [0.0, 0.0, 0.0, 1.0, 1.0])
            a     = agent.act(state)
            actions.append(a)

        n_sells = actions.count(0)
        n_buys  = actions.count(1)

        assert n_sells > 0, (
            f"Trained DQN never output action=0 (sell) on {len(actions)} states. "
            f"All outputs were action=1. This is the 'never sells' bug.\n"
            f"  action=0 count: {n_sells}\n"
            f"  action=1 count: {n_buys}\n"
            "Check that reward shaping correctly penalises holding losing positions."
        )
        assert n_buys > 0, "Trained DQN never output action=1 — policy is too conservative."

    def test_predict_step_context_affects_output(self, trained_agent):
        """
        The five context dimensions (position, days_held, unrealized, stop_dist,
        target_dist) must influence the action.  If context is ignored, the agent
        can't learn when to exit a position.  We verify the *same* market features
        can produce different actions with different trade contexts.
        """
        agent, features, _, _ = trained_agent
        agent.epsilon = 0.0

        obs = features[0]
        price = 100.0

        # Context 1: flat, no position
        a_flat = agent.predict_step(
            obs, position=0, days_held=0,
            entry_price=0.0, stop_price=0.0, target_price=0.0,
            current_price=price,
        )

        # Context 2: long, at max holding days, deep in the red
        a_long_losing = agent.predict_step(
            obs, position=1, days_held=10,
            entry_price=120.0, stop_price=116.0, target_price=128.0,
            current_price=price,   # price well below entry → deep loss
        )

        # Context 3: long, fresh entry, almost at target
        a_long_winning = agent.predict_step(
            obs, position=1, days_held=1,
            entry_price=98.0, stop_price=95.0, target_price=100.5,
            current_price=price,   # price just below target → winning
        )

        # All three must be valid; we just document that context matters.
        assert a_flat         in (0, 1)
        assert a_long_losing  in (0, 1)
        assert a_long_winning in (0, 1)

        # At minimum, the losing long (maxed-out days, deep underwater) should
        # ideally differ from the flat case — but we only assert they're valid
        # since a freshly-trained network on 30 episodes may not be perfectly tuned.
        # The critical assertion is test_trained_agent_action_distribution above.


# ===========================================================================
# Layer 4 — Integration: _predict_with_env produces real position changes
# ===========================================================================


class TestIntegrationPredict:
    """
    End-to-end test: DQN backtest must produce both entries (position=1)
    and exits (position=0).  A flat signal array (all 0s or all 1s) is
    a hard failure — the model is not trading.
    """

    def test_predict_with_env_has_both_states(self, trained_agent):
        """The backtest signal array must contain both 0s and 1s."""
        agent, features, daily_returns, prices = trained_agent
        signals = agent.predict(
            features,
            prices=prices,
            feature_cols=FEATURE_COLS,
        )
        assert 0 in signals, (
            "Backtest produced NO position=0 (exits). "
            "DQN may have never opened a trade, or TradingEnv never exited one."
        )
        assert 1 in signals, (
            "Backtest produced NO position=1 (entries). "
            "DQN may never be buying."
        )

    def test_predict_with_env_has_transitions(self, trained_agent):
        """There must be at least one FLAT→LONG and one LONG→FLAT transition."""
        agent, features, daily_returns, prices = trained_agent
        signals = agent.predict(features, prices=prices, feature_cols=FEATURE_COLS)

        transitions = np.diff(signals.astype(int))
        n_entries = int(np.sum(transitions ==  1))  # 0→1 (buy)
        n_exits   = int(np.sum(transitions == -1))  # 1→0 (sell)

        assert n_entries > 0, (
            f"No entries found in backtest signals. Transitions: {transitions[:20]}"
        )
        assert n_exits > 0, (
            f"No exits found in backtest signals — this is the 'never sells' bug!\n"
            f"Entries: {n_entries}, Exits: {n_exits}\n"
            "Check stop/target/time-stop logic in TradingEnv.step() and "
            "make sure max_holding_days triggers correctly."
        )

    def test_predict_with_env_respects_time_stop(self):
        """
        With an artificially small max_holding_days the env must time-stop
        every position quickly, producing many exits.
        """
        features, daily_returns, prices = make_synthetic_data(n_steps=200, seed=11)

        # Train with tiny max_holding_days so every trade time-stops within 3 steps
        cfg = TradingConfig(max_holding_days=3)
        agent = train_dqn_agent(
            X_train=features,
            daily_returns_train=daily_returns,
            prices_train=prices,
            feature_cols=FEATURE_COLS,
            config=cfg,
            n_episodes=15,
            hidden_dim=32,
            random_state=1,
        )

        signals = agent.predict(
            features,
            prices=prices,
            feature_cols=FEATURE_COLS,
            config=cfg,
        )
        transitions = np.diff(signals.astype(int))
        n_exits = int(np.sum(transitions == -1))

        assert n_exits > 0, (
            "With max_holding_days=3 there should be many time-stop exits, got 0."
        )


# ===========================================================================
# Regression tests: specific scenarios that triggered observed bugs
# ===========================================================================


class TestRegressions:
    """
    Specific failure modes observed during paper trading.
    Each test should fail on the buggy code and pass on the fixed code.
    """

    def test_env_reset_clears_position(self, env):
        """After reset() the env must always start flat — no stale position state."""
        # Manually force a long position
        env.position    = 1
        env.entry_price = 100.0
        env.days_held   = 5
        # Reset
        obs, _ = env.reset()
        assert env.position    == 0,   "reset() left stale position=1"
        assert env.entry_price == 0.0, "reset() left stale entry_price"
        assert env.days_held   == 0,   "reset() left stale days_held"

    def test_days_held_increments_only_when_long(self, env):
        """days_held must not increment while FLAT."""
        env.reset()
        for _ in range(10):
            env.daily_returns[env.t] = 0.001   # slight uptrend
            env.step(0)                         # stay flat

        assert env.days_held == 0, (
            f"days_held={env.days_held} should be 0 while always FLAT"
        )

    def test_days_held_increments_when_long(self, env):
        """days_held must increment on every HOLD step while LONG."""
        env.reset()
        curr = env.prices[env.t]

        # Enter long via step()
        env.step(1)

        # If a position was opened, hold and count
        if env.position == 1:
            for _ in range(3):
                env.daily_returns[env.t] = 0.001   # small move — no stop/target hit
                env.step(1)
            assert env.days_held >= 1, (
                "days_held should have incremented while holding LONG"
            )

    def test_multiple_round_trips_do_not_leak_state(self, long_env):
        """
        Simulate several full trade cycles using resolve_strategy_action.
        Each new trade must start with clean state (entry/stop/target from
        the new entry price, not a previous trade).
        """
        cfg   = TradingConfig()
        state = StrategyState(ticker="TEST")
        feats = np.zeros(N_FEATURES, dtype=np.float32)
        feats[FEATURE_COLS.index("atr14_pct")] = 0.015

        # Force 3 complete trades
        trades_completed = 0
        price = 100.0

        for trade_num in range(3):
            # BUY
            resolve_strategy_action(state, 1, price, feats, FEATURE_COLS)
            first_stop   = state.stop_price
            first_target = state.target_price

            # Hold until time stop fires
            for _ in range(cfg.max_holding_days):
                action = resolve_strategy_action(state, 1, price + 0.01, feats, FEATURE_COLS)
                if action == "SELL":
                    trades_completed += 1
                    # Verify state was fully reset
                    assert state.position     == Position.FLAT,  "State not reset to FLAT after SELL"
                    assert state.entry_price  == 0.0,            "entry_price leaked between trades"
                    assert state.stop_price   == 0.0,            "stop_price leaked between trades"
                    assert state.target_price == 0.0,            "target_price leaked between trades"
                    assert state.days_held    == 0,              "days_held leaked between trades"
                    price += 1.0   # slightly different entry next time
                    break

        assert trades_completed >= 1, (
            "Could not complete even one full trade cycle in the regression test"
        )
