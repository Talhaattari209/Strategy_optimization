"""Tests for RL environment stability and integration."""
from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from autonomous_quant_trading_bot.rl.trading_env import US30TradingEnv


def test_rl_env_runs_1000_steps_reward_stability():
    env = US30TradingEnv(
        config={"drl": {"us30_data_glob": "Blue Capital data/US30_data/*.csv"}},
        timeframe="M5",
        episode_bars=1000,
        seed=42,
    )
    obs, _ = env.reset()
    assert obs.shape[0] == env.observation_space.shape[0]

    rewards = []
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        rewards.append(reward)
        if done:
            obs, _ = env.reset()

    rewards_arr = np.array(rewards, dtype=np.float64)
    assert np.isfinite(rewards_arr).all()
    assert abs(float(np.mean(rewards_arr))) < 1e4
    assert float(np.std(rewards_arr)) < 1e5
    assert "calmar" in info
