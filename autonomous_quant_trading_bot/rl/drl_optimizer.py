from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .trading_env import US30TradingEnv

logger = logging.getLogger("trading_bot")

try:
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback

    SB3_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PPO = SAC = TD3 = None
    DummyVecEnv = SubprocVecEnv = BaseCallback = object
    SB3_AVAILABLE = False


ACTION_KEYS = [
    "position_size_multiplier",
    "stop_distance_multiplier",
    "tp_rr_ratio",
    "pattern_threshold_scaler",
    "garch_order_adjustment",
    "mc_path_count_scaler",
    "risk_multiplier",
]


def _to_action_dict(action: np.ndarray) -> Dict[str, float]:
    clipped = np.clip(np.asarray(action, dtype=np.float64), 0.5, 2.0)
    return {k: float(v) for k, v in zip(ACTION_KEYS, clipped.tolist())}


@dataclass
class TrainingReport:
    accepted: bool
    metrics: Dict[str, float]
    out_of_sample_improvement: float
    model_path: str
    reason: str


class _CalmarCheckpointCallback(BaseCallback):
    def __init__(self, eval_env: US30TradingEnv, eval_freq: int, save_path: Path, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.best_calmar = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        metrics = evaluate_policy(self.model, self.eval_env, n_episodes=2)
        calmar = metrics.get("calmar", -np.inf)
        if calmar > self.best_calmar:
            self.best_calmar = calmar
            self.model.save(str(self.save_path))
            logger.info("Saved improved DRL checkpoint at Calmar=%.4f", calmar)
        return True


class DRLOptimizer:
    def __init__(self, config: Dict | None = None) -> None:
        self.config = config or {}
        drl_cfg = self.config.get("drl", {})
        self.enabled = bool(drl_cfg.get("drl_enabled", True))
        root = Path(__file__).resolve().parents[1]
        default_model_path = root / "results" / "models" / "drl_us30_model"
        self.model_path = Path(drl_cfg.get("drl_model_path", str(default_model_path)))
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        self.algo = str(drl_cfg.get("drl_algo", "PPO")).upper()
        self.num_envs = int(drl_cfg.get("num_envs", 4))
        self.eval_freq = int(drl_cfg.get("eval_freq", 10_000))
        self.model = None
        self._fallback_action = np.ones((len(ACTION_KEYS),), dtype=np.float64)

        if self.enabled and SB3_AVAILABLE and self.model_path.with_suffix(".zip").exists():
            self._load_model()
        elif self.enabled and not SB3_AVAILABLE and self.model_path.with_suffix(".json").exists():
            self._load_fallback_model()

    def _make_env(self, seed: int = 42, timeframe: str = "M5") -> US30TradingEnv:
        return US30TradingEnv(config=self.config, timeframe=timeframe, seed=seed)

    def _load_model(self) -> None:
        model_file = str(self.model_path)
        if self.algo == "SAC":
            self.model = SAC.load(model_file)
        elif self.algo == "TD3":
            self.model = TD3.load(model_file)
        else:
            self.model = PPO.load(model_file)
        logger.info("Loaded DRL model from %s", model_file)

    def _load_fallback_model(self) -> None:
        with open(self.model_path.with_suffix(".json"), "r", encoding="utf-8") as f:
            payload = json.load(f)
        self._fallback_action = np.array(payload.get("best_action", [1.0] * len(ACTION_KEYS)), dtype=np.float64)
        logger.info("Loaded fallback DRL action model from %s", self.model_path.with_suffix(".json"))

    def _save_fallback_model(self, metrics: Dict[str, float]) -> None:
        payload = {"best_action": self._fallback_action.tolist(), "metrics": metrics}
        with open(self.model_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _build_sb3_model(self, vec_env):
        import torch as _torch
        device = "cuda" if _torch.cuda.is_available() else "cpu"
        if self.algo == "SAC":
            return SAC("MlpPolicy", vec_env, verbose=0, device=device)
        if self.algo == "TD3":
            return TD3("MlpPolicy", vec_env, verbose=0, device=device)
        return PPO("MlpPolicy", vec_env, verbose=0, device=device)

    def _train_fallback(self, num_trials: int = 60) -> TrainingReport:
        train_env = self._make_env(seed=123, timeframe="M5")
        val_env = self._make_env(seed=777, timeframe="H1")

        base_metrics = evaluate_random_policy(val_env, n_episodes=2)
        best_score = -np.inf
        best_action = self._fallback_action.copy()
        best_metrics = base_metrics.copy()

        for _ in range(num_trials):
            candidate = np.clip(self._fallback_action + np.random.normal(0, 0.15, size=len(ACTION_KEYS)), 0.5, 2.0)
            trial_metrics = evaluate_fixed_action(train_env, candidate, n_episodes=2)
            score = trial_metrics.get("calmar", 0.0) + 0.2 * trial_metrics.get("sortino", 0.0)
            if score > best_score:
                best_score = score
                best_action = candidate
                best_metrics = evaluate_fixed_action(val_env, candidate, n_episodes=2)

        self._fallback_action = best_action
        self._save_fallback_model(best_metrics)

        improvement = (
            (best_metrics.get("calmar", 0.0) - base_metrics.get("calmar", 0.0))
            / max(abs(base_metrics.get("calmar", 0.0)), 1e-6)
        )
        accepted = best_metrics.get("calmar", 0.0) > 3.0 and improvement > 0.15
        reason = "accepted" if accepted else "rejected: thresholds not met"
        return TrainingReport(
            accepted=accepted,
            metrics=best_metrics,
            out_of_sample_improvement=float(improvement),
            model_path=str(self.model_path.with_suffix(".json")),
            reason=reason,
        )

    def train_drl(self, num_timesteps: int = 100_000, eval_freq: int = 10_000) -> TrainingReport:
        if not self.enabled:
            return TrainingReport(False, {}, 0.0, str(self.model_path), "drl disabled")

        if not SB3_AVAILABLE:
            logger.warning("stable-baselines3 is not installed; using fallback optimizer.")
            report = self._train_fallback(num_trials=max(20, num_timesteps // 3000))
            logger.info("Fallback DRL metrics: %s", report.metrics)
            return report

        eval_freq = eval_freq or self.eval_freq
        eval_env = self._make_env(seed=99, timeframe="H1")
        baseline_metrics = evaluate_random_policy(eval_env, n_episodes=2)

        if self.num_envs > 1:
            vec_env = SubprocVecEnv(
                [lambda i=i: self._make_env(seed=100 + i, timeframe="M5") for i in range(self.num_envs)]
            )
        else:
            vec_env = DummyVecEnv([lambda: self._make_env(seed=100, timeframe="M5")])

        self.model = self._build_sb3_model(vec_env)
        callback = _CalmarCheckpointCallback(
            eval_env=eval_env,
            eval_freq=eval_freq,
            save_path=self.model_path,
            verbose=0,
        )
        self.model.learn(total_timesteps=num_timesteps, callback=callback)

        val_metrics = evaluate_policy(self.model, eval_env, n_episodes=3)
        out_of_sample_improvement = (
            (val_metrics.get("calmar", 0.0) - baseline_metrics.get("calmar", 0.0))
            / max(abs(baseline_metrics.get("calmar", 0.0)), 1e-6)
        )

        accepted = val_metrics.get("calmar", 0.0) > 3.0 and out_of_sample_improvement > 0.15
        if accepted:
            self.model.save(str(self.model_path))
            reason = "accepted"
        else:
            reason = "rejected: thresholds not met"

        logger.info(
            "DRL training metrics | Sharpe=%.3f Sortino=%.3f Calmar=%.3f PF=%.3f Expectancy=%.6f OOS=%.2f%%",
            val_metrics.get("sharpe", 0.0),
            val_metrics.get("sortino", 0.0),
            val_metrics.get("calmar", 0.0),
            val_metrics.get("profit_factor", 0.0),
            val_metrics.get("expectancy", 0.0),
            out_of_sample_improvement * 100.0,
        )
        return TrainingReport(
            accepted=accepted,
            metrics=val_metrics,
            out_of_sample_improvement=float(out_of_sample_improvement),
            model_path=str(self.model_path),
            reason=reason,
        )

    def get_optimal_actions(self, current_state: np.ndarray) -> Dict[str, float]:
        if not self.enabled:
            return _to_action_dict(np.ones((len(ACTION_KEYS),), dtype=np.float64))

        state = np.asarray(current_state, dtype=np.float32).reshape(1, -1)
        if SB3_AVAILABLE:
            if self.model is None and self.model_path.with_suffix(".zip").exists():
                self._load_model()
            if self.model is not None:
                action, _ = self.model.predict(state, deterministic=True)
                return _to_action_dict(np.asarray(action).reshape(-1))

        if self.model_path.with_suffix(".json").exists():
            self._load_fallback_model()
        return _to_action_dict(self._fallback_action)

    def get_action(self, current_state: np.ndarray) -> Dict[str, float]:
        return self.get_optimal_actions(current_state)

    def continue_online_learning(self, num_timesteps: int = 20_000) -> TrainingReport:
        return self.train_drl(num_timesteps=num_timesteps, eval_freq=max(2_000, self.eval_freq // 2))


def evaluate_fixed_action(env: US30TradingEnv, action: np.ndarray, n_episodes: int = 2) -> Dict[str, float]:
    info_rows = []
    for _ in range(n_episodes):
        _, _ = env.reset()
        done = False
        while not done:
            _, _, done, _, info = env.step(action)
        info_rows.append(info)
    return _aggregate_metrics(info_rows)


def evaluate_random_policy(env: US30TradingEnv, n_episodes: int = 2) -> Dict[str, float]:
    info_rows = []
    for _ in range(n_episodes):
        _, _ = env.reset()
        done = False
        while not done:
            a = env.action_space.sample()
            _, _, done, _, info = env.step(a)
        info_rows.append(info)
    return _aggregate_metrics(info_rows)


def evaluate_policy(model, env: US30TradingEnv, n_episodes: int = 3) -> Dict[str, float]:
    info_rows = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
            obs, _, done, _, info = env.step(np.asarray(action).reshape(-1))
        info_rows.append(info)
    return _aggregate_metrics(info_rows)


def _aggregate_metrics(rows: list[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = ["sharpe", "sortino", "calmar", "profit_factor", "expectancy", "drawdown"]
    out: Dict[str, float] = {}
    for k in keys:
        vals = [float(r.get(k, 0.0)) for r in rows]
        out[k] = float(np.mean(vals))
    return out


_OPTIMIZER_SINGLETON: Optional[DRLOptimizer] = None


def _get_optimizer(config: Dict | None = None) -> DRLOptimizer:
    global _OPTIMIZER_SINGLETON
    if _OPTIMIZER_SINGLETON is None:
        _OPTIMIZER_SINGLETON = DRLOptimizer(config=config or {})
    return _OPTIMIZER_SINGLETON


def train_drl(num_timesteps: int = 100_000, eval_freq: int = 10_000, config: Dict | None = None) -> TrainingReport:
    return _get_optimizer(config).train_drl(num_timesteps=num_timesteps, eval_freq=eval_freq)


def get_optimal_actions(current_state: np.ndarray, config: Dict | None = None) -> Dict[str, float]:
    return _get_optimizer(config).get_optimal_actions(current_state)


def get_action(current_state: np.ndarray, config: Dict | None = None) -> Dict[str, float]:
    return get_optimal_actions(current_state=current_state, config=config)

