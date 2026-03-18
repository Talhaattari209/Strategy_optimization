from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - fallback for minimal environments
    import gym
    from gym import spaces

try:
    from autonomous_quant_trading_bot.core.execution_engine import ExecutionEngine
    from autonomous_quant_trading_bot.math_engine.finance_models import ExpectancyCalculator
    from autonomous_quant_trading_bot.math_engine.linear_algebra import PCAReducer
    from autonomous_quant_trading_bot.math_engine.markov_bayesian import (
        BayesianUpdater,
        HiddenMarkovModel,
        MarketRegime,
    )
    from autonomous_quant_trading_bot.math_engine.stochastic_calculus import (
        GirsanovTransform,
        ItoLemma,
        MicrostructureSDE,
    )
    from autonomous_quant_trading_bot.math_engine.stochastic_processes import MonteCarloEngine
    from autonomous_quant_trading_bot.math_engine.time_series import ARIMA, GARCH
except ImportError:
    from core.execution_engine import ExecutionEngine
    from math_engine.finance_models import ExpectancyCalculator
    from math_engine.linear_algebra import PCAReducer
    from math_engine.markov_bayesian import BayesianUpdater, HiddenMarkovModel, MarketRegime
    from math_engine.stochastic_calculus import GirsanovTransform, ItoLemma, MicrostructureSDE
    from math_engine.stochastic_processes import MonteCarloEngine
    from math_engine.time_series import ARIMA, GARCH

logger = logging.getLogger("trading_bot")


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.maximum(peak, 1e-12)
    return float(np.max(dd))


def _safe_polyfit_slope(series: np.ndarray) -> float:
    if series.size < 3:
        return 0.0
    x = np.arange(series.size, dtype=np.float64)
    slope = np.polyfit(x, series, deg=1)[0]
    return float(slope)


class US30TradingEnv(gym.Env):
    """
    Gymnasium-compatible trading environment backed by the project's math engine.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Dict | None = None,
        timeframe: str = "M5",
        window_size: int = 512,
        episode_bars: int = 500,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config or {}
        self.timeframe = timeframe
        self.window_size = window_size
        self.episode_bars = episode_bars
        self.rng = np.random.default_rng(seed)

        self.param_names = [
            "position_size_multiplier",
            "stop_distance_multiplier",
            "tp_rr_ratio",
            "pattern_threshold_scaler",
            "garch_order_adjustment",
            "mc_path_count_scaler",
            "risk_multiplier",
        ]

        self.action_space = spaces.Box(
            low=np.full((len(self.param_names),), 0.5, dtype=np.float32),
            high=np.full((len(self.param_names),), 2.0, dtype=np.float32),
            shape=(len(self.param_names),),
            dtype=np.float32,
        )

        # 4 regime + 10 PCA + 2 forecasts + 4 MC stats + 3 position/equity + 1 spread
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(24,),
            dtype=np.float32,
        )

        self.pca = PCAReducer(n_components=10)
        self.arima = ARIMA(p=5, d=1, q=1)
        self.garch = GARCH()
        self.mc = MonteCarloEngine(n_paths=1000)
        self.mc_state = MonteCarloEngine(n_paths=300)
        self.microstructure = MicrostructureSDE()
        self.bayesian = BayesianUpdater(n_states=4)
        self.hmm = HiddenMarkovModel(n_states=4, n_iter=8)
        self.expectancy = ExpectancyCalculator()
        self.execution_engine = ExecutionEngine(self.config)

        self.dataset = self._load_us30_dataset(self.config)
        selected_df = self.dataset.get(self.timeframe)
        self.current_df = selected_df if selected_df is not None else next(iter(self.dataset.values()))
        self.current_idx = 0
        self.start_idx = 0
        self.end_idx = 0
        self.position_direction = 0
        self.position_entry = 0.0
        self.position_pnl = 0.0
        self.equity = 10000.0
        self.equity_curve: List[float] = []
        self.rewards: List[float] = []
        self.last_info: Dict[str, float] = {}
        self._cached_regime_probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        self._last_regime_update_idx = -1

    @staticmethod
    def _load_us30_dataset(config: Dict) -> Dict[str, pd.DataFrame]:
        root = Path(__file__).resolve().parents[2]
        pattern = config.get("drl", {}).get("us30_data_glob", "Blue Capital data/US30_data/*.csv")
        data_paths = sorted(root.glob(pattern))
        if not data_paths:
            raise FileNotFoundError(
                f"No US30 dataset files found with glob '{pattern}' from '{root}'."
            )

        data_by_tf: Dict[str, pd.DataFrame] = {}
        for p in data_paths:
            rows: List[Tuple[str, float, float, float, float, float]] = []
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 6:
                        continue
                    if parts[0].strip().lower() == "time":
                        continue
                    try:
                        rows.append(
                            (
                                parts[0],
                                float(parts[1]),
                                float(parts[2]),
                                float(parts[3]),
                                float(parts[4]),
                                float(parts[5]),
                            )
                        )
                    except ValueError:
                        continue

            if not rows:
                continue
            raw = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
            ts = pd.to_datetime(raw["time"], errors="coerce")
            df = raw[["open", "high", "low", "close", "volume"]].copy()
            df.index = ts
            df = df.dropna().astype(np.float64)

            name = p.name.upper()
            tf = "M5"
            for candidate in ("M1", "M5", "M15", "M30", "H1", "H4", "D1"):
                if candidate in name:
                    tf = candidate
                    break
            data_by_tf[tf] = df

        if not data_by_tf:
            raise ValueError("US30 dataset exists but does not contain required OHLCV columns.")
        return data_by_tf

    def _current_window(self) -> pd.DataFrame:
        left = max(0, self.current_idx - self.window_size)
        return self.current_df.iloc[left : self.current_idx + 1]

    def _build_feature_matrix(self, window: pd.DataFrame) -> np.ndarray:
        returns = window["close"].pct_change().fillna(0.0)
        hl_range = (window["high"] - window["low"]) / np.maximum(window["close"], 1e-12)
        oc_delta = (window["close"] - window["open"]) / np.maximum(window["open"], 1e-12)
        vol_chg = window["volume"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
        trend_5 = window["close"].pct_change(5).fillna(0.0)
        trend_20 = window["close"].pct_change(20).fillna(0.0)
        roll_std = returns.rolling(20).std().fillna(0.0)
        volume_z = (window["volume"] - window["volume"].rolling(20).mean()) / (
            window["volume"].rolling(20).std().replace(0, 1.0)
        )
        volume_z = volume_z.fillna(0.0)
        high_dist = (window["high"].rolling(50).max() - window["close"]) / np.maximum(
            window["close"], 1e-12
        )
        low_dist = (window["close"] - window["low"].rolling(50).min()) / np.maximum(
            window["close"], 1e-12
        )
        cyc = np.sin(np.linspace(0.0, 8.0 * np.pi, len(window)))

        feats = np.column_stack(
            [
                returns.to_numpy(np.float64),
                hl_range.to_numpy(np.float64),
                oc_delta.to_numpy(np.float64),
                vol_chg.to_numpy(np.float64),
                trend_5.to_numpy(np.float64),
                trend_20.to_numpy(np.float64),
                roll_std.to_numpy(np.float64),
                volume_z.to_numpy(np.float64),
                high_dist.fillna(0.0).to_numpy(np.float64),
                low_dist.fillna(0.0).to_numpy(np.float64),
                cyc.astype(np.float64),
            ]
        )
        return feats

    def _regime_probabilities(self, returns: np.ndarray, force_update: bool = False) -> np.ndarray:
        if returns.size < 30:
            return np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        if not force_update and self.current_idx - self._last_regime_update_idx < 10:
            return self._cached_regime_probs
        self.hmm.fit(returns[-200:])
        hmm_probs = self.hmm.predict_proba(returns[-200:])
        likelihood = np.clip(hmm_probs, 1e-6, 1.0)
        if np.std(returns[-50:]) > np.std(returns[-200:]) * 1.25:
            likelihood[MarketRegime.HIGH_VOL] *= 1.2
        posterior = self.bayesian.update(likelihood)
        self._cached_regime_probs = posterior.astype(np.float64)
        self._last_regime_update_idx = self.current_idx
        return self._cached_regime_probs

    def _compute_state(self) -> np.ndarray:
        window = self._current_window()
        closes = window["close"].to_numpy(np.float64)
        returns = np.diff(np.log(np.maximum(closes, 1e-12)))
        if returns.size < 5:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        regime_probs = self._regime_probabilities(returns)

        feature_matrix = self._build_feature_matrix(window)
        pca_ready = feature_matrix[-min(len(feature_matrix), 200) :]
        pca_reduced = self.pca.fit_transform(pca_ready)
        pca_tail = pca_reduced[-1] if pca_reduced.ndim == 2 else np.zeros(10, dtype=np.float64)
        pca_tail = np.pad(pca_tail, (0, max(0, 10 - pca_tail.shape[0])))[:10]

        self.arima.fit(closes[-200:] if closes.size > 200 else closes)
        self.garch.fit(returns[-300:] if returns.size > 300 else returns)
        arima_forecast = self.arima.forecast(closes, steps=1)[-1]
        garch_vol = self.garch.current_vol(returns)

        mc_paths = self.mc_state.simulate_gbm(
            S0=float(closes[-1]),
            mu=float(np.mean(returns)) * 252,
            sigma=float(max(garch_vol, 1e-6)) * np.sqrt(252),
            T=1.0 / 252.0,
            n_steps=30,
        )
        mc_stats = self.mc.path_statistics(mc_paths)
        mc_state = np.array(
            [
                mc_stats["mean_final"],
                mc_stats["std_final"],
                float(np.quantile(mc_paths[:, -1], 0.05)),
                float(np.quantile(mc_paths[:, -1], 0.95)),
            ],
            dtype=np.float64,
        )

        eq = np.array(self.equity_curve[-120:], dtype=np.float64)
        eq_slope = _safe_polyfit_slope(eq) if eq.size > 2 else 0.0
        dd = _max_drawdown(eq) if eq.size > 2 else 0.0
        position_state = np.array([self.position_pnl, eq_slope, dd], dtype=np.float64)

        spread_now = float((window["high"].iloc[-1] - window["low"].iloc[-1]) / max(closes[-1], 1e-12))
        spread_path = self.microstructure.simulate_spread(max(spread_now, 1e-6), T=1 / 252, n_steps=20)
        spread_estimate = np.array([float(np.mean(spread_path[-5:]))], dtype=np.float64)

        obs = np.concatenate(
            [
                regime_probs,
                pca_tail,
                np.array([garch_vol, arima_forecast], dtype=np.float64),
                mc_state,
                position_state,
                spread_estimate,
            ]
        ).astype(np.float32)
        return obs

    def _performance_metrics(self) -> Dict[str, float]:
        equity = np.array(self.equity_curve, dtype=np.float64)
        if equity.size < 20:
            return {"sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, "profit_factor": 0.0, "expectancy": 0.0}

        rets = np.diff(equity) / np.maximum(equity[:-1], 1e-12)
        mean_r = float(np.mean(rets))
        std_r = float(np.std(rets))
        downside = rets[rets < 0]
        downside_std = float(np.std(downside)) if downside.size else 1e-12
        sharpe = mean_r / max(std_r, 1e-12) * np.sqrt(252)
        sortino = mean_r / max(downside_std, 1e-12) * np.sqrt(252)
        annual_return = mean_r * 252
        mdd = _max_drawdown(equity)
        calmar = annual_return / max(mdd, 1e-6)
        gains = rets[rets > 0]
        losses = np.abs(rets[rets < 0])
        profit_factor = float(np.sum(gains) / max(np.sum(losses), 1e-12))
        expectancy = float(np.mean(rets))
        return {
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "calmar": float(calmar),
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "max_drawdown": float(mdd),
        }

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        super().reset(seed=seed)
        tf = options.get("timeframe", self.timeframe) if options else self.timeframe
        self.current_df = self.dataset.get(tf, next(iter(self.dataset.values())))

        max_start = max(len(self.current_df) - self.episode_bars - 2, self.window_size + 1)
        self.start_idx = int(self.rng.integers(self.window_size + 1, max_start))
        self.current_idx = self.start_idx
        self.end_idx = min(len(self.current_df) - 2, self.start_idx + self.episode_bars)

        self.position_direction = 0
        self.position_entry = 0.0
        self.position_pnl = 0.0
        self.equity = 10000.0
        self.equity_curve = [self.equity]
        self.rewards = []
        self.last_info = {}
        self._cached_regime_probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        self._last_regime_update_idx = self.current_idx

        return self._compute_state(), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), 0.5, 2.0)
        multipliers = dict(zip(self.param_names, action.tolist()))

        window = self._current_window()
        closes = window["close"].to_numpy(np.float64)
        current_price = float(closes[-1])
        next_price_realized = float(self.current_df.iloc[self.current_idx + 1]["close"])
        returns = np.diff(np.log(np.maximum(closes, 1e-12)))

        self.arima.fit(closes[-200:] if closes.size > 200 else closes)
        self.garch.fit(returns[-300:] if returns.size > 300 else returns)
        sigma = float(max(self.garch.current_vol(returns), 1e-6))
        mu = float(np.mean(returns))

        mc_paths = self.mc.simulate_gbm(
            current_price,
            mu * 252.0,
            sigma * np.sqrt(252),
            T=1.0 / 252.0,
            n_steps=25,
        )
        mc_ou = self.mc.simulate_ou(
            current_price,
            theta=1.0,
            mu=float(np.mean(closes[-50:])),
            sigma=sigma,
            T=1.0 / 252.0,
            n_steps=25,
        )
        mc_blend_target = float(0.7 * np.mean(mc_paths[:, -1]) + 0.3 * np.mean(mc_ou[:, -1]))

        log_drift = ItoLemma.log_price_drift(mu * 252.0, sigma * np.sqrt(252))
        girsanov = GirsanovTransform(mu * 252.0, 0.0, sigma * np.sqrt(252))
        rn_paths = girsanov.risk_neutral_paths(current_price, T=1.0 / 252.0, n_steps=25, n_paths=1000)
        rn_target = float(np.mean(rn_paths[:, -1]))

        arima_next = float(self.arima.forecast(closes, steps=1)[-1])
        directional_hint = arima_next - current_price
        self.position_direction = 1 if directional_hint >= 0 else -1
        model_next = float(0.4 * arima_next + 0.4 * mc_blend_target + 0.2 * rn_target + log_drift * current_price / 252)

        spread_estimate = float(np.mean(self.microstructure.simulate_spread(0.0002, T=1 / 252.0, n_steps=10)))
        bid = current_price - spread_estimate / 2
        ask = current_price + spread_estimate / 2

        plan_like = SimpleNamespace(
            direction=self.position_direction,
            position_size=max(0.01, 0.1 * multipliers["position_size_multiplier"]),
            entry=current_price,
        )
        execution = self.execution_engine.execute_market(plan_like, bid, ask)
        executed_price = execution.fill_price if execution.filled else current_price

        pred_pnl = self.position_direction * (model_next - executed_price) * multipliers["risk_multiplier"]
        realized_pnl = self.position_direction * (next_price_realized - executed_price) * multipliers["risk_multiplier"]
        slippage_cost = execution.execution_cost if execution.filled else spread_estimate

        self.position_pnl = realized_pnl - slippage_cost
        self.equity += self.position_pnl
        self.equity_curve.append(self.equity)

        perf = self._performance_metrics()
        calmar = perf.get("calmar", 0.0)
        sortino = perf.get("sortino", 0.0)
        expectancy = perf.get("expectancy", 0.0)
        dd = perf.get("max_drawdown", 0.0)

        expectancy_check = self.expectancy.check_trade(
            entry=current_price,
            target=model_next,
            stop=next_price_realized,
            sigma=max(sigma, 1e-6),
            T=1.0 / 252.0,
        )
        rr = expectancy_check.get("reward_risk_ratio", 0.0)
        microstructure_capture = 1.0 if pred_pnl * realized_pnl > 0 else 0.0

        reward = (
            1.4 * calmar
            + 0.7 * (sortino * expectancy * 10_000.0)
            + 0.15 * rr
            - 10.0 * max(0.0, dd - 0.05)
            - 0.5 * (1.0 - microstructure_capture)
        )

        vol_spike = float(window["volume"].iloc[-1] > window["volume"].tail(50).mean() * 1.2)
        overnight_reversion = float((self.current_idx % 96) < 24 and abs(directional_hint) < np.std(returns) * current_price)
        if vol_spike > 0 and overnight_reversion > 0 and np.max(self._regime_probabilities(returns)) > 0.4:
            reward += 0.35

        self.rewards.append(float(reward))
        self.current_idx += 1
        done = self.current_idx >= self.end_idx

        info = {
            "equity": float(self.equity),
            "position_pnl": float(self.position_pnl),
            "calmar": float(calmar),
            "sortino": float(sortino),
            "sharpe": float(perf.get("sharpe", 0.0)),
            "profit_factor": float(perf.get("profit_factor", 0.0)),
            "expectancy": float(expectancy),
            "drawdown": float(dd),
            "microstructure_capture": float(microstructure_capture),
            "multipliers": multipliers,
        }
        self.last_info = info

        obs = self._compute_state() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(reward), bool(done), False, info

    def current_state(self) -> np.ndarray:
        return self._compute_state()

