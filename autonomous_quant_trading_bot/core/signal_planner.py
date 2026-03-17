"""
Signal Planner — Plan phase.
Takes the base signal from PatternRecognizer and refines entry, TP, SL
using ARIMA, Monte Carlo, Ito's Lemma, and Black-Scholes risk-neutral check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Dict, Optional

from .pattern_recognizer import BaseSignal
from .regime_detector import RegimeState
from ..math_engine.time_series import ARIMA, GARCH
from ..math_engine.stochastic_processes import MonteCarloEngine, GeometricBrownianMotion, OrnsteinUhlenbeck
from ..math_engine.stochastic_calculus import ItoLemma, GirsanovTransform
from ..math_engine.finance_models import BlackScholes, ExpectancyCalculator


@dataclass
class TradePlan:
    direction: int  # 1 = long, -1 = short, 0 = no trade
    entry: float
    take_profit: float
    stop_loss: float
    position_size: float
    confidence: float
    base_signal: BaseSignal
    regime: str
    math_details: Dict[str, float]

    @property
    def reward_risk_ratio(self) -> float:
        reward = abs(self.take_profit - self.entry)
        risk = abs(self.entry - self.stop_loss)
        return reward / risk if risk > 0 else 0.0


class SignalPlanner:
    """
    Refines base signals with quantitative analysis.
    Uses math engine to compute optimal entry, TP, SL.
    """

    def __init__(self, config: Dict | None = None) -> None:
        self.config = config or {}
        self.pip_size: float = self.config.get("broker", {}).get("pip_size", 0.0001)
        self.risk_pct: float = self.config.get("risk", {}).get("max_risk_per_trade_pct", 1.0) / 100.0
        self.mc_paths: int = self.config.get("optimization", {}).get("monte_carlo_paths", 10000)

        self.arima = ARIMA(p=5, d=1, q=1)
        self.garch = GARCH()
        self.mc = MonteCarloEngine(n_paths=self.mc_paths)
        self.expectancy = ExpectancyCalculator()

    def plan(
        self,
        signal: BaseSignal,
        ohlcv: pd.DataFrame,
        regime: RegimeState,
        account_balance: float = 10000.0,
    ) -> Optional[TradePlan]:
        if signal.bias == "neutral":
            return None

        closes = ohlcv["close"].values.astype(np.float64)
        returns = np.diff(np.log(closes))

        if len(returns) < 20:
            return None

        # Fit models on recent data
        self.arima.fit(closes[-100:] if len(closes) > 100 else closes)
        self.garch.fit(returns[-200:] if len(returns) > 200 else returns)

        current_price = float(closes[-1])
        vol = self.garch.current_vol(returns)
        mu = float(np.mean(returns))

        # ARIMA forecast for direction confirmation
        arima_forecast = self.arima.forecast(closes, steps=5)
        arima_direction = 1 if arima_forecast[-1] > current_price else -1

        direction = 1 if signal.bias == "bullish" else -1

        # Monte Carlo simulation from current level
        T_hours = 8.0  # simulate to next session close
        T = T_hours / (252 * 24)  # annualized
        paths = self.mc.simulate_gbm(current_price, mu * 252, vol * np.sqrt(252), T, n_steps=100)
        mc_stats = self.mc.path_statistics(paths)

        # Ito-based TP/SL computation
        log_drift = ItoLemma.log_price_drift(mu * 252, vol * np.sqrt(252))

        # Girsanov risk-neutral check
        girsanov = GirsanovTransform(mu * 252, 0.0, vol * np.sqrt(252))
        rn_paths = girsanov.risk_neutral_paths(current_price, T, 100, n_paths=min(5000, self.mc_paths))

        # Compute TP and SL using MC expected extremes
        if direction == 1:
            tp_target = mc_stats["expected_max"]
            sl_target = mc_stats["expected_min"]
            prob_win = mc_stats["prob_positive_return"]
        else:
            tp_target = mc_stats["expected_min"]
            sl_target = mc_stats["expected_max"]
            prob_win = 1.0 - mc_stats["prob_positive_return"]

        # Ensure minimum 1.5:1 R:R
        reward = abs(tp_target - current_price)
        risk = abs(sl_target - current_price)
        if risk > 0 and reward / risk < 1.5:
            reward = risk * 1.5
            tp_target = current_price + direction * reward

        # BS risk-neutral expectancy at the level
        bs_expectancy = BlackScholes.risk_neutral_expectancy(
            current_price, tp_target, T, 0.0, vol * np.sqrt(252)
        )

        # Position sizing: risk-based
        risk_amount = account_balance * self.risk_pct
        pip_risk = abs(current_price - sl_target) / self.pip_size
        position_size = risk_amount / (pip_risk * self.pip_size * 100000) if pip_risk > 0 else 0.01
        position_size = round(max(0.01, min(position_size, 10.0)), 2)

        # Combined confidence
        arima_agreement = 1.0 if arima_direction == direction else 0.7
        regime_factor = 1.0
        if regime.regime.name == "HIGH_VOL":
            regime_factor = 0.8
        elif (direction == 1 and regime.regime.name == "TRENDING_UP") or \
             (direction == -1 and regime.regime.name == "TRENDING_DOWN"):
            regime_factor = 1.1

        final_confidence = np.clip(
            signal.confidence * 0.5
            + prob_win * 0.25
            + arima_agreement * 0.15
            + regime_factor * 0.10,
            0.0, 1.0,
        )

        return TradePlan(
            direction=direction,
            entry=current_price,
            take_profit=round(tp_target, 5),
            stop_loss=round(sl_target, 5),
            position_size=position_size,
            confidence=float(final_confidence),
            base_signal=signal,
            regime=regime.regime.name,
            math_details={
                "arima_forecast_5": float(arima_forecast[-1]),
                "garch_vol": vol,
                "mc_prob_win": prob_win,
                "mc_expected_max": mc_stats["expected_max"],
                "mc_expected_min": mc_stats["expected_min"],
                "mc_var_95": mc_stats["var_95"],
                "bs_expectancy": bs_expectancy,
                "log_drift": log_drift,
                "rr_ratio": reward / risk if risk > 0 else 0,
            },
        )
