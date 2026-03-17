"""
Risk Manager — dynamic sizing, portfolio VaR, news blackout, drawdown guard.
Volatility-scaled via Monte Carlo + covariance matrix.
Runs BEFORE and AFTER execution in the 6-phase loop.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from .signal_planner import TradePlan
from .regime_detector import RegimeState
from ..math_engine.stochastic_processes import MonteCarloEngine
from ..math_engine.linear_algebra import CovarianceMatrix


@dataclass
class RiskAssessment:
    approved: bool
    adjusted_size: float
    reason: str
    portfolio_var: float
    current_drawdown_pct: float
    risk_score: float


@dataclass
class PositionRisk:
    symbol: str
    size: float
    unrealized_pnl: float
    direction: int


class RiskManager:
    """
    Pre-trade and post-trade risk management.
    Controls: position sizing, max drawdown, correlation, VaR, news filter.
    """

    def __init__(self, config: Dict | None = None) -> None:
        self.config = config or {}
        risk_cfg = self.config.get("risk", {})

        self.max_risk_per_trade: float = risk_cfg.get("max_risk_per_trade_pct", 1.0) / 100.0
        self.max_daily_dd: float = risk_cfg.get("max_daily_drawdown_pct", 3.0) / 100.0
        self.max_total_dd: float = risk_cfg.get("max_total_drawdown_pct", 10.0) / 100.0
        self.max_correlated: int = risk_cfg.get("max_correlated_positions", 3)
        self.news_blackout_min: int = risk_cfg.get("news_blackout_minutes", 30)

        self.mc = MonteCarloEngine(n_paths=5000)

        self._peak_balance: float = 0.0
        self._daily_start_balance: float = 0.0
        self._open_positions: List[PositionRisk] = []
        self._news_times: List[datetime] = []
        self._last_day: int = -1

    def set_initial_balance(self, balance: float) -> None:
        self._peak_balance = balance
        self._daily_start_balance = balance

    def update_balance(self, current_balance: float, current_time: datetime) -> None:
        self._peak_balance = max(self._peak_balance, current_balance)
        day = current_time.day
        if day != self._last_day:
            self._daily_start_balance = current_balance
            self._last_day = day

    def add_news_event(self, event_time: datetime) -> None:
        self._news_times.append(event_time)

    def _is_news_blackout(self, current_time: datetime) -> bool:
        blackout = timedelta(minutes=self.news_blackout_min)
        for nt in self._news_times:
            if abs(current_time - nt) < blackout:
                return True
        return False

    def _current_drawdown(self, current_balance: float) -> float:
        if self._peak_balance <= 0:
            return 0.0
        return (self._peak_balance - current_balance) / self._peak_balance

    def _daily_drawdown(self, current_balance: float) -> float:
        if self._daily_start_balance <= 0:
            return 0.0
        return (self._daily_start_balance - current_balance) / self._daily_start_balance

    def _portfolio_var(self, returns_matrix: NDArray[np.float64] | None = None) -> float:
        if returns_matrix is None or returns_matrix.shape[1] < 2:
            return 0.0

        cov = CovarianceMatrix.compute(returns_matrix)
        n = len(self._open_positions)
        if n == 0:
            return 0.0

        weights = np.array([abs(p.size) for p in self._open_positions[:returns_matrix.shape[1]]])
        total = weights.sum()
        if total > 0:
            weights /= total

        portfolio_var = float(np.sqrt(weights @ cov[:n, :n] @ weights))
        return portfolio_var * 1.65  # 95% VaR

    def assess_trade(
        self,
        plan: TradePlan,
        current_balance: float,
        current_time: datetime,
        regime: RegimeState | None = None,
        returns_matrix: NDArray[np.float64] | None = None,
    ) -> RiskAssessment:
        self.update_balance(current_balance, current_time)

        # News blackout check
        if self._is_news_blackout(current_time):
            return RiskAssessment(False, 0.0, "News blackout period", 0.0, 0.0, 1.0)

        # Max drawdown check
        total_dd = self._current_drawdown(current_balance)
        if total_dd >= self.max_total_dd:
            return RiskAssessment(False, 0.0, f"Max drawdown reached: {total_dd:.1%}", 0.0, total_dd, 1.0)

        daily_dd = self._daily_drawdown(current_balance)
        if daily_dd >= self.max_daily_dd:
            return RiskAssessment(False, 0.0, f"Daily drawdown limit: {daily_dd:.1%}", 0.0, daily_dd, 0.9)

        # Correlated positions check
        same_direction = sum(1 for p in self._open_positions if p.direction == plan.direction)
        if same_direction >= self.max_correlated:
            return RiskAssessment(False, 0.0, f"Max correlated positions ({self.max_correlated})", 0.0, total_dd, 0.8)

        # Volatility-adjusted sizing
        adjusted_size = plan.position_size
        if regime and regime.regime.name == "HIGH_VOL":
            adjusted_size *= 0.5

        # Drawdown scaling: reduce size as drawdown increases
        dd_factor = max(0.25, 1.0 - total_dd / self.max_total_dd)
        adjusted_size *= dd_factor

        # Portfolio VaR
        pvar = self._portfolio_var(returns_matrix)

        risk_score = total_dd / self.max_total_dd if self.max_total_dd > 0 else 0.0

        return RiskAssessment(
            approved=True,
            adjusted_size=round(max(0.01, adjusted_size), 2),
            reason="Trade approved",
            portfolio_var=pvar,
            current_drawdown_pct=total_dd,
            risk_score=risk_score,
        )

    def register_position(self, symbol: str, size: float, direction: int) -> None:
        self._open_positions.append(PositionRisk(symbol, size, 0.0, direction))

    def close_position(self, symbol: str) -> None:
        self._open_positions = [p for p in self._open_positions if p.symbol != symbol]

    def update_positions(self, positions: List[PositionRisk]) -> None:
        self._open_positions = positions

    @property
    def open_exposure(self) -> float:
        return sum(abs(p.size) for p in self._open_positions)
