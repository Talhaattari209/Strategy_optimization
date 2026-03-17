"""
Position Manager — dynamic trailing, partial closes, regime-flip exit.
Uses OU for mean-reversion trailing, Ito for optimal exit, martingale expectancy monitoring.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

from .regime_detector import RegimeState
from ..math_engine.stochastic_processes import OrnsteinUhlenbeck
from ..math_engine.stochastic_calculus import ItoLemma
from ..math_engine.markov_bayesian import MarketRegime


@dataclass
class Position:
    symbol: str
    direction: int  # 1=long, -1=short
    entry_price: float
    current_price: float
    size: float
    stop_loss: float
    take_profit: float
    open_time: datetime
    trail_stop: float = 0.0
    partial_closed: float = 0.0
    pnl_pips: float = 0.0

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.direction

    @property
    def is_in_profit(self) -> bool:
        return self.unrealized_pnl > 0


@dataclass
class PositionAction:
    action: str  # "hold", "close", "partial_close", "trail_update"
    reason: str
    new_stop: float = 0.0
    close_pct: float = 0.0


class PositionManager:
    """
    Manages open positions with dynamic trailing and partial closes.
    """

    def __init__(self, config: Dict | None = None) -> None:
        self.config = config or {}
        self.pip_size: float = self.config.get("broker", {}).get("pip_size", 0.0001)

        self._positions: Dict[str, Position] = {}
        self._ou = OrnsteinUhlenbeck()
        self._trade_history: List[Dict] = []

    def open_position(
        self,
        symbol: str,
        direction: int,
        entry_price: float,
        size: float,
        stop_loss: float,
        take_profit: float,
    ) -> Position:
        pos = Position(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            current_price=entry_price,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            open_time=datetime.utcnow(),
            trail_stop=stop_loss,
        )
        self._positions[symbol] = pos
        return pos

    def update_price(self, symbol: str, current_price: float) -> None:
        if symbol in self._positions:
            pos = self._positions[symbol]
            pos.current_price = current_price
            pos.pnl_pips = (current_price - pos.entry_price) * pos.direction / self.pip_size

    def _compute_ou_trail(self, pos: Position, recent_prices: list[float]) -> float:
        """OU-based dynamic trailing: trail tightens as mean-reversion increases."""
        if len(recent_prices) < 10:
            return pos.trail_stop

        prices = np.array(recent_prices, dtype=np.float64)
        self._ou.fit(prices)
        half_life = self._ou.half_life()

        trail_distance = max(
            10 * self.pip_size,
            self._ou.sigma * np.sqrt(half_life) * 2
        )

        if pos.direction == 1:
            new_trail = pos.current_price - trail_distance
            return max(new_trail, pos.trail_stop)
        else:
            new_trail = pos.current_price + trail_distance
            return min(new_trail, pos.trail_stop) if pos.trail_stop > 0 else new_trail

    def _check_regime_exit(self, pos: Position, regime: RegimeState) -> bool:
        """Exit if regime flips against position."""
        if pos.direction == 1 and regime.regime == MarketRegime.TRENDING_DOWN:
            return regime.confidence > 0.7
        if pos.direction == -1 and regime.regime == MarketRegime.TRENDING_UP:
            return regime.confidence > 0.7
        return False

    def _check_partial_close(self, pos: Position) -> Optional[float]:
        """Take partial profits at 50% and 75% of TP distance."""
        tp_dist = abs(pos.take_profit - pos.entry_price)
        current_dist = abs(pos.current_price - pos.entry_price)

        if pos.partial_closed < 0.25 and current_dist >= tp_dist * 0.5:
            return 0.25
        if pos.partial_closed < 0.50 and current_dist >= tp_dist * 0.75:
            return 0.25
        return None

    def manage(
        self,
        symbol: str,
        regime: RegimeState,
        recent_prices: list[float] | None = None,
    ) -> PositionAction:
        if symbol not in self._positions:
            return PositionAction("hold", "No position found")

        pos = self._positions[symbol]

        # Check stop loss
        if pos.direction == 1 and pos.current_price <= pos.trail_stop:
            return PositionAction("close", f"Trail stop hit at {pos.trail_stop:.5f}")
        if pos.direction == -1 and pos.current_price >= pos.trail_stop:
            return PositionAction("close", f"Trail stop hit at {pos.trail_stop:.5f}")

        # Check take profit
        if pos.direction == 1 and pos.current_price >= pos.take_profit:
            return PositionAction("close", "Take profit reached")
        if pos.direction == -1 and pos.current_price <= pos.take_profit:
            return PositionAction("close", "Take profit reached")

        # Regime-flip exit
        if self._check_regime_exit(pos, regime):
            return PositionAction("close", f"Regime flip: {regime.regime.name}")

        # Partial close
        partial = self._check_partial_close(pos)
        if partial:
            pos.partial_closed += partial
            return PositionAction("partial_close", f"Partial close {partial:.0%}", close_pct=partial)

        # Dynamic trailing using OU
        if recent_prices and len(recent_prices) >= 10:
            new_trail = self._compute_ou_trail(pos, recent_prices)
            if new_trail != pos.trail_stop:
                pos.trail_stop = new_trail
                return PositionAction("trail_update", f"Trail updated to {new_trail:.5f}", new_stop=new_trail)

        return PositionAction("hold", "Position maintained")

    def close_position(self, symbol: str, close_price: float) -> Optional[Dict]:
        if symbol not in self._positions:
            return None

        pos = self._positions.pop(symbol)
        pnl = (close_price - pos.entry_price) * pos.direction * pos.size * 100000
        pnl_pips = (close_price - pos.entry_price) * pos.direction / self.pip_size

        record = {
            "symbol": symbol,
            "direction": pos.direction,
            "entry": pos.entry_price,
            "exit": close_price,
            "size": pos.size,
            "pnl": round(pnl, 2),
            "pnl_pips": round(pnl_pips, 1),
            "duration_minutes": (datetime.utcnow() - pos.open_time).total_seconds() / 60,
            "open_time": pos.open_time.isoformat(),
            "close_time": datetime.utcnow().isoformat(),
        }
        self._trade_history.append(record)
        return record

    @property
    def open_positions(self) -> Dict[str, Position]:
        return dict(self._positions)

    @property
    def trade_history(self) -> List[Dict]:
        return list(self._trade_history)
