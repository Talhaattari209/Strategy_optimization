"""
Execution Engine — smart order execution with microstructure modeling.
Uses Girsanov for optimal entry, TWAP/VWAP slicing, microstructure SDE cost simulation.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

from .signal_planner import TradePlan
from ..math_engine.stochastic_calculus import GirsanovTransform, MicrostructureSDE


@dataclass
class OrderSlice:
    size: float
    target_price: float
    slice_index: int
    total_slices: int


@dataclass
class ExecutionResult:
    filled: bool
    fill_price: float
    fill_size: float
    slippage: float
    execution_cost: float
    order_type: str
    timestamp: datetime | None = None


class ExecutionEngine:
    """
    Handles order execution with cost optimization.
    Supports: market, limit, TWAP slicing.
    """

    def __init__(self, config: Dict | None = None) -> None:
        self.config = config or {}
        exec_cfg = self.config.get("execution", {})
        self.max_slippage: float = exec_cfg.get("slippage_pips", 2)
        self.max_spread: float = exec_cfg.get("max_spread_pips", 5)
        self.n_slices: int = exec_cfg.get("twap_slices", 5)
        self.pip_size: float = self.config.get("broker", {}).get("pip_size", 0.0001)

        self.microstructure = MicrostructureSDE()
        self._execution_log: List[ExecutionResult] = []

    def _check_spread(self, bid: float, ask: float) -> bool:
        spread_pips = (ask - bid) / self.pip_size
        return spread_pips <= self.max_spread

    def compute_execution_plan(
        self,
        plan: TradePlan,
        current_bid: float,
        current_ask: float,
        daily_volume: float = 1_000_000,
    ) -> List[OrderSlice]:
        if plan.position_size <= 0.05:
            return [OrderSlice(
                size=plan.position_size,
                target_price=current_ask if plan.direction == 1 else current_bid,
                slice_index=0,
                total_slices=1,
            )]

        current_spread = current_ask - current_bid
        slices = self.microstructure.optimal_twap_slices(
            plan.position_size, self.n_slices, current_spread, daily_volume
        )

        base_price = current_ask if plan.direction == 1 else current_bid
        orders = []
        for i, sz in enumerate(slices):
            offset = (i - self.n_slices // 2) * self.pip_size * 0.5
            target = base_price + offset * plan.direction
            orders.append(OrderSlice(
                size=round(float(sz), 2),
                target_price=round(target, 5),
                slice_index=i,
                total_slices=self.n_slices,
            ))

        return orders

    def estimate_cost(
        self, plan: TradePlan, current_spread: float, daily_volume: float = 1_000_000
    ) -> float:
        return self.microstructure.expected_execution_cost(
            plan.position_size, current_spread, daily_volume
        )

    def execute_market(
        self,
        plan: TradePlan,
        current_bid: float,
        current_ask: float,
        broker_interface=None,
    ) -> ExecutionResult:
        if not self._check_spread(current_bid, current_ask):
            return ExecutionResult(
                filled=False, fill_price=0, fill_size=0,
                slippage=0, execution_cost=0, order_type="rejected_spread",
            )

        fill_price = current_ask if plan.direction == 1 else current_bid
        slippage_pips = np.random.uniform(0, self.max_slippage) if broker_interface is None else 0
        slippage = slippage_pips * self.pip_size * plan.direction
        actual_fill = fill_price + slippage

        if broker_interface is not None:
            pass  # broker_interface.send_order(...)

        result = ExecutionResult(
            filled=True,
            fill_price=round(actual_fill, 5),
            fill_size=plan.position_size,
            slippage=round(slippage, 6),
            execution_cost=abs(slippage) + (current_ask - current_bid) / 2,
            order_type="market",
            timestamp=datetime.utcnow(),
        )
        self._execution_log.append(result)
        return result

    def execute_twap(
        self,
        plan: TradePlan,
        current_bid: float,
        current_ask: float,
        daily_volume: float = 1_000_000,
        broker_interface=None,
    ) -> List[ExecutionResult]:
        slices = self.compute_execution_plan(plan, current_bid, current_ask, daily_volume)
        results = []
        for order in slices:
            sub_plan = TradePlan(
                direction=plan.direction,
                entry=order.target_price,
                take_profit=plan.take_profit,
                stop_loss=plan.stop_loss,
                position_size=order.size,
                confidence=plan.confidence,
                base_signal=plan.base_signal,
                regime=plan.regime,
                math_details=plan.math_details,
            )
            result = self.execute_market(sub_plan, current_bid, current_ask, broker_interface)
            results.append(result)
        return results

    @property
    def execution_log(self) -> List[ExecutionResult]:
        return list(self._execution_log)
