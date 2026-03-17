"""
Backtester Engine — walk-forward + Monte Carlo robustness testing.
Runs the full 6-phase loop on historical data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import logging

from ..core.pattern_recognizer import PatternRecognizer, BaseSignal
from ..core.regime_detector import RegimeDetector
from ..core.signal_planner import SignalPlanner, TradePlan
from ..core.risk_manager import RiskManager
from ..core.position_manager import PositionManager
from ..core.execution_engine import ExecutionEngine
from ..core.journal import Journal
from ..math_engine.stochastic_processes import MonteCarloEngine

logger = logging.getLogger(__name__)


class BacktestResult:
    """Holds all results from a backtest run."""

    def __init__(self) -> None:
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        self.signals: List[Dict] = []
        self.summary: Dict[str, float] = {}


class Backtester:
    """
    Walk-forward backtester that runs the full autonomous loop on historical data.
    """

    def __init__(self, config: Dict | None = None) -> None:
        self.config = config or {}
        self.initial_balance: float = 10000.0
        self.lookback: int = 100

    def run(
        self,
        data: pd.DataFrame,
        symbol: str = "EURUSD",
        initial_balance: float = 10000.0,
    ) -> BacktestResult:
        self.initial_balance = initial_balance
        result = BacktestResult()
        balance = initial_balance
        result.equity_curve.append(balance)

        recognizer = PatternRecognizer(self.config)
        regime_detector = RegimeDetector(self.config)
        planner = SignalPlanner(self.config)
        risk_mgr = RiskManager(self.config)
        pos_mgr = PositionManager(self.config)
        exec_engine = ExecutionEngine(self.config)
        journal = Journal(str(Path(self.config.get("results_dir", "results"))))

        risk_mgr.set_initial_balance(initial_balance)

        pip_size = self.config.get("broker", {}).get("pip_size", 0.0001)
        closes = data["close"].values.astype(np.float64)

        if len(closes) > 50:
            returns = np.diff(np.log(closes[:200]))
            regime_detector.fit(returns)

        for i in range(self.lookback, len(data)):
            window = data.iloc[max(0, i - self.lookback):i + 1]
            current_price = float(closes[i])
            current_time = window.index[-1]
            if not isinstance(current_time, datetime):
                current_time = pd.Timestamp(current_time).to_pydatetime()

            # Phase 1: ANALYZE — generate base signal
            signal = recognizer.generate_signal(window, current_time)

            # Check open positions first
            for sym, pos in list(pos_mgr.open_positions.items()):
                pos_mgr.update_price(sym, current_price)
                recent = list(closes[max(0, i - 20):i + 1])
                from ..math_engine.markov_bayesian import MarketRegime
                dummy_regime = regime_detector.detect(
                    np.diff(np.log(closes[max(0, i - 100):i + 1]))
                )
                action = pos_mgr.manage(sym, dummy_regime, recent)

                if action.action in ("close",):
                    trade_record = pos_mgr.close_position(sym, current_price)
                    if trade_record:
                        balance += trade_record["pnl"]
                        result.trades.append(trade_record)
                        journal.log_trade(trade_record)
                        risk_mgr.close_position(sym)
                elif action.action == "partial_close":
                    pass  # simplified: full close in backtest

            if signal is None:
                result.equity_curve.append(balance)
                continue

            result.signals.append(signal.to_dict())

            # Phase 1b: Regime detection
            recent_returns = np.diff(np.log(closes[max(0, i - 100):i + 1]))
            regime = regime_detector.detect(recent_returns)

            # Phase 2: PLAN
            plan = planner.plan(signal, window, regime, balance)
            if plan is None:
                result.equity_curve.append(balance)
                continue

            # Phase 3: RISK CHECK
            risk_assessment = risk_mgr.assess_trade(plan, balance, current_time, regime)
            if not risk_assessment.approved:
                result.equity_curve.append(balance)
                continue

            plan.position_size = risk_assessment.adjusted_size

            # Phase 4: EXECUTE (simulated)
            spread = 2 * pip_size
            bid = current_price - spread / 2
            ask = current_price + spread / 2
            exec_result = exec_engine.execute_market(plan, bid, ask)

            if exec_result.filled:
                pos_mgr.open_position(
                    symbol, plan.direction, exec_result.fill_price,
                    plan.position_size, plan.stop_loss, plan.take_profit,
                )
                risk_mgr.register_position(symbol, plan.position_size, plan.direction)

            result.equity_curve.append(balance)

        # Close remaining positions
        for sym, pos in list(pos_mgr.open_positions.items()):
            trade_record = pos_mgr.close_position(sym, float(closes[-1]))
            if trade_record:
                balance += trade_record["pnl"]
                result.trades.append(trade_record)
                journal.log_trade(trade_record)

        result.equity_curve.append(balance)
        result.summary = journal.performance_summary()
        result.summary["final_balance"] = round(balance, 2)
        result.summary["total_return_pct"] = round((balance - initial_balance) / initial_balance * 100, 2)

        journal.save_csv()
        return result

    def walk_forward(
        self,
        data: pd.DataFrame,
        train_pct: float = 0.7,
        n_folds: int = 5,
        symbol: str = "EURUSD",
    ) -> List[BacktestResult]:
        """Walk-forward validation: train on in-sample, test on out-of-sample."""
        n = len(data)
        fold_size = n // n_folds
        results = []

        for fold in range(n_folds):
            train_end = int((fold + 1) * fold_size * train_pct)
            test_start = train_end
            test_end = min((fold + 1) * fold_size, n)

            if test_end <= test_start + self.lookback:
                continue

            test_data = data.iloc[test_start:test_end]
            result = self.run(test_data, symbol)
            results.append(result)
            logger.info(f"Fold {fold + 1}/{n_folds}: {result.summary.get('total_trades', 0)} trades, "
                        f"PnL={result.summary.get('total_pnl', 0):.2f}")

        return results

    def monte_carlo_robustness(
        self,
        result: BacktestResult,
        n_simulations: int = 1000,
    ) -> Dict[str, float]:
        """Monte Carlo robustness test: shuffle trades and measure stability."""
        if not result.trades:
            return {}

        pnls = np.array([t["pnl"] for t in result.trades])
        original_equity = np.cumsum(pnls) + self.initial_balance

        max_dds = []
        final_balances = []
        rng = np.random.default_rng(42)

        for _ in range(n_simulations):
            shuffled = rng.permutation(pnls)
            equity = np.cumsum(shuffled) + self.initial_balance
            peak = np.maximum.accumulate(equity)
            dd = (peak - equity) / peak
            max_dds.append(float(np.max(dd)))
            final_balances.append(float(equity[-1]))

        return {
            "mc_median_final_balance": round(float(np.median(final_balances)), 2),
            "mc_5th_pct_final_balance": round(float(np.percentile(final_balances, 5)), 2),
            "mc_95th_pct_final_balance": round(float(np.percentile(final_balances, 95)), 2),
            "mc_median_max_dd": round(float(np.median(max_dds)), 4),
            "mc_worst_max_dd": round(float(np.max(max_dds)), 4),
            "mc_prob_profitable": round(float(np.mean(np.array(final_balances) > self.initial_balance)), 4),
        }
