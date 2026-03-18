"""
Autonomous Quant Trading Bot — Main Entry Point
6-phase loop: Analyze → Plan → Execute → Risk → Position Management → Journal
Fully autonomous after start. Supports backtest and live modes.
"""
from __future__ import annotations

import argparse
import logging
import time
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional

from data.collector import DataCollector
from core.pattern_recognizer import PatternRecognizer
from core.regime_detector import RegimeDetector, RegimeState
from core.signal_planner import SignalPlanner
from core.execution_engine import ExecutionEngine
from core.risk_manager import RiskManager
from core.position_manager import PositionManager
from core.journal import Journal
from backtester.engine import Backtester
from evolution.autoresearch import Autoresearch
from rl.drl_optimizer import DRLOptimizer
from orchestrator.trading_orchestrator import build_orchestrator_agent
from utils.helpers import load_config, setup_logging, ConsoleDisplay, Timer

logger = logging.getLogger("trading_bot")


class AutonomousBot:
    """
    The complete autonomous trading bot.
    Runs the 6-phase cycle: Analyze → Plan → Execute → Risk → Position → Journal
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config = load_config(config_path)
        self.display = ConsoleDisplay()

        self.collector = DataCollector(self.config)
        self.recognizer = PatternRecognizer(self.config)
        self.regime_detector = RegimeDetector(self.config)
        self.planner = SignalPlanner(self.config)
        self.exec_engine = ExecutionEngine(self.config)
        self.risk_mgr = RiskManager(self.config)
        self.pos_mgr = PositionManager(self.config)
        self.journal = Journal(str(Path("results")))

        self.symbol: str = self.config.get("broker", {}).get("default_symbol", "EURUSD")
        self.balance: float = 10000.0
        self._last_regime_fit: float = 0.0
        self._last_drl_train: float = 0.0
        self._last_regime_name: str | None = None
        self._running: bool = False
        self.drl_optimizer = DRLOptimizer(self.config)
        orch_enabled = self.config.get("orchestrator", {}).get("enabled", False)
        self.orchestrator = build_orchestrator_agent(self, self.config) if orch_enabled else None

    def initialize(self, initial_balance: float = 10000.0) -> None:
        """Load history and calibrate all models."""
        self.balance = initial_balance
        self.risk_mgr.set_initial_balance(initial_balance)
        logger.info(f"Bot initialized: {self.symbol}, balance=${initial_balance:,.2f}")

        try:
            data = self.collector.fetch(self.symbol, bars=500)
            if len(data) > 50:
                returns = np.diff(np.log(data["close"].values.astype(np.float64)))
                self.regime_detector.fit(returns)
                logger.info("Regime detector calibrated on historical data")
        except Exception as e:
            logger.warning(f"Could not fetch initial data: {e}")

        if self.config.get("drl", {}).get("drl_enabled", False):
            logger.info("DRL optimizer initialized (enabled=%s)", self.drl_optimizer.enabled)

    def _phase_analyze(self, ohlcv: pd.DataFrame, current_time: datetime):
        """Phase 1: Analyze — detect session, levels, candles, generate base signal."""
        with Timer("Phase 1: Analyze"):
            signal = self.recognizer.generate_signal(ohlcv, current_time)

            returns = np.diff(np.log(ohlcv["close"].values.astype(np.float64)))
            session_feats = self.recognizer.session_timer.session_weight(current_time)
            regime = self.regime_detector.detect(returns, session_feats)

            if signal:
                self.display.print_signal(signal.to_dict())
            self.display.print_regime({
                "regime": regime.regime.name,
                "confidence": regime.confidence,
                "volatility": regime.volatility,
            })

            return signal, regime

    def _phase_plan(self, signal, ohlcv: pd.DataFrame, regime: RegimeState):
        """Phase 2: Plan — refine entry, TP, SL using math engine."""
        if signal is None:
            return None

        with Timer("Phase 2: Plan"):
            plan = self.planner.plan(signal, ohlcv, regime, self.balance)
            if plan:
                logger.info(
                    f"Plan: {'LONG' if plan.direction == 1 else 'SHORT'} "
                    f"entry={plan.entry:.5f} TP={plan.take_profit:.5f} SL={plan.stop_loss:.5f} "
                    f"size={plan.position_size} conf={plan.confidence:.2%}"
                )
            return plan

    def _phase_risk(self, plan, current_time: datetime, regime: RegimeState):
        """Phase 3: Risk check — approve or reject the trade."""
        if plan is None:
            return None

        with Timer("Phase 3: Risk"):
            assessment = self.risk_mgr.assess_trade(plan, self.balance, current_time, regime)
            if not assessment.approved:
                logger.info(f"Trade REJECTED: {assessment.reason}")
                return None
            plan.position_size = assessment.adjusted_size
            logger.info(f"Trade APPROVED: size={assessment.adjusted_size}, DD={assessment.current_drawdown_pct:.2%}")
            return plan

    def _phase_execute(self, plan):
        """Phase 4: Execute — place the order."""
        if plan is None:
            return None

        with Timer("Phase 4: Execute"):
            tick = self.collector.get_latest_tick(self.symbol)
            bid = tick.get("bid", plan.entry - 0.00010)
            ask = tick.get("ask", plan.entry + 0.00010)

            if bid == 0 or ask == 0:
                bid = plan.entry - 0.00010
                ask = plan.entry + 0.00010

            result = self.exec_engine.execute_market(plan, bid, ask)
            if result.filled:
                self.pos_mgr.open_position(
                    self.symbol, plan.direction, result.fill_price,
                    plan.position_size, plan.stop_loss, plan.take_profit,
                )
                self.risk_mgr.register_position(self.symbol, plan.position_size, plan.direction)
                logger.info(f"FILLED at {result.fill_price:.5f}, slippage={result.slippage:.6f}")
            return result

    def _phase_position(self, regime: RegimeState, recent_prices: list[float]):
        """Phase 5: Position management — trailing, partials, exits."""
        with Timer("Phase 5: Position Mgmt"):
            for sym, pos in list(self.pos_mgr.open_positions.items()):
                tick = self.collector.get_latest_tick(sym)
                current_price = tick.get("bid", 0) if tick.get("bid", 0) > 0 else pos.current_price
                self.pos_mgr.update_price(sym, current_price)

                action = self.pos_mgr.manage(sym, regime, recent_prices)
                if action.action == "close":
                    record = self.pos_mgr.close_position(sym, current_price)
                    if record:
                        self.balance += record["pnl"]
                        self.risk_mgr.close_position(sym)
                        self.display.print_trade(record)
                        return record
                elif action.action == "trail_update":
                    logger.debug(f"Trail updated: {action.new_stop:.5f}")
        return None

    def _phase_journal(self, trade_record=None, signal=None, regime=None):
        """Phase 6: Journal — log everything."""
        if trade_record:
            signal_info = {}
            if signal:
                signal_info = signal.to_dict()
            if regime:
                signal_info["regime"] = regime.regime.name
            self.journal.log_trade(trade_record, signal_info)

        # Periodic summary
        if self.journal.total_trades > 0 and self.journal.total_trades % 10 == 0:
            summary = self.journal.performance_summary()
            self.display.print_summary(summary)

    def tick(self, ohlcv: pd.DataFrame, current_time: datetime) -> None:
        """Run one complete 6-phase cycle."""
        if self.orchestrator is not None:
            self.orchestrator.run_cycle(ohlcv, current_time)
            return

        # Re-fit regime detector periodically
        hours_since = (time.time() - self._last_regime_fit) / 3600
        if self.regime_detector.needs_refit(hours_since):
            returns = np.diff(np.log(ohlcv["close"].values.astype(np.float64)))
            self.regime_detector.fit(returns)
            self._last_regime_fit = time.time()

        # Phase 1: Analyze
        signal, regime = self._phase_analyze(ohlcv, current_time)
        regime_changed = self._last_regime_name is not None and self._last_regime_name != regime.regime.name
        self._last_regime_name = regime.regime.name

        # DRL online refresh every 4h or on regime flip.
        if self.config.get("drl", {}).get("drl_enabled", False):
            hours_since_drl = (time.time() - self._last_drl_train) / 3600
            if regime_changed or hours_since_drl >= 4.0:
                try:
                    report = self.drl_optimizer.continue_online_learning(num_timesteps=5_000)
                    logger.info(
                        "DRL online update: accepted=%s, calmar=%.3f, oos=%.2f%% (%s)",
                        report.accepted,
                        report.metrics.get("calmar", 0.0),
                        report.out_of_sample_improvement * 100.0,
                        report.reason,
                    )
                except Exception as e:
                    logger.warning(f"DRL online update failed: {e}")
                self._last_drl_train = time.time()

        # Phase 5: Position management (runs always)
        recent = list(ohlcv["close"].values[-20:])
        trade_record = self._phase_position(regime, recent)
        self._phase_journal(trade_record, signal, regime)

        # Phases 2-4 only if we have a signal and no open position
        if signal and self.symbol not in self.pos_mgr.open_positions:
            plan = self._phase_plan(signal, ohlcv, regime)
            plan = self._phase_risk(plan, current_time, regime)
            self._phase_execute(plan)

    def run_live(self, poll_seconds: int = 60) -> None:
        """Run the bot in live mode, polling for new data."""
        self._running = True
        logger.info(f"Starting live trading on {self.symbol}")

        while self._running:
            try:
                ohlcv = self.collector.fetch(self.symbol, bars=200)
                if len(ohlcv) > 0:
                    current_time = datetime.utcnow()
                    self.tick(ohlcv, current_time)
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                self._running = False
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)

            if self._running:
                time.sleep(poll_seconds)

        # Save journal on exit
        path = self.journal.save_csv()
        logger.info(f"Journal saved to {path}")
        summary = self.journal.performance_summary()
        if summary:
            self.display.print_summary(summary)

    def run_backtest(
        self, data: pd.DataFrame, initial_balance: float = 10000.0
    ):
        """Run backtest on historical data."""
        backtester = Backtester(self.config)
        result = backtester.run(data, self.symbol, initial_balance)

        self.display.print_summary(result.summary)

        mc = backtester.monte_carlo_robustness(result)
        if mc:
            logger.info("Monte Carlo robustness test:")
            for k, v in mc.items():
                logger.info(f"  {k}: {v}")

        return result

    def run_autoresearch(self, data: pd.DataFrame, n_cycles: int = 10):
        """Run Karpathy-style autoresearch overnight."""
        auto = Autoresearch(self.config)
        best = auto.run_overnight(data, n_cycles)
        logger.info(f"Best params: Calmar={best.calmar_ratio:.4f}, PnL={best.total_pnl:.2f}")
        return best

    def stop(self) -> None:
        self._running = False


def main():
    parser = argparse.ArgumentParser(description="Autonomous Quant Trading Bot")
    parser.add_argument("--mode", choices=["live", "backtest", "autoresearch"], default="backtest")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--data", default=None, help="Path to CSV data for backtest")
    parser.add_argument("--balance", type=float, default=10000.0)
    parser.add_argument("--cycles", type=int, default=10, help="Autoresearch cycles")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    bot = AutonomousBot(args.config)
    if args.symbol:
        bot.symbol = args.symbol

    bot.initialize(args.balance)

    if args.mode == "live":
        bot.run_live()

    elif args.mode == "backtest":
        if args.data:
            data = bot.collector.load_csv(args.data)
        else:
            logger.info("No CSV provided, fetching from MT5...")
            data = bot.collector.fetch(bot.symbol, bars=5000)

        if len(data) == 0:
            logger.error("No data available for backtesting.")
            sys.exit(1)

        result = bot.run_backtest(data, args.balance)
        logger.info(f"Backtest complete: {result.summary}")

    elif args.mode == "autoresearch":
        if args.data:
            data = bot.collector.load_csv(args.data)
        else:
            data = bot.collector.fetch(bot.symbol, bars=5000)

        if len(data) == 0:
            logger.error("No data available for autoresearch.")
            sys.exit(1)

        bot.run_autoresearch(data, args.cycles)


if __name__ == "__main__":
    main()
