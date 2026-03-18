from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

try:
    from autonomous_quant_trading_bot.rl.drl_optimizer import DRLOptimizer
    from autonomous_quant_trading_bot.evolution.autoresearch import Autoresearch
    from autonomous_quant_trading_bot.orchestrator.sub_agents import build_sub_agents
except ImportError:
    from rl.drl_optimizer import DRLOptimizer
    from evolution.autoresearch import Autoresearch
    from orchestrator.sub_agents import build_sub_agents

if TYPE_CHECKING:
    from ..main import AutonomousBot

logger = logging.getLogger("trading_bot")

try:
    from openai import AsyncOpenAI
    from agents import Agent, Runner, OpenAIChatCompletionsModel

    AGENTS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    AsyncOpenAI = Agent = Runner = OpenAIChatCompletionsModel = object
    AGENTS_AVAILABLE = False


def _load_orchestrator_skill() -> str:
    root = Path(__file__).resolve().parents[2]
    skill_path = root / "Orchestrator skill.md"
    if skill_path.exists():
        return skill_path.read_text(encoding="utf-8", errors="ignore")
    return (
        "You are US30_Quant_Orchestrator. Manage Analyze->Plan->Risk->Execute->Position->Journal, "
        "trigger RL and autoresearch adaptively, and keep strict risk guardrails."
    )


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


@dataclass
class OrchestratorDecision:
    run_drl_training: bool
    run_autoresearch: bool
    force_risk_down: bool
    reason: str


class TradingOrchestrator:
    """
    Top-level project manager agent for trading flow control.
    Uses OpenAI Agents SDK + Gemini-compatible endpoint when available.
    Falls back to deterministic local orchestration if SDK is unavailable.
    """

    def __init__(self, bot: "AutonomousBot", config: Dict | None = None) -> None:
        self.bot = bot
        self.config = config or {}
        self.skill_text = _load_orchestrator_skill()
        self._last_drl_trigger = 0.0
        self._last_research_trigger = 0.0
        self._last_regime = None
        self._last_metrics: Dict[str, float] = {}

        self.drl_optimizer = DRLOptimizer(self.config)
        self.autoresearch = Autoresearch(self.config)

        self.model = None
        self.agent = None
        self.sub_agents = {}
        self._setup_agents()

    def _setup_agents(self) -> None:
        if not AGENTS_AVAILABLE:
            logger.warning("OpenAI Agents SDK not installed. Using local orchestrator fallback.")
            return

        orch_cfg = self.config.get("orchestrator", {})
        api_key = orch_cfg.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set. Using local orchestrator fallback.")
            return

        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        model_name = orch_cfg.get("gemini_model", "gemini-2.5-pro-exp")
        self.model = OpenAIChatCompletionsModel(model=model_name, client=client)
        self.sub_agents = build_sub_agents(self.skill_text, model=self.model)

        self.agent = Agent(
            name="US30_Quant_Orchestrator",
            instructions=self.skill_text,
            model=self.model,
            tools=[
                self.run_analyze_phase,
                self.run_plan_phase,
                self.run_execute_phase,
                self.run_risk_phase,
                self.run_position_phase,
                self.run_journal_phase,
                self.trigger_autoresearch,
                self.trigger_drl_training,
                self.get_current_math_state,
                self.evaluate_edge_metrics,
            ],
        )

    def get_current_math_state(self) -> Dict[str, Any]:
        ohlcv = self.bot.collector.fetch(self.bot.symbol, bars=400)
        if len(ohlcv) < 60:
            return {}
        returns = np.diff(np.log(ohlcv["close"].values.astype(np.float64)))
        regime = self.bot.regime_detector.detect(
            returns,
            self.bot.recognizer.session_timer.session_weight(datetime.utcnow()),
        )
        feats = self.bot.collector.build_features(ohlcv).tail(1)
        return {
            "regime": regime.regime.name,
            "regime_probs": regime.probabilities.tolist(),
            "garch_volatility": _safe_float(regime.volatility),
            "dominant_cycle": _safe_float(regime.dominant_cycle),
            "latest_features": feats.to_dict("records")[0] if len(feats) else {},
        }

    def evaluate_edge_metrics(self) -> Dict[str, float]:
        summary = self.bot.journal.performance_summary()
        if not summary:
            return {}
        pf = _safe_float(summary.get("profit_factor", 0.0))
        expectancy = _safe_float(summary.get("expectancy", 0.0))
        calmar = _safe_float(summary.get("calmar_ratio", 0.0))
        win_rate = _safe_float(summary.get("win_rate", 0.0))
        sharpe_proxy = _safe_float(summary.get("total_pnl", 0.0)) / max(_safe_float(summary.get("max_drawdown", 1.0)), 1e-6)
        sortino_proxy = sharpe_proxy if sharpe_proxy > 0 else 0.0
        metrics = {
            "expectancy": expectancy,
            "profit_factor": pf,
            "win_rate": win_rate,
            "sharpe": sharpe_proxy,
            "sortino": sortino_proxy,
            "calmar": calmar,
            "max_drawdown": _safe_float(summary.get("max_drawdown", 0.0)),
        }
        self._last_metrics = metrics
        return metrics

    def trigger_drl_training(self) -> Dict[str, Any]:
        report = self.drl_optimizer.train_drl(num_timesteps=20_000, eval_freq=5_000)
        self._last_drl_trigger = time.time()
        return {
            "accepted": report.accepted,
            "reason": report.reason,
            "model_path": report.model_path,
            "metrics": report.metrics,
            "out_of_sample_improvement": report.out_of_sample_improvement,
        }

    def trigger_autoresearch(self) -> Dict[str, Any]:
        ohlcv = self.bot.collector.fetch(self.bot.symbol, bars=6000)
        if len(ohlcv) == 0:
            return {"ok": False, "reason": "no_data"}
        best = self.autoresearch.run_overnight(ohlcv, n_cycles=2)
        self._last_research_trigger = time.time()
        return {
            "ok": True,
            "param_id": best.param_id,
            "calmar": best.calmar_ratio,
            "pnl": best.total_pnl,
        }

    def _default_decision(self, regime_name: str, metrics: Dict[str, float]) -> OrchestratorDecision:
        calmar = metrics.get("calmar", 0.0)
        max_dd = metrics.get("max_drawdown", 0.0)
        regime_flip = self._last_regime is not None and regime_name != self._last_regime
        self._last_regime = regime_name

        run_drl = regime_flip or ((time.time() - self._last_drl_trigger) / 3600.0 >= 4.0)
        run_research = regime_flip or ((time.time() - self._last_research_trigger) / 3600.0 >= 24.0)
        force_risk_down = max_dd > 0.05 or calmar < 1.0
        reason = "fallback_policy"
        return OrchestratorDecision(run_drl, run_research, force_risk_down, reason)

    async def _agent_decision(self, payload: Dict[str, Any]) -> OrchestratorDecision:
        if self.agent is None or Runner is object:
            return self._default_decision(payload.get("regime", "RANGING"), payload.get("metrics", {}))

        prompt = (
            "Decide flow-control flags for this cycle. Return strict JSON with keys: "
            "run_drl_training, run_autoresearch, force_risk_down, reason.\n"
            f"Input: {json.dumps(payload)}"
        )
        try:
            result = await Runner.run(self.agent, input=prompt)
            output = getattr(result, "final_output", "") if result is not None else ""
            if isinstance(output, str):
                data = json.loads(output)
            elif isinstance(output, dict):
                data = output
            else:
                data = {}
            return OrchestratorDecision(
                run_drl_training=bool(data.get("run_drl_training", False)),
                run_autoresearch=bool(data.get("run_autoresearch", False)),
                force_risk_down=bool(data.get("force_risk_down", False)),
                reason=str(data.get("reason", "agent_decision")),
            )
        except Exception as e:
            logger.warning("Agent decision failed, using fallback policy: %s", e)
            return self._default_decision(payload.get("regime", "RANGING"), payload.get("metrics", {}))

    def run_analyze_phase(self, ohlcv: pd.DataFrame, current_time: datetime):
        return self.bot._phase_analyze(ohlcv, current_time)

    def run_plan_phase(self, signal, ohlcv: pd.DataFrame, regime):
        return self.bot._phase_plan(signal, ohlcv, regime)

    def run_risk_phase(self, plan, current_time: datetime, regime):
        assessed = self.bot._phase_risk(plan, current_time, regime)
        return assessed

    def run_execute_phase(self, plan):
        return self.bot._phase_execute(plan)

    def run_position_phase(self, regime, recent_prices):
        return self.bot._phase_position(regime, recent_prices)

    def run_journal_phase(self, trade_record=None, signal=None, regime=None):
        return self.bot._phase_journal(trade_record, signal, regime)

    def run_cycle(self, ohlcv: pd.DataFrame, current_time: datetime) -> None:
        signal, regime = self.run_analyze_phase(ohlcv, current_time)
        recent = list(ohlcv["close"].values[-20:])
        trade_record = self.run_position_phase(regime, recent)
        self.run_journal_phase(trade_record, signal, regime)

        metrics = self.evaluate_edge_metrics()
        payload = {
            "timestamp": current_time.isoformat(),
            "regime": regime.regime.name,
            "confidence": regime.confidence,
            "metrics": metrics,
            "risk_guardrails": {"max_risk_per_trade": 0.02, "max_portfolio_dd": 0.05},
        }
        decision = asyncio.run(self._agent_decision(payload))

        if decision.force_risk_down:
            self.bot.risk_mgr.max_risk_per_trade = min(self.bot.risk_mgr.max_risk_per_trade, 0.005)
            logger.info("Orchestrator force risk-down enabled (%s)", decision.reason)

        if signal and self.bot.symbol not in self.bot.pos_mgr.open_positions:
            plan = self.run_plan_phase(signal, ohlcv, regime)
            plan = self.run_risk_phase(plan, current_time, regime)
            self.run_execute_phase(plan)

        if decision.run_drl_training:
            drl_result = self.trigger_drl_training()
            logger.info("Orchestrator DRL trigger: %s", drl_result)
        if decision.run_autoresearch:
            ar_result = self.trigger_autoresearch()
            logger.info("Orchestrator autoresearch trigger: %s", ar_result)


def build_orchestrator_agent(bot: "AutonomousBot", config: Dict | None = None) -> TradingOrchestrator:
    return TradingOrchestrator(bot=bot, config=config or {})

