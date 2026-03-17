"""
Karpathy-style Autoresearch Loop — autonomous self-improvement.
Mutates parameters, backtests, promotes winners.
NEVER removes or overrides the core 4-factor edge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from ..backtester.engine import Backtester
from ..core.pattern_recognizer import (
    SESSION_WEIGHTS, LEVEL_WEIGHTS, CANDLE_SIGNAL, MIN_CONFIDENCE_TO_TRADE,
)

logger = logging.getLogger(__name__)


@dataclass
class ParameterSet:
    """A complete set of mutable parameters with lineage tracking."""
    generation: int
    parent_id: str
    param_id: str
    session_weights: Dict[str, float]
    level_weights: Dict[str, float]
    candle_signal: Dict[str, Dict[str, Any]]
    min_confidence: float
    pip_tolerance: int
    candle_thresholds: Dict[str, float]
    calmar_ratio: float = 0.0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "parent_id": self.parent_id,
            "param_id": self.param_id,
            "session_weights": self.session_weights,
            "level_weights": self.level_weights,
            "min_confidence": self.min_confidence,
            "pip_tolerance": self.pip_tolerance,
            "calmar_ratio": self.calmar_ratio,
            "total_pnl": self.total_pnl,
            "win_rate": self.win_rate,
            "timestamp": self.timestamp,
        }


class Autoresearch:
    """
    Autonomous parameter evolution loop.
    Mutates parameters → Backtests → Promotes winners.
    Respects the inviolable 4-factor core edge.
    """

    def __init__(self, config: Dict | None = None) -> None:
        self.config = config or {}
        auto_cfg = self.config.get("autoresearch", {})
        self.calmar_target: float = auto_cfg.get("calmar_target", 2.0)
        self.max_mutations: int = auto_cfg.get("max_mutations_per_cycle", 5)
        self.improvement_threshold: float = 0.05  # 5% improvement required

        self.results_dir = Path(self.config.get("results_dir", "results")) / "evolution"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self._rng = np.random.default_rng(42)
        self._current_best: Optional[ParameterSet] = None
        self._history: List[ParameterSet] = []
        self._generation: int = 0

    def _create_base_params(self) -> ParameterSet:
        """Create initial parameter set from hard-coded client edge values."""
        return ParameterSet(
            generation=0,
            parent_id="origin",
            param_id="gen0_base",
            session_weights=dict(SESSION_WEIGHTS),
            level_weights=dict(LEVEL_WEIGHTS),
            candle_signal=copy.deepcopy(CANDLE_SIGNAL),
            min_confidence=MIN_CONFIDENCE_TO_TRADE,
            pip_tolerance=self.config.get("levels", {}).get("pip_tolerance", 10),
            candle_thresholds={
                "pinbar_wick_body_ratio": self.config.get("candle_patterns", {}).get("pinbar", {}).get("wick_body_ratio", 2.0),
                "doji_body_ratio": self.config.get("candle_patterns", {}).get("doji", {}).get("body_range_ratio", 0.1),
                "engulfing_min_body": self.config.get("candle_patterns", {}).get("engulfing", {}).get("min_body_ratio", 0.6),
                "rejection_wick_pct": self.config.get("candle_patterns", {}).get("rejection", {}).get("wick_pct_of_range", 0.66),
                "impulse_body_pct": self.config.get("candle_patterns", {}).get("impulse", {}).get("body_pct_of_range", 0.75),
            },
            timestamp=datetime.utcnow().isoformat(),
        )

    def _mutate(self, parent: ParameterSet) -> ParameterSet:
        """Create a mutated child parameter set."""
        child = copy.deepcopy(parent)
        child.generation = self._generation
        child.parent_id = parent.param_id
        child.param_id = f"gen{self._generation}_{self._rng.integers(0, 99999):05d}"
        child.timestamp = datetime.utcnow().isoformat()

        mutation_type = self._rng.choice([
            "session_weights", "level_weights", "candle_thresholds",
            "min_confidence", "pip_tolerance",
        ])

        if mutation_type == "session_weights":
            key = self._rng.choice(list(child.session_weights.keys()))
            delta = self._rng.uniform(-0.15, 0.15)
            child.session_weights[key] = np.clip(child.session_weights[key] + delta, 0.1, 1.0)

        elif mutation_type == "level_weights":
            key = self._rng.choice(list(child.level_weights.keys()))
            delta = self._rng.uniform(-0.15, 0.15)
            child.level_weights[key] = np.clip(child.level_weights[key] + delta, 0.1, 1.0)

        elif mutation_type == "candle_thresholds":
            key = self._rng.choice(list(child.candle_thresholds.keys()))
            current = child.candle_thresholds[key]
            factor = self._rng.uniform(0.8, 1.2)
            child.candle_thresholds[key] = round(current * factor, 4)

        elif mutation_type == "min_confidence":
            delta = self._rng.uniform(-0.10, 0.10)
            child.min_confidence = np.clip(child.min_confidence + delta, 0.3, 0.9)

        elif mutation_type == "pip_tolerance":
            delta = self._rng.integers(-5, 6)
            child.pip_tolerance = int(np.clip(child.pip_tolerance + delta, 3, 30))

        return child

    def _apply_params(self, params: ParameterSet) -> Dict:
        """Convert ParameterSet to a config dict usable by the trading system."""
        cfg = copy.deepcopy(self.config)
        cfg["_session_weights"] = params.session_weights
        cfg["_level_weights"] = params.level_weights
        cfg["_min_confidence"] = params.min_confidence
        cfg.setdefault("levels", {})["pip_tolerance"] = params.pip_tolerance

        candle_cfg = cfg.setdefault("candle_patterns", {})
        candle_cfg.setdefault("pinbar", {})["wick_body_ratio"] = params.candle_thresholds.get("pinbar_wick_body_ratio", 2.0)
        candle_cfg.setdefault("doji", {})["body_range_ratio"] = params.candle_thresholds.get("doji_body_ratio", 0.1)
        candle_cfg.setdefault("engulfing", {})["min_body_ratio"] = params.candle_thresholds.get("engulfing_min_body", 0.6)
        candle_cfg.setdefault("rejection", {})["wick_pct_of_range"] = params.candle_thresholds.get("rejection_wick_pct", 0.66)
        candle_cfg.setdefault("impulse", {})["body_pct_of_range"] = params.candle_thresholds.get("impulse_body_pct", 0.75)

        return cfg

    def _evaluate(self, params: ParameterSet, data: pd.DataFrame) -> ParameterSet:
        """Run walk-forward backtest and score the parameter set."""
        cfg = self._apply_params(params)
        backtester = Backtester(cfg)
        results = backtester.walk_forward(data, train_pct=0.7, n_folds=3)

        if not results:
            return params

        all_trades = []
        for r in results:
            all_trades.extend(r.trades)

        if not all_trades:
            return params

        pnls = [t["pnl"] for t in all_trades]
        equity = np.cumsum(pnls)
        peak = np.maximum.accumulate(equity)
        drawdown = peak - equity
        max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 1.0

        annual_return = sum(pnls) * (252 / max(len(pnls), 1))
        calmar = annual_return / max_dd if max_dd > 0 else 0.0

        params.calmar_ratio = calmar
        params.total_pnl = sum(pnls)
        params.win_rate = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0

        return params

    def run_cycle(self, data: pd.DataFrame) -> ParameterSet:
        """Run one complete autoresearch cycle."""
        self._generation += 1
        logger.info(f"Autoresearch cycle {self._generation} starting")

        if self._current_best is None:
            self._current_best = self._create_base_params()
            self._current_best = self._evaluate(self._current_best, data)
            self._history.append(self._current_best)
            logger.info(f"Baseline Calmar: {self._current_best.calmar_ratio:.4f}")

        candidates: List[ParameterSet] = []
        for _ in range(self.max_mutations):
            child = self._mutate(self._current_best)
            child = self._evaluate(child, data)
            candidates.append(child)
            logger.info(f"  Mutation {child.param_id}: Calmar={child.calmar_ratio:.4f}")

        best_candidate = max(candidates, key=lambda p: p.calmar_ratio)
        improvement = (
            (best_candidate.calmar_ratio - self._current_best.calmar_ratio)
            / abs(self._current_best.calmar_ratio)
            if abs(self._current_best.calmar_ratio) > 0
            else 1.0
        )

        if improvement > self.improvement_threshold:
            logger.info(
                f"Promoting {best_candidate.param_id}: Calmar {self._current_best.calmar_ratio:.4f} -> "
                f"{best_candidate.calmar_ratio:.4f} (+{improvement:.1%})"
            )
            self._current_best = best_candidate
        else:
            logger.info(f"No improvement above threshold ({self.improvement_threshold:.0%}), keeping current best")

        self._history.append(self._current_best)
        self._save_history()
        return self._current_best

    def run_overnight(self, data: pd.DataFrame, n_cycles: int = 10) -> ParameterSet:
        """Run multiple autoresearch cycles (designed for overnight execution)."""
        logger.info(f"Starting overnight autoresearch: {n_cycles} cycles")
        for cycle in range(n_cycles):
            result = self.run_cycle(data)
            logger.info(f"Cycle {cycle + 1}/{n_cycles} complete: Calmar={result.calmar_ratio:.4f}")

            if result.calmar_ratio >= self.calmar_target:
                logger.info(f"Target Calmar ({self.calmar_target}) reached!")
                break

        return self._current_best

    def _save_history(self) -> None:
        path = self.results_dir / "evolution_history.json"
        history = [p.to_dict() for p in self._history]
        with open(path, "w") as f:
            json.dump(history, f, indent=2)

    @property
    def best_params(self) -> Optional[ParameterSet]:
        return self._current_best

    @property
    def history(self) -> List[ParameterSet]:
        return list(self._history)
