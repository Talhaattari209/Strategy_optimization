"""
Journal — CSV + internal logging, performance attribution via OLS + SVD.
Tracks every trade and provides factor-based performance analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

from ..math_engine.linear_algebra import OLSRegression, PCAReducer


@dataclass
class JournalEntry:
    timestamp: str
    symbol: str
    direction: int
    entry: float
    exit: float
    size: float
    pnl: float
    pnl_pips: float
    session: str
    level_type: str
    candle_type: str
    regime: str
    confidence: float
    reason: str
    duration_min: float


class Journal:
    """
    Trade journal with performance attribution.
    Logs to CSV and provides OLS/SVD-based feature importance.
    """

    def __init__(self, results_dir: str = "results") -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._entries: List[JournalEntry] = []
        self._ols = OLSRegression()
        self._pca = PCAReducer(n_components=5)

    def log_trade(
        self,
        trade_record: Dict,
        signal_info: Dict | None = None,
    ) -> None:
        sig = signal_info or {}
        entry = JournalEntry(
            timestamp=trade_record.get("close_time", datetime.utcnow().isoformat()),
            symbol=trade_record.get("symbol", ""),
            direction=trade_record.get("direction", 0),
            entry=trade_record.get("entry", 0),
            exit=trade_record.get("exit", 0),
            size=trade_record.get("size", 0),
            pnl=trade_record.get("pnl", 0),
            pnl_pips=trade_record.get("pnl_pips", 0),
            session=sig.get("session", ""),
            level_type=sig.get("level_type", ""),
            candle_type=sig.get("candle_type", ""),
            regime=sig.get("regime", ""),
            confidence=sig.get("confidence", 0),
            reason=sig.get("reason", ""),
            duration_min=trade_record.get("duration_minutes", 0),
        )
        self._entries.append(entry)

    def save_csv(self, filename: str = "trade_journal.csv") -> Path:
        if not self._entries:
            return self.results_dir / filename

        records = [
            {
                "timestamp": e.timestamp, "symbol": e.symbol, "direction": e.direction,
                "entry": e.entry, "exit": e.exit, "size": e.size,
                "pnl": e.pnl, "pnl_pips": e.pnl_pips,
                "session": e.session, "level_type": e.level_type,
                "candle_type": e.candle_type, "regime": e.regime,
                "confidence": e.confidence, "reason": e.reason,
                "duration_min": e.duration_min,
            }
            for e in self._entries
        ]
        df = pd.DataFrame(records)
        path = self.results_dir / filename
        df.to_csv(path, index=False)
        return path

    def performance_summary(self) -> Dict[str, float]:
        if not self._entries:
            return {}

        pnls = [e.pnl for e in self._entries]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_pnl = sum(pnls)
        win_rate = len(wins) / len(pnls) if pnls else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(sum(wins)) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
        expectancy = np.mean(pnls) if pnls else 0

        # Max drawdown on equity curve
        equity = np.cumsum(pnls)
        peak = np.maximum.accumulate(equity)
        drawdown = peak - equity
        max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0

        # Calmar ratio
        annual_return = total_pnl * (252 / max(len(pnls), 1))
        calmar = annual_return / max_dd if max_dd > 0 else float("inf")

        return {
            "total_trades": len(pnls),
            "total_pnl": round(total_pnl, 2),
            "win_rate": round(win_rate, 4),
            "avg_win": round(float(avg_win), 2),
            "avg_loss": round(float(avg_loss), 2),
            "profit_factor": round(profit_factor, 2),
            "expectancy": round(float(expectancy), 2),
            "max_drawdown": round(max_dd, 2),
            "calmar_ratio": round(calmar, 4),
        }

    def performance_by_session(self) -> Dict[str, Dict[str, float]]:
        result = {}
        sessions = set(e.session for e in self._entries if e.session)
        for session in sessions:
            entries = [e for e in self._entries if e.session == session]
            pnls = [e.pnl for e in entries]
            result[session] = {
                "trades": len(pnls),
                "total_pnl": round(sum(pnls), 2),
                "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls), 4) if pnls else 0,
            }
        return result

    def performance_by_level(self) -> Dict[str, Dict[str, float]]:
        result = {}
        levels = set(e.level_type for e in self._entries if e.level_type)
        for level in levels:
            entries = [e for e in self._entries if e.level_type == level]
            pnls = [e.pnl for e in entries]
            result[level] = {
                "trades": len(pnls),
                "total_pnl": round(sum(pnls), 2),
                "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls), 4) if pnls else 0,
            }
        return result

    def feature_attribution(self) -> Dict[str, float]:
        """OLS + PCA-based feature importance on trade outcomes."""
        if len(self._entries) < 10:
            return {}

        features = []
        pnls = []
        for e in self._entries:
            feat = [e.confidence, e.duration_min, e.direction, e.size]
            features.append(feat)
            pnls.append(e.pnl)

        X = np.array(features)
        y = np.array(pnls)

        self._ols.fit(X, y)
        betas = self._ols.beta[1:]

        names = ["confidence", "duration", "direction", "size"]
        importance = np.abs(betas) * np.std(X, axis=0)
        total = importance.sum()
        if total > 0:
            importance /= total

        return {name: round(float(imp), 4) for name, imp in zip(names, importance)}

    @property
    def entries(self) -> List[JournalEntry]:
        return list(self._entries)

    @property
    def total_trades(self) -> int:
        return len(self._entries)
