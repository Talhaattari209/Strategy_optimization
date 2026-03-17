"""
Pattern Recognizer — Core Client Edge Confluence Engine.
THE UNBREAKABLE FOUNDATION: Session + Level + Candle Behavior = Trade Signal.

This module combines ALL four factors of the client's edge:
1. What time (session) is it?
2. At what level?
3. Price behavior (candle type) at that level
4. The confluence of 1+2+3 = base trade signal

NEVER delete or override the core logic in this module.
Autoresearch may mutate parameters but must preserve the 4-factor structure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

from .session_timer import SessionTimer, SessionState, Session
from .level_detector import LevelDetector, LevelState, LevelType, PriceLevel
from .candle_analyzer import CandleAnalyzer, CandlePattern, CandleType, Candle
from .market_structure import MarketStructure, MarketStructureState


@dataclass
class BaseSignal:
    """The core signal dict as specified by the client."""
    bias: str  # "bullish" | "bearish" | "neutral"
    confidence: float  # 0.0-1.0
    reason: str
    level_type: str
    session: str
    candle_type: str
    timestamp: pd.Timestamp | None = None
    level_price: float = 0.0
    entry_price: float = 0.0
    raw_factors: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bias": self.bias,
            "confidence": self.confidence,
            "reason": self.reason,
            "level_type": self.level_type,
            "session": self.session,
            "candle_type": self.candle_type,
            "timestamp": str(self.timestamp) if self.timestamp else None,
            "level_price": self.level_price,
            "entry_price": self.entry_price,
        }


# ── CLIENT EDGE START — do not remove ──────────────────────────────────────

# Session-Level-Candle confluence weights (tunable by autoresearch)
SESSION_WEIGHTS: Dict[str, float] = {
    "Asia": 0.6,
    "London": 1.0,
    "NewYork": 0.95,
    "Sydney": 0.5,
}

# Overlaps get a boost
OVERLAP_BOOST: float = 0.15

# Level-type priority weights
LEVEL_WEIGHTS: Dict[str, float] = {
    "HO_Week": 1.0,
    "LO_Week": 1.0,
    "HOD": 0.9,
    "LOD": 0.9,
    "HO_Session": 0.75,
    "LO_Session": 0.75,
    "Close_of_Day": 0.7,
    "CHOCH": 0.95,
    "BOS": 0.85,
}

# Candle pattern signal strength
CANDLE_SIGNAL: Dict[str, Dict[str, Any]] = {
    "bullish_pinbar": {"bias": "bullish", "strength": 0.8},
    "bearish_pinbar": {"bias": "bearish", "strength": 0.8},
    "bullish_engulfing": {"bias": "bullish", "strength": 0.9},
    "bearish_engulfing": {"bias": "bearish", "strength": 0.9},
    "doji": {"bias": "neutral", "strength": 0.4},
    "bullish_rejection": {"bias": "bullish", "strength": 0.7},
    "bearish_rejection": {"bias": "bearish", "strength": 0.7},
    "bullish_impulse": {"bias": "bullish", "strength": 0.85},
    "bearish_impulse": {"bias": "bearish", "strength": 0.85},
    "inside_bar": {"bias": "neutral", "strength": 0.5},
}

# Confluence thresholds
MIN_CONFIDENCE_TO_TRADE: float = 0.55

# ── CLIENT EDGE END ────────────────────────────────────────────────────────


class PatternRecognizer:
    """
    Core confluence engine: combines session + level + candle → base signal.
    """

    def __init__(self, config: Dict | None = None) -> None:
        self.config = config or {}
        self.session_timer = SessionTimer(self.config)
        self.level_detector = LevelDetector(self.config)
        self.candle_analyzer = CandleAnalyzer(self.config)
        self.market_structure = MarketStructure(self.config)

        self._signal_history: List[BaseSignal] = []

    def _compute_session_score(self, session_state: SessionState) -> float:
        """Score the current session context."""
        if not session_state.active_sessions:
            return 0.0

        scores = [SESSION_WEIGHTS.get(s.value, 0.5) for s in session_state.active_sessions]
        score = max(scores)

        if session_state.is_overlap:
            score += OVERLAP_BOOST

        return min(score, 1.0)

    def _compute_level_score(self, level_state: LevelState) -> tuple:
        """Score how significant the current level is. Returns (score, best_level)."""
        if level_state.at_level is None:
            return 0.0, None

        level = level_state.at_level
        weight = LEVEL_WEIGHTS.get(level.level_type.value, 0.5)
        return weight * level.strength, level

    def _compute_candle_score(self, patterns: List[CandlePattern]) -> tuple:
        """Score candle behavior. Returns (score, bias, best_pattern)."""
        if not patterns:
            return 0.0, "neutral", None

        best = max(patterns, key=lambda p: p.strength)
        signal = CANDLE_SIGNAL.get(best.candle_type.value, {"bias": "neutral", "strength": 0.3})
        return signal["strength"], signal["bias"], best

    def _compute_structure_context(
        self, structure_state: MarketStructureState, bias: str
    ) -> float:
        """Boost or penalize based on market structure alignment."""
        if structure_state.trend == bias:
            return 0.1
        if structure_state.trend != "ranging" and structure_state.trend != bias:
            return -0.1
        return 0.0

    def generate_signal(
        self,
        ohlcv: pd.DataFrame,
        current_time: datetime,
    ) -> Optional[BaseSignal]:
        """
        Core 4-factor confluence engine.
        ohlcv must have columns: open, high, low, close, volume with DatetimeIndex.
        """
        if len(ohlcv) < 10:
            return None

        # Factor 1: Session
        session_state = self.session_timer.get_state(current_time)
        session_score = self._compute_session_score(session_state)
        if session_score == 0:
            return None

        session_name = session_state.primary_session.value if session_state.primary_session else "Unknown"

        # Update levels with latest data
        for i in range(len(ohlcv)):
            row = ohlcv.iloc[i]
            ts = ohlcv.index[i] if isinstance(ohlcv.index[i], pd.Timestamp) else pd.Timestamp(ohlcv.index[i])
            self.level_detector.update(
                row["high"], row["low"], row["close"], ts, session_name
            )

        # Update market structure
        highs = ohlcv["high"].values
        lows = ohlcv["low"].values
        closes = ohlcv["close"].values
        timestamps = [
            ts if isinstance(ts, pd.Timestamp) else pd.Timestamp(ts)
            for ts in ohlcv.index
        ]
        structure_state = self.market_structure.update(highs, lows, closes, timestamps)

        # Add CHOCH/BOS as structure levels
        if structure_state.last_choch:
            lt = LevelType.CHOCH
            self.level_detector.add_structure_level(
                lt, structure_state.last_choch.broken_level, timestamps[-1]
            )
        if structure_state.last_bos:
            lt = LevelType.BOS
            self.level_detector.add_structure_level(
                lt, structure_state.last_bos.broken_level, timestamps[-1]
            )

        # Factor 2: Level
        current_price = float(closes[-1])
        level_state = self.level_detector.get_state(current_price, timestamps[-1])
        level_score, active_level = self._compute_level_score(level_state)

        if active_level is None:
            return None

        # Factor 3: Candle behavior at level
        candles = [
            Candle(float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]),
                   float(r.get("volume", 0)))
            for _, r in ohlcv.tail(5).iterrows()
        ]
        patterns = self.candle_analyzer.analyze_at_level(
            candles, active_level.price,
            self.config.get("broker", {}).get("pip_size", 0.0001),
            self.config.get("levels", {}).get("pip_tolerance", 10),
        )
        candle_score, candle_bias, best_pattern = self._compute_candle_score(patterns)

        if candle_bias == "neutral" and candle_score < 0.5:
            return None

        # Factor 4: Confluence — combine all scores
        raw_confidence = (
            session_score * 0.25
            + level_score * 0.35
            + candle_score * 0.30
        )

        # Structure context adjustment
        structure_adj = self._compute_structure_context(structure_state, candle_bias)
        raw_confidence += structure_adj * 0.10

        confidence = np.clip(raw_confidence, 0.0, 1.0)

        if confidence < MIN_CONFIDENCE_TO_TRADE:
            return None

        # Build the reason string
        overlap_str = " (overlap)" if session_state.is_overlap else ""
        candle_name = best_pattern.candle_type.value if best_pattern else "unknown"
        reason = f"{session_name}{overlap_str} + at {active_level.level_type.value} + {candle_name}"

        signal = BaseSignal(
            bias=candle_bias,
            confidence=float(confidence),
            reason=reason,
            level_type=active_level.level_type.value,
            session=session_name,
            candle_type=candle_name,
            timestamp=timestamps[-1],
            level_price=active_level.price,
            entry_price=current_price,
            raw_factors={
                "session_score": session_score,
                "level_score": level_score,
                "candle_score": candle_score,
                "structure_adj": structure_adj,
                "trend": structure_state.trend,
            },
        )

        self._signal_history.append(signal)
        return signal

    def get_signal_history(self) -> List[BaseSignal]:
        return list(self._signal_history)

    def get_features(self, current_price: float, current_time: datetime, ohlcv: pd.DataFrame) -> Dict[str, float]:
        """Flat feature vector for math engine consumption."""
        session_feats = self.session_timer.session_weight(current_time)
        ts = pd.Timestamp(current_time) if not isinstance(current_time, pd.Timestamp) else current_time
        level_feats = self.level_detector.level_features(current_price, ts)
        candles = [
            Candle(float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]),
                   float(r.get("volume", 0)))
            for _, r in ohlcv.tail(3).iterrows()
        ]
        candle_feats = self.candle_analyzer.candle_features(candles)
        structure_feats = self.market_structure.structure_features()

        features = {}
        features.update({f"session_{k}": v for k, v in session_feats.items()})
        features.update({f"level_{k}": v for k, v in level_feats.items()})
        features.update({f"candle_{k}": v for k, v in candle_feats.items()})
        features.update({f"structure_{k}": v for k, v in structure_feats.items()})
        return features
