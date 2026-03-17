"""
Candle Analyzer — detects candle patterns at key levels.
Part of the client's core 4-factor edge: "Price behavior at that level."
Supported: pinbar, engulfing, doji, rejection, impulse, inside bar.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class CandleType(Enum):
    BULLISH_PINBAR = "bullish_pinbar"
    BEARISH_PINBAR = "bearish_pinbar"
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"
    DOJI = "doji"
    BULLISH_REJECTION = "bullish_rejection"
    BEARISH_REJECTION = "bearish_rejection"
    BULLISH_IMPULSE = "bullish_impulse"
    BEARISH_IMPULSE = "bearish_impulse"
    INSIDE_BAR = "inside_bar"
    NONE = "none"


@dataclass
class Candle:
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def body_pct(self) -> float:
        return self.body / self.range if self.range > 0 else 0.0

    @property
    def upper_wick_pct(self) -> float:
        return self.upper_wick / self.range if self.range > 0 else 0.0

    @property
    def lower_wick_pct(self) -> float:
        return self.lower_wick / self.range if self.range > 0 else 0.0


@dataclass
class CandlePattern:
    candle_type: CandleType
    direction: str  # "bullish", "bearish", "neutral"
    strength: float  # 0-1
    wick_body_ratio: float
    description: str


class CandleAnalyzer:
    """
    Detects candle type within tolerance of a level.
    Configurable thresholds via config dict.
    """

    def __init__(self, config: Dict | None = None) -> None:
        cfg = config or {}
        candle_cfg = cfg.get("candle_patterns", {})

        self.pinbar_wick_body_ratio: float = candle_cfg.get("pinbar", {}).get("wick_body_ratio", 2.0)
        self.doji_body_ratio: float = candle_cfg.get("doji", {}).get("body_range_ratio", 0.1)
        self.engulfing_min_body: float = candle_cfg.get("engulfing", {}).get("min_body_ratio", 0.6)
        self.rejection_wick_pct: float = candle_cfg.get("rejection", {}).get("wick_pct_of_range", 0.66)
        self.impulse_body_pct: float = candle_cfg.get("impulse", {}).get("body_pct_of_range", 0.75)

    def _detect_pinbar(self, c: Candle) -> Optional[CandlePattern]:
        if c.body < 1e-10:
            return None

        if c.lower_wick / c.body >= self.pinbar_wick_body_ratio and c.lower_wick > c.upper_wick:
            return CandlePattern(
                CandleType.BULLISH_PINBAR, "bullish",
                min(c.lower_wick / c.body / self.pinbar_wick_body_ratio, 1.0),
                c.lower_wick / c.body,
                "Long lower wick rejection — bullish pinbar",
            )
        if c.upper_wick / c.body >= self.pinbar_wick_body_ratio and c.upper_wick > c.lower_wick:
            return CandlePattern(
                CandleType.BEARISH_PINBAR, "bearish",
                min(c.upper_wick / c.body / self.pinbar_wick_body_ratio, 1.0),
                c.upper_wick / c.body,
                "Long upper wick rejection — bearish pinbar",
            )
        return None

    def _detect_doji(self, c: Candle) -> Optional[CandlePattern]:
        if c.range > 0 and c.body_pct <= self.doji_body_ratio:
            return CandlePattern(
                CandleType.DOJI, "neutral",
                1.0 - c.body_pct / self.doji_body_ratio,
                0.0,
                "Indecision — doji",
            )
        return None

    def _detect_engulfing(self, current: Candle, previous: Candle) -> Optional[CandlePattern]:
        if current.body_pct < self.engulfing_min_body:
            return None

        if (current.is_bullish and not previous.is_bullish
                and current.open <= previous.close and current.close >= previous.open):
            return CandlePattern(
                CandleType.BULLISH_ENGULFING, "bullish",
                current.body_pct,
                current.body / previous.body if previous.body > 0 else 1.0,
                "Bullish engulfing — strong reversal signal",
            )
        if (not current.is_bullish and previous.is_bullish
                and current.open >= previous.close and current.close <= previous.open):
            return CandlePattern(
                CandleType.BEARISH_ENGULFING, "bearish",
                current.body_pct,
                current.body / previous.body if previous.body > 0 else 1.0,
                "Bearish engulfing — strong reversal signal",
            )
        return None

    def _detect_rejection(self, c: Candle) -> Optional[CandlePattern]:
        if c.range <= 0:
            return None

        if c.lower_wick_pct >= self.rejection_wick_pct:
            return CandlePattern(
                CandleType.BULLISH_REJECTION, "bullish",
                c.lower_wick_pct,
                c.lower_wick / max(c.body, 1e-10),
                "Long lower wick — bullish rejection",
            )
        if c.upper_wick_pct >= self.rejection_wick_pct:
            return CandlePattern(
                CandleType.BEARISH_REJECTION, "bearish",
                c.upper_wick_pct,
                c.upper_wick / max(c.body, 1e-10),
                "Long upper wick — bearish rejection",
            )
        return None

    def _detect_impulse(self, c: Candle) -> Optional[CandlePattern]:
        if c.body_pct >= self.impulse_body_pct:
            direction = "bullish" if c.is_bullish else "bearish"
            ctype = CandleType.BULLISH_IMPULSE if c.is_bullish else CandleType.BEARISH_IMPULSE
            return CandlePattern(
                ctype, direction,
                c.body_pct,
                c.body / max(c.range, 1e-10),
                f"Strong {direction} impulse candle",
            )
        return None

    def _detect_inside_bar(self, current: Candle, previous: Candle) -> Optional[CandlePattern]:
        if current.high <= previous.high and current.low >= previous.low:
            return CandlePattern(
                CandleType.INSIDE_BAR, "neutral",
                1.0 - current.range / max(previous.range, 1e-10),
                0.0,
                "Inside bar — consolidation / breakout setup",
            )
        return None

    def analyze(self, candles: List[Candle]) -> List[CandlePattern]:
        """Detect all patterns on the latest candle(s)."""
        if not candles:
            return []

        current = candles[-1]
        patterns = []

        for detector in [self._detect_pinbar, self._detect_doji, self._detect_rejection, self._detect_impulse]:
            result = detector(current)
            if result:
                patterns.append(result)

        if len(candles) >= 2:
            previous = candles[-2]
            for detector in [self._detect_engulfing, self._detect_inside_bar]:
                result = detector(current, previous)
                if result:
                    patterns.append(result)

        return patterns

    def analyze_at_level(
        self, candles: List[Candle], level_price: float, pip_size: float, pip_tolerance: int = 10
    ) -> List[CandlePattern]:
        """Only return patterns if price is within tolerance of a level."""
        if not candles:
            return []

        current = candles[-1]
        tolerance = pip_tolerance * pip_size
        near_level = (
            abs(current.close - level_price) <= tolerance
            or abs(current.low - level_price) <= tolerance
            or abs(current.high - level_price) <= tolerance
        )
        if not near_level:
            return []

        return self.analyze(candles)

    def candle_features(self, candles: List[Candle]) -> Dict[str, float]:
        if not candles:
            return {"body_pct": 0, "upper_wick_pct": 0, "lower_wick_pct": 0, "is_bullish": 0, "pattern_count": 0}

        c = candles[-1]
        patterns = self.analyze(candles)
        return {
            "body_pct": c.body_pct,
            "upper_wick_pct": c.upper_wick_pct,
            "lower_wick_pct": c.lower_wick_pct,
            "is_bullish": 1.0 if c.is_bullish else 0.0,
            "pattern_count": float(len(patterns)),
        }
