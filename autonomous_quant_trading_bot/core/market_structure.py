"""
Market Structure — CHOCH, BOS, swing detection.
Fractal/swing-based structure analysis with confirmation.
Feeds structure levels into LevelDetector.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum


class StructureType(Enum):
    SWING_HIGH = "swing_high"
    SWING_LOW = "swing_low"
    BOS_BULLISH = "bos_bullish"
    BOS_BEARISH = "bos_bearish"
    CHOCH_BULLISH = "choch_bullish"
    CHOCH_BEARISH = "choch_bearish"


@dataclass
class SwingPoint:
    price: float
    index: int
    timestamp: pd.Timestamp
    swing_type: str  # "high" or "low"
    confirmed: bool = False


@dataclass
class StructureBreak:
    break_type: StructureType
    price: float
    index: int
    timestamp: pd.Timestamp
    broken_level: float
    confirmation_candles: int


@dataclass
class MarketStructureState:
    trend: str  # "bullish", "bearish", "ranging"
    swing_highs: List[SwingPoint]
    swing_lows: List[SwingPoint]
    last_bos: Optional[StructureBreak]
    last_choch: Optional[StructureBreak]
    structure_levels: List[float]


class MarketStructure:
    """
    Detects market structure: swing points, BOS, CHOCH.
    BOS = Break of Structure (trend continuation)
    CHOCH = Change of Character (trend reversal)
    Uses fractal detection with configurable lookback.
    """

    def __init__(self, config: Dict | None = None) -> None:
        cfg = config or {}
        levels_cfg = cfg.get("levels", {})
        self.fractal_period: int = levels_cfg.get("fractal_period", 5)
        self.swing_lookback: int = levels_cfg.get("swing_lookback", 20)

        self._swing_highs: List[SwingPoint] = []
        self._swing_lows: List[SwingPoint] = []
        self._breaks: List[StructureBreak] = []
        self._current_trend: str = "ranging"

    def _detect_fractal_high(self, highs: NDArray[np.float64], idx: int) -> bool:
        if idx < self.fractal_period or idx >= len(highs) - self.fractal_period:
            return False
        center = highs[idx]
        left = highs[idx - self.fractal_period:idx]
        right = highs[idx + 1:idx + self.fractal_period + 1]
        return bool(np.all(center > left) and np.all(center > right))

    def _detect_fractal_low(self, lows: NDArray[np.float64], idx: int) -> bool:
        if idx < self.fractal_period or idx >= len(lows) - self.fractal_period:
            return False
        center = lows[idx]
        left = lows[idx - self.fractal_period:idx]
        right = lows[idx + 1:idx + self.fractal_period + 1]
        return bool(np.all(center < left) and np.all(center < right))

    def detect_swings(
        self, highs: NDArray[np.float64], lows: NDArray[np.float64], timestamps: List[pd.Timestamp]
    ) -> tuple:
        new_swing_highs = []
        new_swing_lows = []

        for i in range(self.fractal_period, len(highs) - self.fractal_period):
            if self._detect_fractal_high(highs, i):
                sp = SwingPoint(float(highs[i]), i, timestamps[i], "high", confirmed=True)
                new_swing_highs.append(sp)
            if self._detect_fractal_low(lows, i):
                sp = SwingPoint(float(lows[i]), i, timestamps[i], "low", confirmed=True)
                new_swing_lows.append(sp)

        self._swing_highs = new_swing_highs[-self.swing_lookback:]
        self._swing_lows = new_swing_lows[-self.swing_lookback:]

        return self._swing_highs, self._swing_lows

    def detect_bos(self, close: float, index: int, timestamp: pd.Timestamp) -> Optional[StructureBreak]:
        """Break of Structure — continuation signal."""
        if len(self._swing_highs) >= 2 and self._current_trend == "bullish":
            last_sh = self._swing_highs[-1]
            if close > last_sh.price:
                brk = StructureBreak(
                    StructureType.BOS_BULLISH, close, index, timestamp,
                    last_sh.price, index - last_sh.index,
                )
                self._breaks.append(brk)
                return brk

        if len(self._swing_lows) >= 2 and self._current_trend == "bearish":
            last_sl = self._swing_lows[-1]
            if close < last_sl.price:
                brk = StructureBreak(
                    StructureType.BOS_BEARISH, close, index, timestamp,
                    last_sl.price, index - last_sl.index,
                )
                self._breaks.append(brk)
                return brk

        return None

    def detect_choch(self, close: float, index: int, timestamp: pd.Timestamp) -> Optional[StructureBreak]:
        """Change of Character — reversal signal."""
        if len(self._swing_lows) >= 1 and self._current_trend == "bullish":
            last_sl = self._swing_lows[-1]
            if close < last_sl.price:
                brk = StructureBreak(
                    StructureType.CHOCH_BEARISH, close, index, timestamp,
                    last_sl.price, index - last_sl.index,
                )
                self._breaks.append(brk)
                self._current_trend = "bearish"
                return brk

        if len(self._swing_highs) >= 1 and self._current_trend == "bearish":
            last_sh = self._swing_highs[-1]
            if close > last_sh.price:
                brk = StructureBreak(
                    StructureType.CHOCH_BULLISH, close, index, timestamp,
                    last_sh.price, index - last_sh.index,
                )
                self._breaks.append(brk)
                self._current_trend = "bullish"
                return brk

        return None

    def update(
        self,
        highs: NDArray[np.float64],
        lows: NDArray[np.float64],
        closes: NDArray[np.float64],
        timestamps: List[pd.Timestamp],
    ) -> MarketStructureState:
        self.detect_swings(highs, lows, timestamps)

        if len(closes) > 0:
            idx = len(closes) - 1
            ts = timestamps[idx]
            c = float(closes[idx])
            self.detect_bos(c, idx, ts)
            self.detect_choch(c, idx, ts)

            if len(self._swing_highs) >= 2 and len(self._swing_lows) >= 2:
                if (self._swing_highs[-1].price > self._swing_highs[-2].price
                        and self._swing_lows[-1].price > self._swing_lows[-2].price):
                    self._current_trend = "bullish"
                elif (self._swing_highs[-1].price < self._swing_highs[-2].price
                      and self._swing_lows[-1].price < self._swing_lows[-2].price):
                    self._current_trend = "bearish"

        structure_levels = (
            [sh.price for sh in self._swing_highs[-5:]]
            + [sl.price for sl in self._swing_lows[-5:]]
        )

        last_bos = next((b for b in reversed(self._breaks) if "BOS" in b.break_type.name), None)
        last_choch = next((b for b in reversed(self._breaks) if "CHOCH" in b.break_type.name), None)

        return MarketStructureState(
            trend=self._current_trend,
            swing_highs=list(self._swing_highs),
            swing_lows=list(self._swing_lows),
            last_bos=last_bos,
            last_choch=last_choch,
            structure_levels=structure_levels,
        )

    def structure_features(self) -> Dict[str, float]:
        return {
            "trend_bullish": 1.0 if self._current_trend == "bullish" else 0.0,
            "trend_bearish": 1.0 if self._current_trend == "bearish" else 0.0,
            "swing_high_count": float(len(self._swing_highs)),
            "swing_low_count": float(len(self._swing_lows)),
            "bos_count": float(sum(1 for b in self._breaks if "BOS" in b.break_type.name)),
            "choch_count": float(sum(1 for b in self._breaks if "CHOCH" in b.break_type.name)),
        }
