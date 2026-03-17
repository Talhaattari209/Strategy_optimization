"""
Level Detector — HOD/LOD, session/week highs-lows, close-of-day.
Part of the client's core 4-factor edge: "At what level?"
Rolling calculations reset at day/session/week boundaries.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class LevelType(Enum):
    HOD = "HOD"
    LOD = "LOD"
    HO_SESSION = "HO_Session"
    LO_SESSION = "LO_Session"
    HO_WEEK = "HO_Week"
    LO_WEEK = "LO_Week"
    CLOSE_OF_DAY = "Close_of_Day"
    CHOCH = "CHOCH"
    BOS = "BOS"


@dataclass
class PriceLevel:
    level_type: LevelType
    price: float
    timestamp: pd.Timestamp
    strength: float = 1.0  # how many times tested

    def distance_pips(self, current_price: float, pip_size: float) -> float:
        return abs(current_price - self.price) / pip_size


@dataclass
class LevelState:
    levels: List[PriceLevel]
    nearest_above: Optional[PriceLevel]
    nearest_below: Optional[PriceLevel]
    at_level: Optional[PriceLevel]
    distance_to_nearest_pips: float


class LevelDetector:
    """
    Detects and tracks all key price levels:
    - HOD/LOD: reset at day start (NY close or user-defined)
    - Session HO/LOS: reset at session start
    - Weekly HO/LO: reset Sunday/Monday
    - Close-of-Day: previous day close
    """

    def __init__(self, config: Dict | None = None) -> None:
        cfg = config or {}
        levels_cfg = cfg.get("levels", {})
        self.pip_tolerance: int = levels_cfg.get("pip_tolerance", 10)
        self.pip_size: float = cfg.get("broker", {}).get("pip_size", 0.0001)
        self.day_reset_hour: int = levels_cfg.get("day_reset_hour", 21)
        self.week_reset_day: int = levels_cfg.get("week_reset_day", 0)

        self._levels: List[PriceLevel] = []
        self._day_high: float = -np.inf
        self._day_low: float = np.inf
        self._session_high: float = -np.inf
        self._session_low: float = np.inf
        self._week_high: float = -np.inf
        self._week_low: float = np.inf
        self._prev_close: float = 0.0
        self._last_day: int = -1
        self._last_week: int = -1
        self._last_session_start: str = ""

    def _check_day_reset(self, ts: pd.Timestamp) -> bool:
        day_marker = ts.day if ts.hour >= self.day_reset_hour else ts.day - 1
        if day_marker != self._last_day:
            self._last_day = day_marker
            return True
        return False

    def _check_week_reset(self, ts: pd.Timestamp) -> bool:
        week = ts.isocalendar()[1]
        if week != self._last_week:
            self._last_week = week
            return True
        return False

    def update(
        self,
        high: float,
        low: float,
        close: float,
        timestamp: pd.Timestamp,
        session_name: str = "",
    ) -> None:
        if self._check_day_reset(timestamp):
            if self._day_high > -np.inf:
                self._prev_close = close
            self._day_high = -np.inf
            self._day_low = np.inf

        if self._check_week_reset(timestamp):
            self._week_high = -np.inf
            self._week_low = np.inf

        if session_name and session_name != self._last_session_start:
            self._session_high = -np.inf
            self._session_low = np.inf
            self._last_session_start = session_name

        self._day_high = max(self._day_high, high)
        self._day_low = min(self._day_low, low)
        self._session_high = max(self._session_high, high)
        self._session_low = min(self._session_low, low)
        self._week_high = max(self._week_high, high)
        self._week_low = min(self._week_low, low)

    def get_levels(self, timestamp: pd.Timestamp) -> List[PriceLevel]:
        levels = []

        if self._day_high > -np.inf:
            levels.append(PriceLevel(LevelType.HOD, self._day_high, timestamp))
        if self._day_low < np.inf:
            levels.append(PriceLevel(LevelType.LOD, self._day_low, timestamp))
        if self._session_high > -np.inf:
            levels.append(PriceLevel(LevelType.HO_SESSION, self._session_high, timestamp))
        if self._session_low < np.inf:
            levels.append(PriceLevel(LevelType.LO_SESSION, self._session_low, timestamp))
        if self._week_high > -np.inf:
            levels.append(PriceLevel(LevelType.HO_WEEK, self._week_high, timestamp))
        if self._week_low < np.inf:
            levels.append(PriceLevel(LevelType.LO_WEEK, self._week_low, timestamp))
        if self._prev_close > 0:
            levels.append(PriceLevel(LevelType.CLOSE_OF_DAY, self._prev_close, timestamp))

        return levels

    def get_state(self, current_price: float, timestamp: pd.Timestamp) -> LevelState:
        levels = self.get_levels(timestamp)
        if not levels:
            return LevelState([], None, None, None, float("inf"))

        above = [l for l in levels if l.price > current_price]
        below = [l for l in levels if l.price < current_price]

        nearest_above = min(above, key=lambda l: l.price - current_price) if above else None
        nearest_below = max(below, key=lambda l: l.price) if below else None

        at_level = None
        tolerance = self.pip_tolerance * self.pip_size
        for l in levels:
            if abs(current_price - l.price) <= tolerance:
                at_level = l
                break

        dist = float("inf")
        if nearest_above:
            dist = min(dist, (nearest_above.price - current_price) / self.pip_size)
        if nearest_below:
            dist = min(dist, (current_price - nearest_below.price) / self.pip_size)

        return LevelState(
            levels=levels,
            nearest_above=nearest_above,
            nearest_below=nearest_below,
            at_level=at_level,
            distance_to_nearest_pips=dist,
        )

    def add_structure_level(
        self, level_type: LevelType, price: float, timestamp: pd.Timestamp, strength: float = 1.0
    ) -> None:
        self._levels.append(PriceLevel(level_type, price, timestamp, strength))

    def get_all_levels(self, current_price: float, timestamp: pd.Timestamp) -> List[PriceLevel]:
        dynamic = self.get_levels(timestamp)
        structure = [l for l in self._levels]
        return dynamic + structure

    def level_features(self, current_price: float, timestamp: pd.Timestamp) -> Dict[str, float]:
        state = self.get_state(current_price, timestamp)
        return {
            "dist_to_hod": abs(current_price - self._day_high) / self.pip_size if self._day_high > -np.inf else 999,
            "dist_to_lod": abs(current_price - self._day_low) / self.pip_size if self._day_low < np.inf else 999,
            "dist_to_session_high": abs(current_price - self._session_high) / self.pip_size if self._session_high > -np.inf else 999,
            "dist_to_session_low": abs(current_price - self._session_low) / self.pip_size if self._session_low < np.inf else 999,
            "dist_to_week_high": abs(current_price - self._week_high) / self.pip_size if self._week_high > -np.inf else 999,
            "dist_to_week_low": abs(current_price - self._week_low) / self.pip_size if self._week_low < np.inf else 999,
            "dist_to_prev_close": abs(current_price - self._prev_close) / self.pip_size if self._prev_close > 0 else 999,
            "at_level": 1.0 if state.at_level is not None else 0.0,
            "nearest_dist_pips": state.distance_to_nearest_pips,
        }
