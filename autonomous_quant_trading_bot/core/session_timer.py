"""
Session Timer — precise session & overlap detection.
Detects: Asia, London, New York, Sydney sessions and their overlaps.
Part of the client's core 4-factor edge: "What time (session) is it?"
"""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional


class Session(Enum):
    ASIA = "Asia"
    LONDON = "London"
    NEW_YORK = "NewYork"
    SYDNEY = "Sydney"


@dataclass
class SessionWindow:
    name: Session
    start: time
    end: time

    def is_active(self, t: time) -> bool:
        if self.start <= self.end:
            return self.start <= t < self.end
        return t >= self.start or t < self.end


@dataclass
class SessionState:
    active_sessions: List[Session]
    is_overlap: bool
    overlap_sessions: List[tuple]
    session_minutes_elapsed: Dict[Session, int]
    session_minutes_remaining: Dict[Session, int]
    primary_session: Optional[Session]


class SessionTimer:
    """
    Detects current trading session and overlaps using UTC times.
    Configurable via config dict for broker timezone offset.
    """

    def __init__(self, config: Dict | None = None) -> None:
        cfg = config or {}
        sessions_cfg = cfg.get("sessions", {})
        self.broker_offset_hours: int = cfg.get("broker", {}).get("timezone_offset_utc", 0)

        self.windows: List[SessionWindow] = [
            SessionWindow(Session.ASIA, *self._parse_times(sessions_cfg.get("asia", {"start": "00:00", "end": "08:00"}))),
            SessionWindow(Session.LONDON, *self._parse_times(sessions_cfg.get("london", {"start": "08:00", "end": "16:00"}))),
            SessionWindow(Session.NEW_YORK, *self._parse_times(sessions_cfg.get("new_york", {"start": "13:00", "end": "21:00"}))),
            SessionWindow(Session.SYDNEY, *self._parse_times(sessions_cfg.get("sydney", {"start": "21:00", "end": "05:00"}))),
        ]

        self._overlap_pairs = [
            (Session.LONDON, Session.NEW_YORK),
            (Session.SYDNEY, Session.ASIA),
            (Session.ASIA, Session.LONDON),
        ]

    @staticmethod
    def _parse_times(cfg: Dict) -> tuple:
        start = time.fromisoformat(cfg["start"])
        end = time.fromisoformat(cfg["end"])
        return start, end

    def _to_utc_time(self, dt: datetime) -> time:
        utc_dt = dt - timedelta(hours=self.broker_offset_hours)
        return utc_dt.time()

    def _minutes_elapsed(self, current: time, start: time) -> int:
        c = current.hour * 60 + current.minute
        s = start.hour * 60 + start.minute
        diff = c - s
        if diff < 0:
            diff += 24 * 60
        return diff

    def _session_duration(self, w: SessionWindow) -> int:
        s = w.start.hour * 60 + w.start.minute
        e = w.end.hour * 60 + w.end.minute
        dur = e - s
        if dur <= 0:
            dur += 24 * 60
        return dur

    def get_state(self, dt: datetime) -> SessionState:
        t = self._to_utc_time(dt)
        active = []
        elapsed = {}
        remaining = {}

        for w in self.windows:
            if w.is_active(t):
                active.append(w.name)
                e = self._minutes_elapsed(t, w.start)
                d = self._session_duration(w)
                elapsed[w.name] = e
                remaining[w.name] = max(d - e, 0)

        overlaps = []
        for s1, s2 in self._overlap_pairs:
            if s1 in active and s2 in active:
                overlaps.append((s1, s2))

        primary = None
        if active:
            primary = active[0]
            if len(active) > 1:
                primary = min(active, key=lambda s: elapsed.get(s, 0))

        return SessionState(
            active_sessions=active,
            is_overlap=len(overlaps) > 0,
            overlap_sessions=overlaps,
            session_minutes_elapsed=elapsed,
            session_minutes_remaining=remaining,
            primary_session=primary,
        )

    def session_weight(self, dt: datetime) -> Dict[str, float]:
        """Encode session as features: sine/cosine of time + binary flags."""
        t = self._to_utc_time(dt)
        minutes = t.hour * 60 + t.minute
        frac = minutes / (24 * 60)

        import math
        state = self.get_state(dt)

        return {
            "time_sin": math.sin(2 * math.pi * frac),
            "time_cos": math.cos(2 * math.pi * frac),
            "asia_active": 1.0 if Session.ASIA in state.active_sessions else 0.0,
            "london_active": 1.0 if Session.LONDON in state.active_sessions else 0.0,
            "ny_active": 1.0 if Session.NEW_YORK in state.active_sessions else 0.0,
            "sydney_active": 1.0 if Session.SYDNEY in state.active_sessions else 0.0,
            "is_overlap": 1.0 if state.is_overlap else 0.0,
        }

    def is_session_open(self, dt: datetime) -> bool:
        state = self.get_state(dt)
        return len(state.active_sessions) > 0

    def get_session_start_time(self, session: Session, dt: datetime) -> datetime:
        for w in self.windows:
            if w.name == session:
                return dt.replace(hour=w.start.hour, minute=w.start.minute, second=0, microsecond=0)
        return dt
