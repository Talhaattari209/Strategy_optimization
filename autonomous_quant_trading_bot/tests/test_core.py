"""Tests for the core trading modules — client edge."""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.session_timer import SessionTimer, Session
from core.level_detector import LevelDetector, LevelType
from core.candle_analyzer import CandleAnalyzer, Candle, CandleType
from core.market_structure import MarketStructure


def make_ohlcv(n: int = 100, start_price: float = 1.1000) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-02 08:00", periods=n, freq="h")
    returns = np.random.randn(n) * 0.001
    closes = start_price + np.cumsum(returns)
    highs = closes + abs(np.random.randn(n) * 0.0005)
    lows = closes - abs(np.random.randn(n) * 0.0005)
    opens = closes - returns / 2
    volumes = np.random.randint(100, 10000, n)

    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }, index=dates)


class TestSessionTimer:
    def test_london_session(self):
        timer = SessionTimer()
        dt = datetime(2024, 1, 15, 10, 0)
        state = timer.get_state(dt)
        assert Session.LONDON in state.active_sessions

    def test_asia_session(self):
        timer = SessionTimer()
        dt = datetime(2024, 1, 15, 3, 0)
        state = timer.get_state(dt)
        assert Session.ASIA in state.active_sessions

    def test_ny_session(self):
        timer = SessionTimer()
        dt = datetime(2024, 1, 15, 15, 0)
        state = timer.get_state(dt)
        assert Session.NEW_YORK in state.active_sessions

    def test_london_ny_overlap(self):
        timer = SessionTimer()
        dt = datetime(2024, 1, 15, 14, 0)
        state = timer.get_state(dt)
        assert Session.LONDON in state.active_sessions
        assert Session.NEW_YORK in state.active_sessions
        assert state.is_overlap

    def test_session_weights(self):
        timer = SessionTimer()
        dt = datetime(2024, 1, 15, 10, 0)
        weights = timer.session_weight(dt)
        assert "time_sin" in weights
        assert "london_active" in weights
        assert weights["london_active"] == 1.0


class TestLevelDetector:
    def test_hod_lod(self):
        detector = LevelDetector()
        ts = pd.Timestamp("2024-01-15 10:00")
        detector.update(1.1050, 1.0950, 1.1020, ts, "London")
        detector.update(1.1080, 1.0970, 1.1060, ts + pd.Timedelta(hours=1), "London")

        levels = detector.get_levels(ts + pd.Timedelta(hours=1))
        hod = [l for l in levels if l.level_type == LevelType.HOD]
        lod = [l for l in levels if l.level_type == LevelType.LOD]
        assert len(hod) == 1
        assert hod[0].price == 1.1080
        assert len(lod) == 1
        assert lod[0].price == 1.0950

    def test_at_level(self):
        detector = LevelDetector({"levels": {"pip_tolerance": 10}, "broker": {"pip_size": 0.0001}})
        ts = pd.Timestamp("2024-01-15 10:00")
        detector.update(1.1050, 1.0950, 1.1000, ts, "London")

        state = detector.get_state(1.1048, ts)
        assert state.at_level is not None

    def test_level_features(self):
        detector = LevelDetector()
        ts = pd.Timestamp("2024-01-15 10:00")
        detector.update(1.1050, 1.0950, 1.1000, ts, "London")
        feats = detector.level_features(1.1000, ts)
        assert "dist_to_hod" in feats
        assert "dist_to_lod" in feats
        assert "at_level" in feats


class TestCandleAnalyzer:
    def test_bullish_pinbar(self):
        analyzer = CandleAnalyzer()
        candle = Candle(open=1.1010, high=1.1015, low=1.0950, close=1.1012)
        patterns = analyzer.analyze([candle])
        pinbars = [p for p in patterns if "pinbar" in p.candle_type.value.lower()]
        assert len(pinbars) > 0
        assert pinbars[0].direction == "bullish"

    def test_bearish_engulfing(self):
        analyzer = CandleAnalyzer()
        prev = Candle(open=1.0990, high=1.1010, low=1.0985, close=1.1005)
        curr = Candle(open=1.1010, high=1.1015, low=1.0970, close=1.0975)
        patterns = analyzer.analyze([prev, curr])
        engulfing = [p for p in patterns if "engulfing" in p.candle_type.value.lower()]
        assert len(engulfing) > 0
        assert engulfing[0].direction == "bearish"

    def test_doji(self):
        analyzer = CandleAnalyzer()
        candle = Candle(open=1.1000, high=1.1020, low=1.0980, close=1.1001)
        patterns = analyzer.analyze([candle])
        dojis = [p for p in patterns if p.candle_type == CandleType.DOJI]
        assert len(dojis) > 0

    def test_impulse(self):
        analyzer = CandleAnalyzer()
        candle = Candle(open=1.0980, high=1.1050, low=1.0975, close=1.1045)
        patterns = analyzer.analyze([candle])
        impulses = [p for p in patterns if "impulse" in p.candle_type.value.lower()]
        assert len(impulses) > 0
        assert impulses[0].direction == "bullish"

    def test_inside_bar(self):
        analyzer = CandleAnalyzer()
        prev = Candle(open=1.0990, high=1.1050, low=1.0950, close=1.1020)
        curr = Candle(open=1.1000, high=1.1030, low=1.0960, close=1.1010)
        patterns = analyzer.analyze([prev, curr])
        inside = [p for p in patterns if p.candle_type == CandleType.INSIDE_BAR]
        assert len(inside) > 0

    def test_analyze_at_level(self):
        analyzer = CandleAnalyzer()
        candle = Candle(open=1.1010, high=1.1015, low=1.0950, close=1.1012)
        patterns = analyzer.analyze_at_level([candle], 1.1015, 0.0001, 10)
        assert len(patterns) > 0


class TestMarketStructure:
    def test_swing_detection(self):
        ms = MarketStructure({"levels": {"fractal_period": 3, "swing_lookback": 20}})
        n = 50
        np.random.seed(42)
        highs = 100 + np.cumsum(np.random.randn(n) * 0.1)
        lows = highs - abs(np.random.randn(n) * 0.2)
        timestamps = [pd.Timestamp("2024-01-15") + pd.Timedelta(hours=i) for i in range(n)]

        sh, sl = ms.detect_swings(highs, lows, timestamps)
        assert isinstance(sh, list)
        assert isinstance(sl, list)

    def test_structure_update(self):
        ms = MarketStructure({"levels": {"fractal_period": 3, "swing_lookback": 20}})
        n = 50
        np.random.seed(42)
        highs = 100 + np.cumsum(np.random.randn(n) * 0.1)
        lows = highs - abs(np.random.randn(n) * 0.2)
        closes = (highs + lows) / 2
        timestamps = [pd.Timestamp("2024-01-15") + pd.Timedelta(hours=i) for i in range(n)]

        state = ms.update(highs, lows, closes, timestamps)
        assert state.trend in ("bullish", "bearish", "ranging")
        assert isinstance(state.structure_levels, list)

    def test_structure_features(self):
        ms = MarketStructure()
        feats = ms.structure_features()
        assert "trend_bullish" in feats
        assert "bos_count" in feats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
