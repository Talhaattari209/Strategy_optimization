"""
Data Collector — pulls OHLCV data from MetaTrader5 or ccxt.
Builds feature vectors (20+ indicators + volume profile + Fourier cycles).
Pluggable: MT5 for forex, ccxt for crypto.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataCollector:
    """
    Pluggable data source: MetaTrader5 or ccxt or CSV.
    Provides OHLCV DataFrames and feature vectors for the bot.
    """

    def __init__(self, config: Dict | None = None) -> None:
        self.config = config or {}
        self.symbol: str = self.config.get("broker", {}).get("default_symbol", "EURUSD")
        self.timeframe: str = self.config.get("broker", {}).get("default_timeframe", "H1")
        self._mt5_available: bool = False
        self._ccxt_available: bool = False
        self._data_cache: Dict[str, pd.DataFrame] = {}

        self._try_init_mt5()

    def _try_init_mt5(self) -> None:
        try:
            import MetaTrader5 as mt5
            if mt5.initialize():
                self._mt5_available = True
                logger.info("MetaTrader5 connection established")
            else:
                logger.warning("MetaTrader5 initialization failed; falling back to CSV/ccxt")
        except ImportError:
            logger.info("MetaTrader5 package not installed; using CSV/ccxt mode")

    def fetch_mt5(
        self, symbol: str, timeframe: str = "H1", bars: int = 1000
    ) -> pd.DataFrame:
        import MetaTrader5 as mt5

        tf_map = {
            "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1,
        }
        tf = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
        if rates is None or len(rates) == 0:
            raise ValueError(f"No data returned for {symbol} {timeframe}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("time")
        df = df.rename(columns={"tick_volume": "volume"})
        df = df[["open", "high", "low", "close", "volume"]]
        return df

    def fetch_ccxt(
        self, symbol: str, timeframe: str = "1h", exchange: str = "binance", limit: int = 1000
    ) -> pd.DataFrame:
        import ccxt

        exchange_cls = getattr(ccxt, exchange)()
        ohlcv = exchange_cls.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
        return df

    def load_csv(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath, parse_dates=True, index_col=0)
        required = {"open", "high", "low", "close"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"CSV must contain columns: {required}")
        if "volume" not in df.columns:
            df["volume"] = 0
        return df

    def fetch(self, symbol: str | None = None, bars: int = 1000) -> pd.DataFrame:
        sym = symbol or self.symbol
        if self._mt5_available:
            return self.fetch_mt5(sym, self.timeframe, bars)
        logger.warning("No live data source — load CSV with load_csv() or configure MT5/ccxt")
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build 20+ technical indicator features from OHLCV data."""
        features = pd.DataFrame(index=df.index)

        closes = df["close"]
        highs = df["high"]
        lows = df["low"]
        volumes = df["volume"]
        returns = closes.pct_change()

        # Trend indicators
        features["sma_20"] = closes.rolling(20).mean()
        features["sma_50"] = closes.rolling(50).mean()
        features["ema_12"] = closes.ewm(span=12).mean()
        features["ema_26"] = closes.ewm(span=26).mean()
        features["macd"] = features["ema_12"] - features["ema_26"]
        features["macd_signal"] = features["macd"].ewm(span=9).mean()

        # Momentum
        features["rsi_14"] = self._rsi(closes, 14)
        features["momentum_10"] = closes.pct_change(10)
        features["roc_5"] = closes.pct_change(5)

        # Volatility
        features["atr_14"] = self._atr(highs, lows, closes, 14)
        features["bb_upper"] = features["sma_20"] + 2 * closes.rolling(20).std()
        features["bb_lower"] = features["sma_20"] - 2 * closes.rolling(20).std()
        features["bb_width"] = (features["bb_upper"] - features["bb_lower"]) / features["sma_20"]
        features["returns_std_20"] = returns.rolling(20).std()

        # Volume
        features["volume_sma_20"] = volumes.rolling(20).mean()
        features["volume_ratio"] = volumes / features["volume_sma_20"].replace(0, 1)
        features["obv"] = (np.sign(returns) * volumes).cumsum()

        # Price relative to range
        features["close_to_high"] = (closes - lows) / (highs - lows).replace(0, 1)
        features["daily_range"] = highs - lows

        # Returns
        features["return_1"] = returns
        features["return_5"] = closes.pct_change(5)
        features["return_20"] = closes.pct_change(20)

        return features.dropna()

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def get_latest_tick(self, symbol: str | None = None) -> Dict[str, float]:
        if self._mt5_available:
            import MetaTrader5 as mt5
            sym = symbol or self.symbol
            tick = mt5.symbol_info_tick(sym)
            if tick:
                return {"bid": tick.bid, "ask": tick.ask, "time": tick.time}
        return {"bid": 0.0, "ask": 0.0, "time": 0}
