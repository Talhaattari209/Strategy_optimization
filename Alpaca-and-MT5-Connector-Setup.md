# Alpaca API & MT5 Connector Setup Guide

This document describes how this project connects to **Alpaca** and **MetaTrader 5 (MT5)** and provides all code and instructions to replicate the same setup in another project.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Dependencies](#2-dependencies)
3. [Environment Variables](#3-environment-variables)
4. [Project Structure (Connectors)](#4-project-structure-connectors)
5. [Base Connector Interface](#5-base-connector-interface)
6. [Alpaca API Setup](#6-alpaca-api-setup)
7. [MT5 Setup](#7-mt5-setup)
8. [Connector Factory](#8-connector-factory)
9. [Supporting Modules](#9-supporting-modules)
10. [Optional: Alpaca MCP Server](#10-optional-alpaca-mcp-server)
11. [Quick Start for a New Project](#11-quick-start-for-a-new-project)

---

## 1. Overview

- **Alpaca**: REST + WebSocket for US equities (paper/live). Uses `alpaca-py` (official SDK). Works on any OS.
- **MT5**: Windows-only native DLL; data via `copy_rates_from_pos`, execution via `order_send`. Zero-MQL: no Expert Advisors; terminal is a headless execution gateway. On WSL/Linux, the connector fails gracefully with instructions to use a Windows-hosted REST/Socket bridge.
- **Unified interface**: Both implement `BaseConnector` so callers can swap brokers via `BROKER_TYPE` (env) or `get_connector(broker_type)`.

---

## 2. Dependencies

**Using uv / pyproject.toml (this project):**

```toml
[project]
requires-python = ">=3.10"
dependencies = [
    "pandas",
    "python-dotenv",
    # ... your other deps
]

[project.optional-dependencies]
connectors = ["alpaca-py", "MetaTrader5"]
```

Install with connectors:

```bash
pip install -e ".[connectors]"
# or
uv pip install -e ".[connectors]"
```

**Using requirements.txt:**

```text
pandas>=2.0
python-dotenv>=1.0
alpaca-py>=0.43
MetaTrader5>=5.0.45   # Windows only; omit on Linux/WSL or use optional install
```

**Notes:**

- **Alpaca**: `alpaca-py` works on Windows, macOS, Linux.
- **MT5**: `MetaTrader5` is Windows-only (official wheel). On WSL/Linux, do not install it (or make it optional); the connector will raise a clear error and suggest a Windows bridge.

---

## 3. Environment Variables

Create a `.env` file (and use `.env.example` for documentation). **Never commit real keys.**

### Alpaca

| Variable | Required | Description |
|----------|----------|-------------|
| `ALPACA_API_KEY` | Yes (for Alpaca) | Alpaca API key |
| `ALPACA_SECRET_KEY` | Yes (for Alpaca) | Alpaca secret key |
| `ALPACA_PAPER` | No | `true` (default) = paper trading; `false` = live |
| `ALPACA_API_KEY_2` | No | Backup account key (multi-account failover) |
| `ALPACA_SECRET_KEY_2` | No | Backup account secret |
| `ALPACA_TICKER_SYMBOL` | No | Default symbol for “live ticker” (e.g. `SPY`) |
| `ALPACA_MAX_ORDERS_PER_MIN` | No | Rate limit for token bucket (default `200`) |

### MT5

- No env vars required for basic use. MT5 terminal must be **running and logged in** on Windows.
- Optional: pass the path to the MT5 terminal executable to `MT5Connector(path="C:/path/to/MetaTrader5/terminal64.exe")` if you need to specify it.

### Broker selection

| Variable | Values | Description |
|----------|--------|-------------|
| `BROKER_TYPE` | `alpaca` \| `mt5` | Which connector to use. Default: `alpaca`. |

### Optional (cache, logging, risk)

| Variable | Description |
|----------|-------------|
| `CACHE_TTL_SECONDS` | TTL for OHLCV cache (default `60`) |
| `CACHE_DIR` | Directory for disk-backed cache (empty = in-memory only) |
| `CONNECTOR_MAX_RISK_FRACTION` | Max fraction of equity per order (risk check) |
| `ORDER_BATCH_WINDOW_MS` | Batch window for execution manager (ms) |

**Example `.env.example`:**

```env
# Broker: alpaca | mt5 (MT5 is Windows-only; on WSL use a Windows bridge)
BROKER_TYPE=alpaca

# Alpaca – Primary
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret

# Alpaca – Paper (default true)
# ALPACA_PAPER=true

# Alpaca – Backup (optional, for failover)
# ALPACA_API_KEY_2=your_second_key
# ALPACA_SECRET_KEY_2=your_second_secret
```

---

## 4. Project Structure (Connectors)

Minimal layout to replicate in another project:

```text
your_project/
├── .env
├── .env.example
├── pyproject.toml   # or requirements.txt
├── src/
│   └── connectors/
│       ├── __init__.py
│       ├── base_connector.py
│       ├── exceptions.py
│       ├── log_utils.py
│       ├── connector_factory.py
│       ├── alpaca_connector.py
│       ├── mt5_connector.py
│       ├── cache_manager.py      # optional, used by Alpaca
│       └── execution_manager.py  # optional, for throttling/failover
└── logs/                          # created automatically for connector logs
```

---

## 5. Base Connector Interface

All brokers implement this abstract base so the rest of the app can use one interface.

**File: `src/connectors/base_connector.py`**

```python
"""
Unified connector interface (Pluggable Senses).
All brokers (MT5, Alpaca) implement this base.
"""
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseConnector(ABC):
    @abstractmethod
    def connect(self) -> bool:
        """Initialize session and authenticate. Returns True on success."""
        pass

    @abstractmethod
    def get_ohlcv(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        """
        Fetch historical data. Returns DataFrame with columns:
        Open, High, Low, Close, Volume; index = Timestamp (datetime).
        """
        pass

    @abstractmethod
    def execute_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        **kwargs: Any,
    ) -> dict:
        """
        Execute order. Returns dict with at least:
        status ('success' | 'rejected' | 'error'), order_id (if success), message (if error).
        """
        pass

    @abstractmethod
    def get_account_state(self) -> dict:
        """Return balance, equity, and optional margin/drawdown info."""
        pass
```

---

## 6. Alpaca API Setup

### 6.1 Get credentials

1. Sign up at [Alpaca](https://alpaca.markets/).
2. In the dashboard, create **Paper** and/or **Live** keys.
3. Set in `.env`: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`. Use paper keys for testing.

### 6.2 Connector code

**File: `src/connectors/exceptions.py`** (required by Alpaca connector)

```python
class ConnectorError(Exception):
    pass

class BrokerConnectionError(ConnectorError):
    """Raised when connection to the broker fails."""
    pass

class InsufficientLiquidityError(ConnectorError):
    """Raised when order rejected due to risk/position limits."""
    pass

class AuthenticationError(ConnectorError):
    """Raised when API credentials are invalid or expired."""
    pass
```

**File: `src/connectors/log_utils.py`**

```python
import logging
from datetime import datetime
from pathlib import Path

def get_connector_logger(broker: str, logs_dir: str | Path | None = None) -> logging.Logger:
    logs_path = Path(logs_dir) if logs_dir else Path("logs")
    logs_path.mkdir(parents=True, exist_ok=True)
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    log_file = logs_path / f"connector_{broker}_{date_str}.log"
    name = f"connector.{broker}"
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return logger
```

**File: `src/connectors/alpaca_connector.py`** (core logic; optional: cache, streaming, backup account)

Key parts:

- **Connection**: `TradingClient` (paper/live) + `StockHistoricalDataClient` (bars).
- **Credentials**: from constructor or `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`. Optional backup: `ALPACA_API_KEY_2` / `ALPACA_SECRET_KEY_2`.
- **Data**: `get_ohlcv(symbol, timeframe, count)` uses `StockBarsRequest` and optional cache. Timeframes: `1m`, `5m`, `15m`, `1h`, `1d` etc. Use **IEX** feed for free-tier (SIP needs subscription).
- **Execution**: `execute_order(symbol, side, qty, order_type="market", limit_price=None)`. Uses `MarketOrderRequest` / `LimitOrderRequest` and risk check (buying power).

Minimal **working** Alpaca connector (no cache, no streaming, no backup) for another project:

```python
# src/connectors/alpaca_connector.py (minimal)
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from .base_connector import BaseConnector
from .exceptions import AuthenticationError, InsufficientLiquidityError
from .log_utils import get_connector_logger


def _parse_timeframe(tf: str):
    tf = (tf or "1h").strip().lower()
    unit = "h"
    if tf.endswith("m"): unit = "m"
    elif tf.endswith("h"): unit = "h"
    elif tf.endswith("d"): unit = "d"
    amount_str = tf.rstrip("mhd") or "1"
    try:
        amount = int(amount_str)
    except ValueError:
        amount = 1
    return amount, unit


class AlpacaConnector(BaseConnector):
    def __init__(self, api_key=None, secret_key=None, paper=True, logs_dir=None):
        self._api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        self._paper = paper
        self._logger = get_connector_logger("alpaca", logs_dir)
        self._trading_client = None
        self._data_client = None

    def connect(self) -> bool:
        if not self._api_key or not self._secret_key:
            raise AuthenticationError("ALPACA_API_KEY and ALPACA_SECRET_KEY are required")
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
            self._trading_client = TradingClient(
                api_key=self._api_key, secret_key=self._secret_key, paper=self._paper
            )
            self._data_client = StockHistoricalDataClient(
                api_key=self._api_key, secret_key=self._secret_key
            )
            self._logger.info("Connected to Alpaca (paper=%s)", self._paper)
            return True
        except Exception as e:
            self._logger.exception("Alpaca connection failed")
            raise AuthenticationError(f"Alpaca connection failed: {e}") from e

    def _ensure_connected(self):
        if self._trading_client is None:
            self.connect()

    def get_ohlcv(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        self._ensure_connected()
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from alpaca.data.enums import DataFeed
        amount, unit = _parse_timeframe(timeframe)
        unit_map = {"m": TimeFrameUnit.Minute, "h": TimeFrameUnit.Hour, "d": TimeFrameUnit.Day}
        alpaca_tf = TimeFrame(amount=amount, unit=unit_map.get(unit, TimeFrameUnit.Hour))
        end = datetime.now(timezone.utc)
        if unit == "m":   start = end - timedelta(minutes=amount * count * 2)
        elif unit == "h": start = end - timedelta(hours=amount * count * 2)
        else:             start = end - timedelta(days=amount * count * 2)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=alpaca_tf,
            start=start, end=end, limit=count,
            feed=DataFeed.IEX,
        )
        bars = self._data_client.get_stock_bars(request)
        return self._bars_to_dataframe(bars, count)

    @staticmethod
    def _bars_to_dataframe(bars, count: int) -> pd.DataFrame:
        if bars is None or (hasattr(bars, "data") and not bars.data):
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        if hasattr(bars, "df") and bars.df is not None and not bars.df.empty:
            df = bars.df.copy()
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
                if "timestamp" in df.columns:
                    df = df.rename(columns={"timestamp": "Timestamp"})
                df = df.set_index("Timestamp")
            for c in list(df.columns):
                if c.lower() in ("open", "high", "low", "close", "volume"):
                    df = df.rename(columns={c: c.capitalize()})
            if "Volume" not in df.columns:
                df["Volume"] = 0
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            return df.sort_index().tail(count)
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    def get_account_state(self) -> dict:
        self._ensure_connected()
        acc = self._trading_client.get_account()
        return {
            "balance": float(getattr(acc, "cash", 0) or 0),
            "equity": float(getattr(acc, "equity", 0) or 0),
            "buying_power": float(getattr(acc, "buying_power", 0) or 0),
            "drawdown": None,
        }

    def execute_order(self, symbol: str, side: str, qty: float, order_type: str = "market",
                      limit_price: float | None = None, **kwargs) -> dict:
        self._ensure_connected()
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        order_side = OrderSide.BUY if (side or "buy").strip().lower() == "buy" else OrderSide.SELL
        try:
            if order_type == "limit" and limit_price is not None:
                req = LimitOrderRequest(symbol=symbol, qty=qty, side=order_side,
                                        time_in_force=TimeInForce.GTC, limit_price=limit_price)
            else:
                req = MarketOrderRequest(symbol=symbol, qty=qty, side=order_side, time_in_force=TimeInForce.GTC)
            order = self._trading_client.submit_order(order_data=req)
            oid = getattr(order, "id", None)
            return {"status": "success", "order_id": str(oid) if oid else None}
        except Exception as e:
            msg = str(e).lower()
            if "insufficient" in msg or "buying power" in msg:
                raise InsufficientLiquidityError(str(e)) from e
            return {"status": "error", "order_id": None, "message": str(e)}
```

### 6.3 Usage (Alpaca)

```python
import os
from dotenv import load_dotenv
load_dotenv()

from src.connectors.alpaca_connector import AlpacaConnector

conn = AlpacaConnector()  # uses ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True by default
conn.connect()

# OHLCV
df = conn.get_ohlcv("AAPL", "1h", 100)   # 100 bars, 1h

# Account
state = conn.get_account_state()  # balance, equity, buying_power

# Order (paper if ALPACA_PAPER=true)
result = conn.execute_order("AAPL", "buy", 1.0, "market")
# result = {"status": "success", "order_id": "..."} or {"status": "error", "message": "..."}
```

---

## 7. MT5 Setup

### 7.1 Prerequisites

- **Windows**: Install [MetaTrader 5](https://www.metatrader5.com/) and log in to your broker. Ensure **“Allow Algorithmic Trading”** is enabled in the terminal.
- **WSL/Linux**: MT5 is Windows-only. The connector will raise `BrokerConnectionError` with a message to use a Windows-hosted REST or Socket bridge that talks to the MT5 terminal.

### 7.2 Connector code

**File: `src/connectors/mt5_connector.py`**

```python
"""
MetaTrader 5 connector (Zero-MQL): headless execution gateway.
Windows-only. On WSL2/Linux, connect() fails with instructions for a Windows bridge.
"""
import sys
from typing import Any

import pandas as pd

from .base_connector import BaseConnector
from .exceptions import BrokerConnectionError
from .log_utils import get_connector_logger

WSL_BRIDGE_MSG = (
    "MT5 is Windows-native and cannot run inside WSL2/Linux. "
    "Run a REST or Socket bridge on the Windows host that connects to the MT5 terminal."
)

def _is_windows() -> bool:
    return sys.platform == "win32"

def _get_mt5():
    import MetaTrader5 as mt5
    return mt5

def _mt5_timeframe(tf: str):
    tf = (tf or "1h").strip().lower()
    mt5 = _get_mt5()
    mapping = {
        "1m": mt5.TIMEFRAME_M1, "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15, "30m": mt5.TIMEFRAME_M30,
        "1h": mt5.TIMEFRAME_H1, "2h": mt5.TIMEFRAME_H2,
        "4h": mt5.TIMEFRAME_H4, "1d": mt5.TIMEFRAME_D1,
        "1w": mt5.TIMEFRAME_W1,
    }
    return mapping.get(tf, mt5.TIMEFRAME_H1)


class MT5Connector(BaseConnector):
    def __init__(self, path: str | None = None, logs_dir: str | None = None):
        self._path = path
        self._logger = get_connector_logger("mt5", logs_dir)
        self._connected = False

    def connect(self) -> bool:
        if not _is_windows():
            self._logger.warning(WSL_BRIDGE_MSG)
            raise BrokerConnectionError(WSL_BRIDGE_MSG)
        mt5 = _get_mt5()
        ok = mt5.initialize(self._path) if self._path else mt5.initialize()
        if not ok:
            err = mt5.last_error()
            raise BrokerConnectionError(f"MT5 initialize failed: {err}")
        self._connected = True
        self._logger.info("MT5 connected (Allow Algorithmic Trading must be enabled)")
        return True

    def _ensure_connected(self):
        if not self._connected:
            self.connect()

    def get_ohlcv(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        self._ensure_connected()
        mt5 = _get_mt5()
        rates = mt5.copy_rates_from_pos(symbol, _mt5_timeframe(timeframe), 0, count)
        if rates is None:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        vol = rates["tick_volume"] if "tick_volume" in rates.dtype.names else 0
        df = pd.DataFrame({
            "Timestamp": pd.to_datetime(rates["time"], unit="s"),
            "Open": rates["open"], "High": rates["high"],
            "Low": rates["low"], "Close": rates["close"],
            "Volume": vol,
        })
        df.set_index("Timestamp", inplace=True)
        return df.sort_index()

    def get_account_state(self) -> dict:
        self._ensure_connected()
        mt5 = _get_mt5()
        acc = mt5.account_info()
        if acc is None:
            raise BrokerConnectionError(f"MT5 account_info failed: {mt5.last_error()}")
        balance = float(getattr(acc, "balance", 0) or 0)
        equity = float(getattr(acc, "equity", 0) or 0)
        margin = float(getattr(acc, "margin", 0) or 0)
        return {
            "balance": balance,
            "equity": equity,
            "margin": margin,
            "drawdown": (equity - balance) if equity < balance else None,
        }

    def execute_order(self, symbol: str, side: str, qty: float, order_type: str = "market",
                      sl: float | None = None, tp: float | None = None, **kwargs: Any) -> dict:
        """qty = volume in lots (MT5)."""
        self._ensure_connected()
        mt5 = _get_mt5()
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {"status": "error", "order_id": None, "message": f"Symbol {symbol} not available"}
        price = float(tick.ask) if (side or "buy").strip().lower() == "buy" else float(tick.bid)
        mt5_type = mt5.ORDER_TYPE_BUY if (side or "buy").strip().lower() == "buy" else mt5.ORDER_TYPE_SELL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": round(qty, 2),
            "type": mt5_type,
            "price": price,
            "magic": kwargs.get("magic", 0),
            "comment": kwargs.get("comment", ""),
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        if sl is not None: request["sl"] = float(sl)
        if tp is not None: request["tp"] = float(tp)
        result = mt5.order_send(request)
        if result is None:
            return {"status": "error", "order_id": None, "message": str(mt5.last_error())}
        retcode = getattr(result, "retcode", None)
        if retcode != mt5.TRADE_RETCODE_DONE:
            return {"status": "error", "order_id": getattr(result, "order", None), "message": f"retcode={retcode}"}
        return {"status": "success", "order_id": str(getattr(result, "order", ""))}
```

### 7.3 Usage (MT5)

- Ensure MT5 terminal is **running and logged in** on Windows.
- Symbols must match your broker (e.g. `US30`, `US100`, `EURUSD`).

```python
from src.connectors.mt5_connector import MT5Connector

conn = MT5Connector()  # or MT5Connector(path="C:/path/to/terminal64.exe")
conn.connect()

df = conn.get_ohlcv("US30", "1h", 100)
state = conn.get_account_state()
result = conn.execute_order("US30", "buy", 0.01, "market")  # 0.01 lots
```

---

## 8. Connector Factory

Use one entry point and switch brokers via env or parameter.

**File: `src/connectors/connector_factory.py`**

```python
import logging
import os
import sys
from typing import Optional

from .base_connector import BaseConnector
from .exceptions import BrokerConnectionError

logger = logging.getLogger(__name__)

BROKER_TYPE_ENV = "BROKER_TYPE"
BROKER_ALPACA = "alpaca"
BROKER_MT5 = "mt5"

def _is_wsl_or_linux() -> bool:
    if sys.platform == "linux":
        return True
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    return False

def _get_alpaca_connector() -> Optional[BaseConnector]:
    try:
        from .alpaca_connector import AlpacaConnector
        return AlpacaConnector()
    except Exception as e:
        logger.warning("Alpaca connector not available: %s", e)
        return None

def _get_mt5_connector() -> Optional[BaseConnector]:
    if _is_wsl_or_linux():
        logger.warning("MT5 is Windows-native; use a Windows bridge for WSL/Linux.")
        return None
    try:
        from .mt5_connector import MT5Connector
        return MT5Connector()
    except ImportError as e:
        logger.warning("MT5 connector not available: %s", e)
        return None

def get_connector(broker_type: Optional[str] = None) -> BaseConnector:
    broker = (broker_type or os.environ.get(BROKER_TYPE_ENV) or BROKER_ALPACA).strip().lower()
    if broker == BROKER_ALPACA:
        conn = _get_alpaca_connector()
        if conn is None:
            raise BrokerConnectionError("Alpaca not initialized. Set ALPACA_API_KEY and ALPACA_SECRET_KEY.")
        return conn
    if broker == BROKER_MT5:
        conn = _get_mt5_connector()
        if conn is None:
            raise BrokerConnectionError("MT5 not available. On Windows: install MetaTrader5 and run terminal.")
        return conn
    raise BrokerConnectionError(f"Unknown BROKER_TYPE: {broker}. Use 'alpaca' or 'mt5'.")
```

**Usage:**

```python
from src.connectors.connector_factory import get_connector

# Uses BROKER_TYPE from env (default alpaca)
conn = get_connector()
conn.connect()

# Or explicit
conn = get_connector("alpaca")
conn = get_connector("mt5")
```

**File: `src/connectors/__init__.py`**

```python
from .base_connector import BaseConnector
from .connector_factory import get_connector
from .exceptions import BrokerConnectionError, AuthenticationError, InsufficientLiquidityError, ConnectorError

__all__ = [
    "BaseConnector",
    "get_connector",
    "ConnectorError",
    "BrokerConnectionError",
    "InsufficientLiquidityError",
    "AuthenticationError",
]
```

---

## 9. Supporting Modules

### 9.1 CacheManager (optional, for Alpaca)

Used by the full `AlpacaConnector` to cache OHLCV and reduce REST calls. See `src/connectors/cache_manager.py` in this repo: in-memory TTL cache with optional disk (Parquet). Key API: `make_key(symbol, timeframe, count)`, `get(key)`, `set(key, df)`.

### 9.2 ExecutionManager (optional, for Alpaca)

Token-bucket throttling and optional multi-account failover. See `src/connectors/execution_manager.py`. Use when submitting many orders (e.g. respecting Alpaca’s ~200 orders/min). Non-async entry: `submit_order_sync(conn, symbol, side, qty, ...)`.

### 9.3 API service (optional, for REST/UI)

This project’s FastAPI app uses `src/api/alpaca_service.py`: lazy `AlpacaConnector`, `is_alpaca_available()`, `get_account()`, `get_positions()`, `get_portfolio_history()`, `get_last_quote()`. You can copy that module and point it at your `AlpacaConnector` for a “Trader-Suit” style API.

---

## 10. Optional: Alpaca MCP Server

The repo includes an MCP server that exposes Alpaca to Cursor/IDE via the Model Context Protocol (stdio).

- **Entrypoint**: `python -m src.mcp.alpaca_server`
- **Config** (e.g. Cursor MCP): in `mcp_config.json`:
  ```json
  {
    "mcpServers": {
      "alpaca": {
        "command": "python",
        "args": ["-m", "src.mcp.alpaca_server"],
        "cwd": "C:\\path\\to\\your\\project"
      }
    }
  }
  ```
- **Requirements**: Same as Alpaca connector; optional `mcp` SDK (falls back to a minimal JSON-RPC server).
- **Tools**: `get_market_data`, `submit_order`, `get_positions`, `get_account_info`. Resources: `alpaca://market_status`, `alpaca://account`.

To reuse in another project: copy `src/mcp/alpaca_server.py` and ensure `src.connectors.alpaca_connector` is on the path and env has Alpaca keys.

---

## 11. Quick Start for a New Project

1. **Dependencies**  
   Add `alpaca-py` (and optionally `MetaTrader5` on Windows) and `pandas`, `python-dotenv`.

2. **Env**  
   Create `.env` with `BROKER_TYPE=alpaca`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`. Use paper keys first.

3. **Files to copy**  
   - `base_connector.py`, `exceptions.py`, `log_utils.py`  
   - `alpaca_connector.py` (full or minimal above)  
   - `mt5_connector.py` (if you need MT5 on Windows)  
   - `connector_factory.py`, `__init__.py`

4. **Connect and use**  
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   from src.connectors import get_connector

   conn = get_connector()
   conn.connect()
   df = conn.get_ohlcv("AAPL", "1h", 100)
   ```

5. **MT5 (Windows only)**  
   Install MetaTrader 5, log in, enable “Allow Algorithmic Trading”. Set `BROKER_TYPE=mt5` and use `get_connector("mt5")`.

6. **Alpaca data note**  
   Free-tier Alpaca uses IEX feed; some symbols (e.g. US30 CFD) may not be available. Use US equities (e.g. AAPL, SPY) for testing.

This gives you the same Alpaca + MT5 connector setup in another project with minimal copy-paste and env configuration.
