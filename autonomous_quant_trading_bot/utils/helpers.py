"""
Utility helpers — config loading, console output, timing.
"""
from __future__ import annotations

import yaml
import time
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


logger = logging.getLogger("trading_bot")


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class ConsoleDisplay:
    """Beautiful console output with session + level + candle info."""

    def __init__(self) -> None:
        self.console = Console() if RICH_AVAILABLE else None

    def print_signal(self, signal: Dict) -> None:
        if not RICH_AVAILABLE:
            print(f"[SIGNAL] {signal.get('bias', 'N/A')} | {signal.get('reason', '')} | conf={signal.get('confidence', 0):.2f}")
            return

        bias = signal.get("bias", "neutral")
        color = "green" if bias == "bullish" else "red" if bias == "bearish" else "yellow"

        table = Table(title=f"Trade Signal", border_style=color)
        table.add_column("Field", style="bold")
        table.add_column("Value")
        table.add_row("Bias", f"[{color}]{bias.upper()}[/{color}]")
        table.add_row("Confidence", f"{signal.get('confidence', 0):.2%}")
        table.add_row("Session", signal.get("session", ""))
        table.add_row("Level", signal.get("level_type", ""))
        table.add_row("Candle", signal.get("candle_type", ""))
        table.add_row("Reason", signal.get("reason", ""))
        self.console.print(table)

    def print_trade(self, trade: Dict) -> None:
        if not RICH_AVAILABLE:
            print(f"[TRADE] {trade.get('symbol', '')} | PnL={trade.get('pnl', 0):.2f} | {trade.get('pnl_pips', 0):.1f} pips")
            return

        pnl = trade.get("pnl", 0)
        color = "green" if pnl > 0 else "red"
        direction = "LONG" if trade.get("direction", 0) == 1 else "SHORT"

        panel = Panel(
            f"[bold]{direction}[/bold] {trade.get('symbol', '')}\n"
            f"Entry: {trade.get('entry', 0):.5f} -> Exit: {trade.get('exit', 0):.5f}\n"
            f"PnL: [{color}]{pnl:+.2f}[/{color}] ({trade.get('pnl_pips', 0):+.1f} pips)\n"
            f"Duration: {trade.get('duration_minutes', 0):.0f} min",
            title="Trade Closed",
            border_style=color,
        )
        self.console.print(panel)

    def print_regime(self, regime: Dict) -> None:
        if not RICH_AVAILABLE:
            print(f"[REGIME] {regime.get('regime', 'N/A')} | vol={regime.get('volatility', 0):.6f}")
            return

        self.console.print(f"[bold cyan]Regime:[/bold cyan] {regime.get('regime', 'N/A')} "
                           f"(conf={regime.get('confidence', 0):.2%}, vol={regime.get('volatility', 0):.6f})")

    def print_summary(self, summary: Dict) -> None:
        if not RICH_AVAILABLE:
            for k, v in summary.items():
                print(f"  {k}: {v}")
            return

        table = Table(title="Performance Summary", border_style="blue")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        for k, v in summary.items():
            table.add_row(k.replace("_", " ").title(), str(v))
        self.console.print(table)


class Timer:
    """Context manager for timing operations."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.elapsed: float = 0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed = time.perf_counter() - self._start
        if self.name:
            logger.debug(f"{self.name}: {self.elapsed:.4f}s")
