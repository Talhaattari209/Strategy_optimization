
---

**PROJECT NAME:** `autonomous_quant_trading_bot`  
**LANGUAGE:** Pure Python 3.11+ (no MQL5, no .mqh)  
**GOAL:** Build a complete, production-ready, fully autonomous trading bot that **starts exactly with the client’s proven edge** and then autonomously evolves it forever using Karpathy-style Autoresearch + all the advanced math topics.

**Client’s Exact Edge (MUST be hard-coded as the unbreakable foundation):**
The bot’s base signals are 100% derived from these four combined factors:
1. **What time (session) is it?** → Detect current session (Asia, London, New York, Sydney) and session overlaps.
2. **At what level?** → 
   - HOD (High of Day), LOD (Low of Day)
   - HO Session / LOS (High/Low of current session)
   - HO Week / LO Week (High/Low of Week)
   - Major structure levels: CHOCH (Change of Character), BOS (Break of Structure)
   - Close of Day level
3. **Price behavior at that level** → What candles are forming right now at the level (pinbar, engulfing, doji, rejection wick, strong impulse candle, inside bar, etc.).
4. The confluence of 1+2+3 creates the base trade signal (e.g., “London open + price at weekly high + bearish engulfing = short bias”).

All other logic (volume, news filter, volatility, risk) from the original request must still be included but **only as supporting filters** — never override the above core edge.

### 1. Core Requirements (unchanged except edge encoding)
- Fully autonomous 6-phase loop: Analyze → Plan → Execute → Risk → Position Management → Journal.
- Regime detection + self-optimization + edge evolution.
- Hybrid Edge: Client’s exact rules above = foundation. Autoresearch mutates parameters, adds micro-variations, discovers new candle-level-session confluences, but **never deletes** the core logic.
- Math mastery: Every topic from the sequence implemented and used exactly where it adds value.
- Backtesting & Live mode (MetaTrader5 Python package or ccxt pluggable).
- Karpathy Autoresearch loop for overnight self-improvement.

### 2. Exact File Structure (same as before + new emphasis)
```
autonomous_quant_trading_bot/
├── main.py
├── config.yaml                      # session times, level tolerances, candle definitions
├── data/collector.py
├── core/
│   ├── regime_detector.py
│   ├── market_structure.py          # CHOCH, BOS, swing detection
│   ├── level_detector.py            # NEW: HOD/LOD, session/week highs-lows, close-of-day
│   ├── session_timer.py             # NEW: precise session & overlap detection
│   ├── candle_analyzer.py           # NEW: detects candle types at levels
│   ├── pattern_recognizer.py        # Core client edge confluence engine
│   ├── signal_planner.py
│   ├── execution_engine.py
│   ├── risk_manager.py
│   ├── position_manager.py
│   └── journal.py
├── math_engine/                     # (exactly as before)
├── evolution/                       # Karpathy loop
├── backtester/
├── utils/
├── tests/
└── results/
```

### 3. Math Sequence & Exact Usage Mapping (unchanged but now tied to client edge)
**Linear Algebra** → PCA on features including “distance to HOD/LOD”, “session time as sine/cosine”, candle stats.  
**Markov + Bayesian** → Regime detector now includes session state + level proximity as hidden states.  
**Fourier** → Extract session-cycle components.  
**ARIMA/GARCH** → Forecast volatility inside current session.  
**Monte Carlo + GBM + OU** → Simulate paths from current level (HOD/LOD etc.) until next session close.  
**Ito’s Lemma + Girsanov** → Optimal TP/SL and execution slicing at key levels.  
**Black-Scholes risk-neutral + technical Fama-French** → Expectancy check when price reacts at CHOCH/BOS.

### 4. Client Edge Implementation Details (MANDATORY — code exactly this)
In `pattern_recognizer.py` + `level_detector.py` + `session_timer.py` + `candle_analyzer.py`:

- **Session detection** (use UTC + configurable broker times):
  - Asia: 00:00–08:00
  - London: 08:00–16:00
  - New York: 13:00–21:00
  - Overlaps flagged automatically.

- **Level calculation** (rolling):
  - HOD/LOD: reset at day start (New York close or user-defined).
  - Session HO/LOS: reset at session start.
  - Weekly HO/LO: reset Sunday/Monday.
  - CHOCH/BOS: use fractal/swing detection with confirmation.
  - Close-of-Day: previous day close.

- **Price behavior at level**:
  - Detect candle type within 5–10 pips of level.
  - Supported patterns (expandable via autoresearch): pinbar (wick > 2× body), engulfing, doji, rejection (long wick), impulse (strong body), inside bar.
  - Measure wick/body ratio, direction relative to level.

- **Confluence engine** returns a base signal dict:
  ```python
  {
    "bias": "bullish" | "bearish" | "neutral",
    "confidence": 0.0–1.0,
    "reason": "London open + at weekly high + bearish engulfing",
    "level_type": "HO Week",
    "session": "London",
    "candle_type": "bearish_engulfing"
  }
  ```

This base signal is then passed to **Plan** phase where math layers (Monte Carlo, Ito, etc.) refine entry, TP, SL.

### 5. Autoresearch Layer (evolution/program.md must start with this)
```markdown
You are a senior quant researcher working for a $5k client project.
Client edge foundation (NEVER remove or override):
- Session timing + exact levels (HOD, LOD, HO/LOS session, HO/LO week, CHOCH, BOS, close-of-day)
- Price behavior = candle type at the level

Your job: mutate parameters (level tolerance, candle thresholds, session weights), add new micro-confluences, evolve candle detection logic, improve risk scaling inside sessions — but always keep the four-factor core intact.

Goal metric: strictly higher Calmar ratio on 3-year walk-forward out-of-sample.
```

### 6. Technical Stack & Additional Requirements (same as before)
- numpy, scipy, pandas, statsmodels, vectorbt
- Full type hints + docstrings explaining which math topic + which part of client edge is used
- Configurable via config.yaml (session times, pip tolerances, candle definitions)
- Walk-forward + Monte Carlo robustness
- Beautiful console output with session + level + candle info

**Start building in this order:**
1. math_engine/ (full sequence)
2. session_timer.py + level_detector.py + candle_analyzer.py + market_structure.py
3. pattern_recognizer.py (core client edge)
4. Rest of core/ + 6-phase loop
5. evolution/ Autoresearch

Deliverables after first pass:
- Math modules with tests
- Full backtest on EURUSD/GBPUSD H1 2020–2026 showing client edge working
- Autoresearch loop ready to run overnight

This spec is now 100% complete with the client’s exact edge. Cursor will build precisely what you need.

---

**Paste the whole thing into Cursor now.**  
It will generate the bot with your client’s real edge hard-coded as the foundation (session + levels + candle behavior), then layer all the math and autoresearch on top.  

When it finishes the first version, paste the main output here and I’ll give you the next refinements (exact candle definitions, level tolerance logic, live MT5 bridge, etc.). This is now perfectly tailored to your client.