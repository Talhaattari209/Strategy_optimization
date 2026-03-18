# Google Colab Runner — Setup Guide

Run the entire Autonomous Quant Trading Bot on **Google Colab GPU/TPU** while editing all `.py` files in Cursor.  
All files and the Python environment are persisted permanently on Google Drive.

---

## How It Works

```
Cursor IDE (your machine)          Google Colab (cloud)
──────────────────────────         ─────────────────────────────────────
Edit .py files on Drive    ───►    main.ipynb reads them via sys.path
Open main.ipynb in Cursor  ───►    Colab extension runs cells on GPU
View outputs / charts      ◄───    Results saved back to Drive
```

---

## Step 1 — Copy Project to Google Drive

Upload the entire `autonomous_quant_trading_bot/` folder to your Google Drive:

```
MyDrive/
└── autonomous_quant_trading_bot/
    ├── main.ipynb          ← the notebook (already created)
    ├── main.py
    ├── config.yaml
    ├── requirements.txt
    ├── core/
    ├── data/
    ├── rl/
    ├── math_engine/
    ├── evolution/
    ├── orchestrator/
    ├── backtester/
    └── utils/
```

---

## Step 2 — Open the Notebook in Cursor

1. In Cursor, open **`autonomous_quant_trading_bot/main.ipynb`**
2. Click the **kernel picker** (top-right of the notebook editor)
3. Select **"Connect to Colab"** (requires the Colab extension)
4. Sign in with your Google account → Colab runtime starts

> **Runtime type**: Go to `Runtime → Change runtime type → GPU (T4 or A100)` in Colab for hardware acceleration.

---

## Step 3 — Configure & Run All

### 3a: Edit the Master Config cell (top of notebook)

All parameters live in **one single cell** at the top of `main.ipynb`:

```python
# API Keys
GEMINI_API_KEY  = 'your-key'
OPENAI_API_KEY  = ''

# Symbol & balance
SYMBOL          = 'US30'
INITIAL_BALANCE = 10_000.0

# Data
DATA_CSV        = 'US30_H1.csv'   # or '' to auto-scan data/ folder

# Task toggles — True = run, False = skip
RUN_BACKTEST    = True
RUN_DRL         = True
RUN_DRL_ONLINE  = True
RUN_AUTORESEARCH= True
RUN_VISUALISE   = True
RUN_LIVE        = False   # set True only when broker is connected

# DRL settings
DRL_ALGO        = 'PPO'        # 'PPO' | 'SAC' | 'TD3'
DRL_TIMESTEPS   = 200_000

# Autoresearch
AUTO_CYCLES     = 20
```

### 3b: Run All

Go to **`Runtime → Run all`** — the entire pipeline executes top to bottom:

```
Master Config → Drive Mount → Install → GPU Detect → Imports → Load Config
     → Load Data → Backtest → DRL Train → DRL Online → Autoresearch
     → Visualise → Live Trading → Resource Stats
```

Every task prints `⏭ skipped` if its toggle is `False`, so `Run All` is always safe.

---

## GPU Acceleration Details

The following modules are automatically GPU-accelerated when CUDA is available:

| Module | Acceleration |
|--------|-------------|
| `math_engine/stochastic_processes.py` | PyTorch CUDA — GBM & OU Monte Carlo paths |
| `rl/drl_optimizer.py` | Stable-Baselines3 — PPO/SAC/TD3 training on `device="cuda"` |
| Cell 14 (notebook) | Pure PyTorch GBM benchmark |

All other modules use NumPy/SciPy on CPU (already fast enough).

---

## API Keys

Set your keys in **Cell 5** of the notebook:

```python
GEMINI_API_KEY = 'your-key-here'   # Required for orchestrator agent
OPENAI_API_KEY = 'your-key-here'   # Optional
```

Keys are stored only in Colab memory for the session — never written to disk.

---

## Data Files

Upload your historical CSVs to `MyDrive/autonomous_quant_trading_bot/data/`.

Supported formats:
- **Standard**: `time, open, high, low, close, volume`
- **MT5 export**: `<DATE>, <TIME>, <OPEN>, <HIGH>, <LOW>, <CLOSE>, <TICKVOL>`

Cell 6 auto-detects and normalises both formats.

---

## Subsequent Sessions

After the first install, subsequent sessions only need:
1. Run Cell 1 (mount Drive)
2. Run Cell 2 (re-install packages — ~2 min, cached by pip)
3. Run Cells 3–5 (GPU detect + imports + config)
4. Run your task cell

> **Tip**: Colab Pro / Pro+ gives you longer runtimes, more RAM, and A100 GPU access — highly recommended for overnight autoresearch runs.
