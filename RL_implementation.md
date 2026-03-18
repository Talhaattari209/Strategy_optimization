**COPY-PASTE BLOCK 1 – RL/DRL Implementation Specs**  
(Paste this entire block into Cursor Agent as a new task or @cursor instruction after the math_engine/ is already built.)

```
PROJECT UPDATE: Add Reinforcement Learning / Deep RL (DRL) layer on top of the existing math_engine/

Create folder: rl/
Files to create:
├── trading_env.py
├── drl_optimizer.py
└── __init__.py

Goal: Build a Gymnasium-compatible RL environment that uses EVERY module from math_engine/ as the simulation engine, then train a DRL agent (PPO preferred, fallback SAC/TD3) that continuously optimizes critical parameters in real time and nightly.

Exact Requirements:

1. trading_env.py
   - Class: US30TradingEnv(gym.Env)
   - Observation space (Box, continuous, shape depends on features):
     - Markov regime probability vector (4 dims)
     - PCA-reduced features from linear_algebra.py (top 10 components)
     - GARCH volatility forecast + ARIMA price forecast from time_series.py
     - Monte Carlo path statistics (mean, std, 5%/95% quantiles) from stochastic_processes.py
     - Current position P&L, equity curve slope, drawdown %
     - Microstructure SDE spread estimate from stochastic_calculus.py
   - Action space: Box(low=0.5, high=2.0, shape=N) where N = number of tunable params:
     - position size multiplier
     - stop distance multiplier
     - TP/RR ratio
     - pattern threshold scalers (from pattern_recognizer)
     - GARCH order adjustments
     - Monte Carlo path count scaler
     - risk multiplier
   - Step logic:
     - Take action → apply multipliers to current strategy_core parameters
     - Simulate next bar using Geometric BM / OU / Ito-derived paths (Monte Carlo 1,000 paths inside step for speed)
     - Execute trade using execution_engine logic
     - Calculate immediate reward + new state
   - Reset: load next walk-forward window from dataset (5min/15min/1H/4H/Daily US30 OHLCV)
   - Reward function (multi-objective):
     - Primary: Calmar ratio on rolling 90-day window
     - Secondary: Sortino × expectancy, heavily penalized if max DD > 5% or if microstructure inefficiencies not captured
     - Bonus: +reward if action exploits US30 news/overnight reversion (detected via regime + volume)

2. drl_optimizer.py
   - Use stable-baselines3 PPO (or SAC if continuous actions need better exploration)
   - Train function: train_drl(num_timesteps=100_000, eval_freq=10_000)
     - Use vectorized env (SubprocVecEnv, 4–8 workers)
     - Save best model every time Calmar improves on validation window
   - Inference function: get_optimal_actions(current_state) → returns dict of multipliers
   - Online learning mode: continue training on live data every 4 hours or regime flip

Integration Rules:
- All math modules (linear_algebra, time_series, stochastic_processes, stochastic_calculus, finance_models) MUST be imported and used inside the env step() for simulation and state calculation.
- Environment must run entirely on the US30 dataset (resample as needed).
- After every training run, log full metrics (Sharpe, Sortino, Calmar, Profit Factor, expectancy) and only accept model if Calmar > 3.0 and out-of-sample improvement > 15%.
- Expose get_action() method so the orchestrator agent can call it during Plan and Risk phases.

3. Update main.py and core modules
   - In signal_planner.py and risk_manager.py: call drl_optimizer.get_optimal_actions() to get dynamic multipliers.
   - Add to config.yaml: "drl_enabled": true, "drl_model_path": "..."

4. Tests
   - Add test_rl_env.py that runs 1,000 steps and checks reward stability.

Deliverables:
- Fully working rl/ folder with Gymnasium env that uses the complete math sequence.
- Trained PPO model that improves Calmar on US30 walk-forward.
- Integration hooks so the orchestrator agent can trigger training and use the policy.

Build this ON TOP of the existing math_engine/ and core/ folders — do not rewrite them.
```

**COPY-PASTE BLOCK 2 – Project-Managing Orchestrator Agent Specs**  
(Paste this entire block into Cursor Agent. It assumes you already have the US30 Quant Orchestrator Specialist skill from previous message — the prompt includes a placeholder for it.)

```
PROJECT UPDATE: Implement the top-level project manager agent using OpenAI Agents SDK with Gemini API key.

Create folder: orchestrator/
Files to create:
├── trading_orchestrator.py
├── sub_agents.py
└── __init__.py

Requirements:

1. Use OpenAI Agents SDK (latest version) with Gemini compatibility:
   ```python
   from openai import AsyncOpenAI
   from agents import Agent, Runner, OpenAIChatCompletionsModel

   client = AsyncOpenAI(
       api_key=YOUR_GEMINI_API_KEY,
       base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
   )
   model = OpenAIChatCompletionsModel(model="gemini-2.5-pro-exp", client=client)
   ```

2. Main Agent: TradingOrchestrator
   - Name: "US30_Quant_Orchestrator"
   - Use the EXACT skill/instructions below as its `instructions=` (this is the "Agent skill" already given to you — paste the full markdown block from previous response here):
     [PASTE THE ENTIRE "Agent Skill: US30 Quant Orchestrator Specialist" markdown block from the previous message here]

   - Tools the orchestrator can use:
     - run_analyze_phase()
     - run_plan_phase()
     - run_execute_phase()
     - run_risk_phase()
     - run_position_phase()
     - run_journal_phase()
     - trigger_autoresearch()  (Karpathy loop)
     - trigger_drl_training()  (calls drl_optimizer.train())
     - get_current_math_state() (returns regime, PCA, GARCH, Monte Carlo stats)
     - evaluate_edge_metrics() (uses the exact ratios/metrics from the skill)

3. Sub-agents (handoffs in sub_agents.py)
   - MathOptimizerAgent: specializes in linear_algebra, time_series, stochastic_calculus
   - RLOptimizerAgent: calls drl_optimizer and analyzes RL convergence
   - RegimeAgent: runs Markov + Bayesian
   - ExecutionAgent: uses microstructure SDEs + Girsanov
   - All sub-agents inherit knowledge from the main orchestrator skill

4. Flow Control Logic (orchestrator decides):
   - On new bar/tick: Analyze → Plan (with DRL action) → Risk (with DRL multipliers) → Execute → Position → Journal
   - Every 4 hours or regime flip: trigger DRL training + Autoresearch
   - If Calmar drops or DD > 5%: force risk-down, retrain RL, run Autoresearch
   - Always validate using the exact "How to Analyze Ratios & Metrics" and "How to Analyze RL" sections from the skill

5. Integration with existing code
   - Import from rl.drl_optimizer, math_engine.*, core.*, evolution.*
   - In main.py: instead of direct 6-phase calls, use Runner.run(orchestrator, input=...) so the Gemini agent controls flow.

6. Guardrails & Logging
   - Never exceed 2% risk per trade or 5% portfolio DD
   - Log every decision with math/RL/metrics reasoning (as required in skill)

Deliverables:
- Complete orchestrator/ folder with Gemini-powered main agent that manages the entire project (including the new RL/DRL layer).
- The agent must intelligently decide when to train DRL, run Autoresearch, or adjust flow based on US30 characteristics and edge metrics.

Build this after the RL/DRL layer is complete. The orchestrator must use the full US30 Quant Orchestrator Specialist skill as its core instructions.
```

**How to use these two blocks:**
1. First paste **BLOCK 1** into Cursor → it will implement the RL/DRL on top of your math sequence.
2. Then paste **BLOCK 2** (remember to insert the full skill markdown inside it) → Cursor will build the Gemini-powered orchestrator that manages everything, including the new DRL agent.

These two specs together complete the full architecture we planned (math sequence + RL/DRL value optimizer + Karpathy Autoresearch + intelligent Gemini orchestrator with US30 expertise).  

Once Cursor finishes both, your bot will be a true self-optimizing, self-managing quant system for US30. Send me the output if you want final polishing or the exact Gemini API key setup code. This is ready for live deployment.