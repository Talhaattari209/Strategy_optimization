from __future__ import annotations

from typing import Dict, Optional

try:
    from agents import Agent
except ImportError:  # pragma: no cover - optional dependency
    Agent = object


def _sub_instructions(base_skill: str, specialization: str) -> str:
    return (
        f"{base_skill}\n\n"
        "You are a specialist sub-agent operating under the US30_Quant_Orchestrator.\n"
        f"Specialization: {specialization}\n"
        "Always obey risk guardrails: max 2% risk per trade and max 5% portfolio drawdown.\n"
        "Return concise reasoning plus recommended next action."
    )


def build_sub_agents(base_skill: str, model=None) -> Dict[str, Optional[Agent]]:
    """
    Build sub-agents used as handoff specialists from the orchestrator.
    Returns a dict even when agents SDK is unavailable.
    """
    if Agent is object:
        return {
            "math_optimizer": None,
            "rl_optimizer": None,
            "regime_agent": None,
            "execution_agent": None,
        }

    return {
        "math_optimizer": Agent(
            name="MathOptimizerAgent",
            instructions=_sub_instructions(
                base_skill,
                "Optimize linear_algebra, time_series, and stochastic_calculus usage and stability.",
            ),
            model=model,
        ),
        "rl_optimizer": Agent(
            name="RLOptimizerAgent",
            instructions=_sub_instructions(
                base_skill,
                "Analyze DRL convergence, overfitting, validation gap, and retraining actions.",
            ),
            model=model,
        ),
        "regime_agent": Agent(
            name="RegimeAgent",
            instructions=_sub_instructions(
                base_skill,
                "Run Markov + Bayesian regime interpretation and regime-flip implications.",
            ),
            model=model,
        ),
        "execution_agent": Agent(
            name="ExecutionAgent",
            instructions=_sub_instructions(
                base_skill,
                "Optimize execution with microstructure SDE and Girsanov-guided entry adaptation.",
            ),
            model=model,
        ),
    }

