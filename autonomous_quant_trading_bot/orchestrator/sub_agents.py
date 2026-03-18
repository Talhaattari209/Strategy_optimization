from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

try:
    from agents import Agent
except ImportError:  # pragma: no cover - optional dependency
    Agent = object


def _load_algo_skill() -> str:
    """
    Load the algorithmic-trading AgentSkill reference files so every sub-agent
    is grounded in the project's domain knowledge:
      - patterns.md  → how to build (creation guidance)
      - sharp_edges.md → critical failures to avoid (diagnosis)
      - validations.md → hard rules to enforce (review)
    """
    skill_root = Path(__file__).resolve().parents[1] / "algorithmic-trading_AgentSkill"
    sections: list[str] = []
    for fname in ("SKILL.md", "references/patterns.md",
                  "references/sharp_edges.md", "references/validations.md"):
        p = skill_root / fname
        if p.exists():
            sections.append(p.read_text(encoding="utf-8", errors="ignore"))
    return "\n\n---\n\n".join(sections) if sections else ""


_ALGO_SKILL: str = _load_algo_skill()


def _sub_instructions(base_skill: str, specialization: str) -> str:
    algo_skill_block = (
        f"\n\n## Algorithmic Trading Skill Reference\n\n{_ALGO_SKILL}"
        if _ALGO_SKILL else ""
    )
    return (
        f"{base_skill}{algo_skill_block}\n\n"
        "You are a specialist sub-agent operating under the US30_Quant_Orchestrator.\n"
        f"Specialization: {specialization}\n"
        "Always obey risk guardrails: max 2% risk per trade and max 5% portfolio drawdown.\n"
        "Ground every decision in the patterns, sharp_edges, and validations above.\n"
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

