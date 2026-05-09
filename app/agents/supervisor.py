"""
app/agents/supervisor.py

Supervisor orchestrator for the Resume Coach pipeline.

Responsibilities:
  1. Budget enforcement — tracks token cost and time across all agents
  2. Routing decisions — decides which agents to run based on request type
  3. Retry logic — retries failed agents once with simplified prompt signal
  4. Final validation — checks pipeline completeness before returning

Budget defaults (configurable via env):
  MAX_TOKENS_PER_RUN  = 8000   (Replicate/Mistral tokens)
  MAX_COST_USD        = 0.10   (10 cents per run hard cap)
  MAX_TIME_SECS       = 120    (2 minute wall clock limit)

Routing modes:
  full        → all 5 agents (default)
  quick       → gap_analyzer + ats_scorer only (fast path)
  coach_only  → gap_analyzer + coach_writer only
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Literal

from app.core.state import ResumeCoachState

logger = logging.getLogger(__name__)

# ── Budget defaults ───────────────────────────────────────────────────────────
DEFAULT_MAX_TOKENS   = 8_000
DEFAULT_MAX_COST_USD = 0.10
DEFAULT_MAX_TIME_SECS = 120

# ── Cost per 1K tokens by provider (approximate) ─────────────────────────────
COST_PER_1K_TOKENS = {
    "replicate": 0.0002,    # ~$0.20 per 1M tokens for Llama-3-8B
    "sagemaker": 0.0,       # self-hosted — no per-token cost, just instance time
    "ollama":    0.0,       # local — no cost
    "bedrock":   0.003,     # Claude Sonnet on Bedrock
    "openai":    0.005,     # GPT-4o
}

RoutingMode = Literal["full", "quick", "coach_only"]


@dataclass
class BudgetGuard:
    """
    Tracks and enforces resource budgets across a pipeline run.
    Passed through state and updated after each agent completes.
    """
    max_tokens:    int   = DEFAULT_MAX_TOKENS
    max_cost_usd:  float = DEFAULT_MAX_COST_USD
    max_time_secs: float = DEFAULT_MAX_TIME_SECS

    elapsed_tokens: int   = 0
    elapsed_cost:   float = 0.0
    start_time:     float = field(default_factory=time.time)
    provider:       str   = "replicate"

    # Per-agent token estimates (prompt + completion)
    AGENT_TOKEN_ESTIMATES = {
        "ml_fast_pass":        0,     # no LLM call
        "gap_analyzer":        2_500,
        "ats_scorer":          2_000,
        "coach_writer":        3_000,
        "interview_generator": 2_000,
    }

    def can_run(self, agent_name: str) -> tuple[bool, str]:
        """
        Check if an agent can run within remaining budget.
        Returns (allowed, reason_if_blocked).
        """
        estimated_tokens = self.AGENT_TOKEN_ESTIMATES.get(agent_name, 1_000)
        estimated_cost   = (estimated_tokens / 1000) * COST_PER_1K_TOKENS.get(self.provider, 0.0)
        elapsed_secs     = time.time() - self.start_time

        if self.elapsed_tokens + estimated_tokens > self.max_tokens:
            return False, (
                f"Token budget exhausted: {self.elapsed_tokens}/{self.max_tokens} used, "
                f"{agent_name} needs ~{estimated_tokens} more"
            )

        if self.elapsed_cost + estimated_cost > self.max_cost_usd:
            return False, (
                f"Cost budget exhausted: ${self.elapsed_cost:.4f}/${self.max_cost_usd:.2f} used"
            )

        if elapsed_secs > self.max_time_secs:
            return False, (
                f"Time budget exhausted: {elapsed_secs:.0f}s/{self.max_time_secs:.0f}s elapsed"
            )

        return True, ""

    def consume(self, agent_name: str, actual_tokens: int | None = None) -> None:
        """Record resource consumption after an agent completes."""
        tokens = actual_tokens or self.AGENT_TOKEN_ESTIMATES.get(agent_name, 1_000)
        cost   = (tokens / 1000) * COST_PER_1K_TOKENS.get(self.provider, 0.0)
        self.elapsed_tokens += tokens
        self.elapsed_cost   += cost
        logger.debug(
            "BudgetGuard.consume: agent=%s tokens=%d total_tokens=%d cost=$%.4f",
            agent_name, tokens, self.elapsed_tokens, self.elapsed_cost,
        )

    def summary(self) -> dict:
        return {
            "elapsed_tokens":  self.elapsed_tokens,
            "elapsed_cost_usd": round(self.elapsed_cost, 4),
            "elapsed_secs":    round(time.time() - self.start_time, 1),
            "token_budget_pct": round(self.elapsed_tokens / self.max_tokens * 100, 1),
            "cost_budget_pct":  round(self.elapsed_cost / self.max_cost_usd * 100, 1) if self.max_cost_usd > 0 else 0,
        }


class SupervisorAgent:
    """
    Orchestrator that wraps the LangGraph pipeline with:
      - Pre-flight budget checks before each agent
      - Routing decisions based on request type
      - Retry logic for failed agents
      - Post-run validation and summary
    """

    def __init__(
        self,
        provider: str = "replicate",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_cost_usd: float = DEFAULT_MAX_COST_USD,
        max_time_secs: float = DEFAULT_MAX_TIME_SECS,
    ):
        self.provider      = provider
        self.max_tokens    = max_tokens
        self.max_cost_usd  = max_cost_usd
        self.max_time_secs = max_time_secs

    def decide_routing(self, state: ResumeCoachState) -> RoutingMode:
        """
        Determine which agents to run based on input characteristics.

        Rules:
          - Resume < 500 chars → quick mode (likely test/minimal input)
          - JD < 200 chars     → quick mode (no JD to coach against)
          - Otherwise          → full pipeline
        """
        resume_len = len(state.get("resume_text", ""))
        jd_len     = len(state.get("jd_text", ""))

        if resume_len < 500 or jd_len < 200:
            logger.info("Supervisor: routing=quick (short input resume=%d jd=%d)", resume_len, jd_len)
            return "quick"

        logger.info("Supervisor: routing=full (resume=%d jd=%d)", resume_len, jd_len)
        return "full"

    def check_budget(self, agent_name: str, guard: BudgetGuard) -> bool:
        """
        Pre-flight budget check. Logs a warning and returns False if blocked.
        """
        allowed, reason = guard.can_run(agent_name)
        if not allowed:
            logger.warning("Supervisor: BLOCKED %s — %s", agent_name, reason)
        return allowed

    def validate_output(self, state: ResumeCoachState, routing: RoutingMode) -> dict:
        """
        Post-run validation. Checks that expected outputs are present.
        Returns a validation summary dict.
        """
        checks = {
            "gap_analysis":   bool(state.get("gap_analysis")),
            "ats_report":     bool(state.get("ats_report")) if routing == "full" else None,
            "coaching_report": bool(state.get("coaching_report")) if routing in ("full", "coach_only") else None,
            "interview_questions": bool(state.get("interview_questions")) if routing == "full" else None,
        }

        missing = [k for k, v in checks.items() if v is False]
        passed  = len(missing) == 0

        if not passed:
            logger.warning("Supervisor: post-run validation failed — missing: %s", missing)
        else:
            logger.info("Supervisor: post-run validation passed")

        return {
            "passed":  passed,
            "checks":  checks,
            "missing": missing,
        }

    def build_supervisor_report(
        self,
        state: ResumeCoachState,
        guard: BudgetGuard,
        routing: RoutingMode,
        validation: dict,
    ) -> dict:
        """Build the final supervisor report attached to state."""
        return {
            "routing_mode":   routing,
            "budget_summary": guard.summary(),
            "validation":     validation,
            "errors":         state.get("errors", []),
            "completed_nodes": state.get("completed_nodes", []),
        }


def make_budget_guard(settings=None) -> BudgetGuard:
    """
    Factory: create a BudgetGuard from settings or environment.
    Used by graph.py to create the guard at pipeline start.
    """
    import os
    provider = os.getenv("LLM_PROVIDER", "replicate")

    return BudgetGuard(
        provider=provider,
        max_tokens=int(os.getenv("BUDGET_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
        max_cost_usd=float(os.getenv("BUDGET_MAX_COST_USD", DEFAULT_MAX_COST_USD)),
        max_time_secs=float(os.getenv("BUDGET_MAX_TIME_SECS", DEFAULT_MAX_TIME_SECS)),
    )