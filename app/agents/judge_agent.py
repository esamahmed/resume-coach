"""
app/agents/judge_agent.py

LLM-as-judge evaluator for the Resume Coach pipeline.
Scores each agent's output on dimensions specific to what that agent controls.

Usage (live mode):
    from app.agents.judge_agent import JudgeAgent
    judge = JudgeAgent()
    result = judge.evaluate("gap_analyzer", input_data, output_data)

Usage (mock mode — for CI/unit tests, no LLM call):
    judge = JudgeAgent(mock=True)
    result = judge.evaluate("gap_analyzer", input_data, output_data)

Score schema per agent:
    gap_analyzer       → accuracy, completeness
    ats_scorer         → ats_alignment
    coach_writer       → completeness, actionability, tone_grounding
    interview_generator → actionability, tone_grounding
    qa_agent           → accuracy
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Pass threshold ────────────────────────────────────────────────────────────
PASS_THRESHOLD = 7.5

# ── Dimension weights per agent ───────────────────────────────────────────────
AGENT_WEIGHTS: dict[str, dict[str, float]] = {
    "gap_analyzer": {
        "accuracy":     0.90,   # hallucinated gaps = hard failure
        "completeness": 0.75,
    },
    "ats_scorer": {
        "ats_alignment": 0.85,
    },
    "coach_writer": {
        "completeness":    0.75,
        "actionability":   0.80,
        "tone_grounding":  0.70,
    },
    "interview_generator": {
        "actionability":  0.80,
        "tone_grounding": 0.70,
    },
    "qa_agent": {
        "accuracy": 0.90,
    },
}

# ── Judge system prompt ───────────────────────────────────────────────────────
JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for an AI resume coaching system.
You will receive:
  1. agent_name  — which agent produced the output
  2. input_data  — what the agent received (resume text, JD text, or prior agent output)
  3. output_data — what the agent produced

Score the output ONLY on the dimensions relevant to this agent (see rules below).
Return ONLY valid JSON — no preamble, no markdown fences, no explanation outside the JSON.

=== SCORING DIMENSIONS ===

accuracy (gap_analyzer, qa_agent only):
  10 = every claimed gap/strength is verifiable from the input resume or JD text
   7 = minor overstatements but no direct contradictions
   0 = ANY claimed gap contradicts explicit text in the resume (CRITICAL FAILURE)
  Rule: if a skill listed in missing_required is visibly present in the resume → score = 0

completeness (gap_analyzer, coach_writer):
  10 = all major JD requirements addressed; no obvious omissions
   7 = 1-2 minor omissions
   0 = more than 3 major JD requirements completely unaddressed

actionability (coach_writer, interview_generator):
  10 = every recommendation/question is specific, concrete, immediately usable
   7 = mostly specific with 1-2 vague items
   0 = generic advice ("improve your skills") with no specifics

ats_alignment (ats_scorer only):
  10 = keyword match scores reflect true semantic overlap; no false positives or negatives
   7 = minor scoring errors, directionally correct
   0 = major false positive (high score for irrelevant resume) or false negative

tone_grounding (coach_writer, interview_generator):
  10 = sounds like a real senior engineer wrote it; no fabricated metrics; grounded in input
   7 = mostly grounded with minor generic phrases
   0 = generic filler ("results-oriented professional") or claims not in resume

=== OUTPUT SCHEMA ===
{
  "agent": "<agent_name>",
  "dimensions": {
    "accuracy":       {"score": <0-10>, "rationale": "<one sentence>"} or null,
    "completeness":   {"score": <0-10>, "rationale": "<one sentence>"} or null,
    "actionability":  {"score": <0-10>, "rationale": "<one sentence>"} or null,
    "ats_alignment":  {"score": <0-10>, "rationale": "<one sentence>"} or null,
    "tone_grounding": {"score": <0-10>, "rationale": "<one sentence>"} or null
  },
  "weighted_score": <float 0-10, you compute this>,
  "pass": <true if weighted_score >= 7.5>,
  "critical_failure": <true if accuracy == 0>
}

=== RULES ===
- Set irrelevant dimensions to null — do NOT score what this agent doesn't control
- weighted_score = sum(score * weight) / sum(weights) for non-null dimensions only
- Weights: accuracy=0.9, completeness=0.75, actionability=0.8, ats_alignment=0.85, tone_grounding=0.7
- critical_failure=true means the run is failed regardless of weighted_score
- Be strict: a 7 means "acceptable but has real issues", a 9 means "near perfect"
- Rationale must reference specific content from the output, not generic statements
"""


@dataclass
class DimensionScore:
    score: float
    rationale: str


@dataclass
class JudgeResult:
    agent: str
    dimensions: dict[str, DimensionScore | None]
    weighted_score: float
    passed: bool
    critical_failure: bool
    raw_response: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "dimensions": {
                k: {"score": v.score, "rationale": v.rationale} if v else None
                for k, v in self.dimensions.items()
            },
            "weighted_score": round(self.weighted_score, 2),
            "pass": self.passed,
            "critical_failure": self.critical_failure,
            "error": self.error,
        }


# ── Mock scores for CI/unit tests ─────────────────────────────────────────────
MOCK_SCORES: dict[str, dict] = {
    "gap_analyzer": {
        "dimensions": {
            "accuracy":     {"score": 8.5, "rationale": "Mock: gaps verified against resume"},
            "completeness": {"score": 8.0, "rationale": "Mock: major JD requirements covered"},
            "actionability": None,
            "ats_alignment": None,
            "tone_grounding": None,
        },
        "weighted_score": 8.27,
        "pass": True,
        "critical_failure": False,
    },
    "ats_scorer": {
        "dimensions": {
            "accuracy": None,
            "completeness": None,
            "actionability": None,
            "ats_alignment": {"score": 7.5, "rationale": "Mock: keyword overlap directionally correct"},
            "tone_grounding": None,
        },
        "weighted_score": 7.5,
        "pass": True,
        "critical_failure": False,
    },
    "coach_writer": {
        "dimensions": {
            "accuracy": None,
            "completeness":   {"score": 8.0, "rationale": "Mock: all gaps addressed"},
            "actionability":  {"score": 7.5, "rationale": "Mock: recommendations are specific"},
            "ats_alignment": None,
            "tone_grounding": {"score": 8.0, "rationale": "Mock: grounded in resume content"},
        },
        "weighted_score": 7.84,
        "pass": True,
        "critical_failure": False,
    },
    "interview_generator": {
        "dimensions": {
            "accuracy": None,
            "completeness": None,
            "actionability":  {"score": 8.5, "rationale": "Mock: questions are specific and targeted"},
            "ats_alignment": None,
            "tone_grounding": {"score": 8.0, "rationale": "Mock: questions grounded in actual gaps"},
        },
        "weighted_score": 8.27,
        "pass": True,
        "critical_failure": False,
    },
}


class JudgeAgent:
    """
    Evaluates agent outputs using an LLM-as-judge pattern.

    Args:
        mock: If True, returns deterministic mock scores (for CI/unit tests).
              No LLM call is made in mock mode.
        model: Anthropic model to use as judge. Defaults to claude-sonnet-4-6.
               Using a separate model from the pipeline agents ensures
               independent evaluation.
    """

    def __init__(
        self,
        mock: bool = False,
        model: str = "claude-sonnet-4-6-20260217",
    ):
        self.mock = mock
        self.model = model

    def evaluate(
        self,
        agent_name: str,
        input_data: Any,
        output_data: Any,
    ) -> JudgeResult:
        """
        Score a single agent's output.

        Args:
            agent_name: One of gap_analyzer, ats_scorer, coach_writer,
                        interview_generator, qa_agent
            input_data: What the agent received (dict or str)
            output_data: What the agent produced (dict or str)

        Returns:
            JudgeResult with scores, rationales, pass/fail, and critical_failure flag
        """
        if self.mock:
            return self._mock_result(agent_name)

        if agent_name not in AGENT_WEIGHTS:
            logger.warning("JudgeAgent: unknown agent '%s' — skipping", agent_name)
            return JudgeResult(
                agent=agent_name,
                dimensions={},
                weighted_score=0.0,
                passed=False,
                critical_failure=False,
                error=f"Unknown agent: {agent_name}",
            )

        user_message = self._build_user_message(agent_name, input_data, output_data)

        try:
            raw = self._call_judge(user_message)
            return self._parse_response(agent_name, raw)
        except Exception as e:
            logger.error("JudgeAgent evaluation failed for %s: %s", agent_name, e)
            return JudgeResult(
                agent=agent_name,
                dimensions={},
                weighted_score=0.0,
                passed=False,
                critical_failure=False,
                error=str(e),
            )

    def evaluate_pipeline(
        self,
        pipeline_output: dict,
        input_resume: str,
        input_jd: str,
    ) -> dict[str, JudgeResult]:
        """
        Evaluate all agents in a completed pipeline run.

        Args:
            pipeline_output: Full /analyze response JSON
            input_resume: Raw resume text
            input_jd: Raw job description text

        Returns:
            Dict of agent_name → JudgeResult
        """
        results = {}
        base_input = {"resume": input_resume[:2000], "jd": input_jd[:2000]}

        agent_outputs = {
            "gap_analyzer":        pipeline_output.get("gap_analysis", {}),
            "ats_scorer":          pipeline_output.get("ats_report", {}),
            "coach_writer":        pipeline_output.get("coaching_report", {}),
            "interview_generator": pipeline_output.get("interview_questions", []),
        }

        for agent_name, output in agent_outputs.items():
            if not output:
                logger.warning("JudgeAgent: no output found for %s — skipping", agent_name)
                continue
            logger.info("JudgeAgent: evaluating %s", agent_name)
            results[agent_name] = self.evaluate(agent_name, base_input, output)

        return results

    def _build_user_message(
        self,
        agent_name: str,
        input_data: Any,
        output_data: Any,
    ) -> str:
        input_str = (
            json.dumps(input_data, indent=2)
            if isinstance(input_data, dict)
            else str(input_data)
        )
        output_str = (
            json.dumps(output_data, indent=2)
            if isinstance(output_data, (dict, list))
            else str(output_data)
        )

        # Truncate to avoid token overflow — judge needs context, not full text
        input_str = input_str[:3000]
        output_str = output_str[:4000]

        return (
            f"agent_name: {agent_name}\n\n"
            f"=== INPUT ===\n{input_str}\n\n"
            f"=== OUTPUT ===\n{output_str}"
        )

    def _call_judge(self, user_message: str) -> str:
        """Use pipeline LLM as judge temporarily until Anthropic billing is set up."""
        from app.core.llm import get_llm
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = get_llm()
        messages = [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=user_message)
        ]
        response = llm.invoke(messages)
        return response.content

    def _parse_response(self, agent_name: str, raw: str) -> JudgeResult:
        """Parse the judge's JSON response into a JudgeResult."""
        # Strip markdown fences if present
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()

        try:
            data = json.loads(clean)
        except json.JSONDecodeError as e:
            logger.error("JudgeAgent: failed to parse JSON — %s\nRaw: %s", e, raw[:500])
            return JudgeResult(
                agent=agent_name,
                dimensions={},
                weighted_score=0.0,
                passed=False,
                critical_failure=False,
                raw_response=raw,
                error=f"JSON parse error: {e}",
            )

        # Parse dimensions
        dimensions: dict[str, DimensionScore | None] = {}
        all_dims = ["accuracy", "completeness", "actionability", "ats_alignment", "tone_grounding"]

        for dim in all_dims:
            raw_dim = data.get("dimensions", {}).get(dim)
            if raw_dim is None:
                dimensions[dim] = None
            else:
                dimensions[dim] = DimensionScore(
                    score=float(raw_dim.get("score", 0)),
                    rationale=raw_dim.get("rationale", ""),
                )

        # Recompute weighted score locally for safety
        weights = AGENT_WEIGHTS.get(agent_name, {})
        total_weight = 0.0
        weighted_sum = 0.0
        for dim, weight in weights.items():
            if dimensions.get(dim) is not None:
                weighted_sum += dimensions[dim].score * weight
                total_weight += weight

        weighted_score = (weighted_sum / total_weight) if total_weight > 0 else 0.0
        critical_failure = (
            dimensions.get("accuracy") is not None
            and dimensions["accuracy"].score == 0
        )
        passed = weighted_score >= PASS_THRESHOLD and not critical_failure

        return JudgeResult(
            agent=agent_name,
            dimensions=dimensions,
            weighted_score=weighted_score,
            passed=passed,
            critical_failure=critical_failure,
            raw_response=raw,
        )

    def _mock_result(self, agent_name: str) -> JudgeResult:
        """Return deterministic mock result for testing."""
        mock = MOCK_SCORES.get(agent_name, {
            "dimensions": {},
            "weighted_score": 8.0,
            "pass": True,
            "critical_failure": False,
        })

        dimensions = {}
        for dim, val in mock.get("dimensions", {}).items():
            dimensions[dim] = (
                DimensionScore(score=val["score"], rationale=val["rationale"])
                if val else None
            )

        return JudgeResult(
            agent=agent_name,
            dimensions=dimensions,
            weighted_score=mock["weighted_score"],
            passed=mock["pass"],
            critical_failure=mock["critical_failure"],
        )