"""
app/agents/coach_writer.py

LangGraph Node 2: Coach Writer

Reads gap_analysis from shared state (never from a string serialization)
and produces a structured coaching report with actionable bullet rewrites.

IMPORTANT: Returns ONLY fields this agent owns — never spreads full state.
           Spreading causes parallel branch race conditions in LangGraph.
"""
from __future__ import annotations
import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm import get_llm
from app.core.state import ResumeCoachState
from app.observability.langfuse_tracer import get_tracer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a world-class resume coach and career strategist specializing in \
software engineering and AI/ML roles.

Given a gap analysis, produce a structured coaching report that gives the \
candidate clear, actionable steps to improve their resume for this specific role.

You MUST respond with ONLY valid JSON — no prose, no markdown, no preamble:

{
  "executive_summary": "<3 sentences: fit assessment + top priority action>",
  "section_scores": {
    "skills": <int 0-100>,
    "experience": <int 0-100>,
    "education": <int 0-100>,
    "impact_statements": <int 0-100>,
    "ats_optimization": <int 0-100>
  },
  "recommendations": [
    {
      "priority": <1-10, 10=most urgent>,
      "section": "<resume section>",
      "action": "<specific action verb>",
      "detail": "<concrete instruction>",
      "example": "<optional example text>"
    }
  ],
  "rewritten_bullets": [
    {
      "original": "<original resume bullet>",
      "rewritten": "<improved bullet with metrics and keywords>",
      "why": "<brief explanation of what changed>"
    }
  ],
  "missing_keywords_to_add": ["<keyword>"],
  "cover_letter_angles": ["<angle>"]
}
"""

USER_PROMPT = """\
GAP ANALYSIS FINDINGS:
{gap_analysis}

CRITICAL GAPS IDENTIFIED: {critical_gaps}
OVERALL FIT SCORE: {fit_score}/100

RESUME EXCERPT (for bullet rewriting):
{resume_excerpt}

JD KEY REQUIREMENTS:
{jd_excerpt}

Produce the coaching report now. Respond with JSON only.
"""


def run(state: ResumeCoachState) -> dict:
    tracer = get_tracer()
    session_id = state.get("session_id", "")
    gap = state.get("gap_analysis", {})

    with tracer.span("coach_writer", session_id=session_id,
                     input_data={"fit_score": gap.get("overall_fit_score")}) as span:
        try:
            llm = get_llm()
            prompt = USER_PROMPT.format(
                gap_analysis=json.dumps(gap, indent=2)[:3000],
                critical_gaps=", ".join(gap.get("critical_gaps", [])[:5]),
                fit_score=gap.get("overall_fit_score", "N/A"),
                resume_excerpt=state["resume_text"][:3000],
                jd_excerpt=state["jd_text"][:2000],
            )

            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]

            response = llm.invoke(messages)
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            coaching_report = json.loads(raw)
            recs = len(coaching_report.get("recommendations", []))
            span.update(output={"recommendations_count": recs})
            logger.info("Coach writer complete — %d recommendations", recs)

            # ── Return ONLY owned fields — no **state spread ──────────────
            return {
                "coaching_report": coaching_report,
                "completed_nodes": ["coach_writer"],
            }

        except Exception as e:
            logger.error("coach_writer failed: %s", e)
            span.update(metadata={"error": str(e)})
            return {
                "coaching_report": {"error": str(e)},
                "errors": [f"coach_writer: {e}"],
                "completed_nodes": ["coach_writer"],
            }
