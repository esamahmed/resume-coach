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
from app.utils.json_utils import parse_llm_json
from app.core.state import ResumeCoachState
from app.observability.langfuse_tracer import get_tracer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a resume coach. Given a gap analysis, return ONLY this JSON — \
no prose, no markdown, no extra fields:

{
  "executive_summary": "One sentence assessment.",
  "recommendations": [
    {"priority": 1, "action": "Add LangGraph to skills", "detail": "One sentence instruction."},
    {"priority": 2, "action": "Quantify ML impact", "detail": "One sentence instruction."},
    {"priority": 3, "action": "Add cloud certifications", "detail": "One sentence instruction."}
  ],
  "keywords_to_add": ["keyword1", "keyword2", "keyword3"]
}

Rules: exactly those 3 keys, recommendations has 2-3 items with short strings, \
executive_summary is one sentence.
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
                resume_excerpt=state["resume_text"][:1500],
                jd_excerpt=state["jd_text"][:1000],
            )

            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]

            response = llm.invoke(messages)
            coaching_report = parse_llm_json(response.content)
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
