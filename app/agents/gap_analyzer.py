"""
app/agents/gap_analyzer.py

LangGraph Node 1: Gap Analyzer

Compares resume against JD and produces a structured gap analysis.
Output is stored in state["gap_analysis"] — downstream agents
(coach writer, ATS scorer, Q&A) read this directly from shared state.

Enforces JSON schema output — per capstone rubric (25% prompt engineering).

IMPORTANT: Returns ONLY fields this agent owns — never spreads full state.
           Spreading causes parallel branch race conditions in LangGraph.
"""
from __future__ import annotations
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm import get_llm
from app.utils.json_utils import parse_llm_json
from app.core.state import ResumeCoachState
from app.observability.langfuse_tracer import get_tracer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert technical recruiter. Analyse the resume vs job description \
and respond with ONLY this JSON — no prose, no markdown, no extra fields:

{
  "overall_fit_score": 72,
  "fit_summary": "One sentence max.",
  "missing_required_skills": ["Skill A", "Skill B", "Skill C"],
  "strengths": ["Strength A", "Strength B", "Strength C"],
  "critical_gaps": ["Gap A", "Gap B"]
}

Rules: exactly those 5 keys, lists have 2-3 short items each (combine related \
skills into one item), fit_summary is one sentence.
"""

USER_PROMPT = """\
RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

ATS keyword match score (from classical ML): {ats_score:.1%}
Missing JD keywords: {missing_keywords}

Perform the gap analysis now. Respond with JSON only.
"""


def run(state: ResumeCoachState) -> dict:
    tracer = get_tracer()
    session_id = state.get("session_id", "")

    with tracer.span("gap_analyzer", session_id=session_id,
                     input_data={"resume_len": len(state.get("resume_text", ""))}) as span:
        try:
            llm = get_llm()
            prompt = USER_PROMPT.format(
                resume_text=state["resume_text"][:3500],
                jd_text=state["jd_text"][:2000],
                ats_score=state.get("tfidf_ats_score", 0.0),
                missing_keywords=", ".join(state.get("missing_keywords", [])[:20]),
            )

            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]

            response = llm.invoke(messages)
            gap_analysis = parse_llm_json(response.content)
            span.update(output={"fit_score": gap_analysis.get("overall_fit_score")})
            logger.info("Gap analysis complete — fit_score=%s", gap_analysis.get("overall_fit_score"))

            # ── Return ONLY owned fields — no **state spread ──────────────
            return {
                "gap_analysis": gap_analysis,
                "completed_nodes": ["gap_analyzer"],
            }

        except Exception as e:
            logger.error("gap_analyzer failed: %s", e)
            span.update(metadata={"error": str(e)})
            return {
                "gap_analysis": {"error": str(e), "overall_fit_score": 0},
                "errors": [f"gap_analyzer: {e}"],
                "completed_nodes": ["gap_analyzer"],
            }
