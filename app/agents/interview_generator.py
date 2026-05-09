"""
app/agents/interview_generator.py

LangGraph Node 4: Interview Question Generator

Gap-aware: generates questions specifically targeting the candidate's
identified weaknesses, not generic interview questions.

Reads gap_analysis from shared state.

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
You are a technical interviewer. Return ONLY this JSON — no prose, no markdown:

{
  "questions": [
    {"question": "How have you used LangGraph in production?", "type": "TECHNICAL", "difficulty": "MEDIUM", "gap_it_targets": "LangGraph experience"},
    {"question": "Describe a time you improved ML model performance.", "type": "BEHAVIORAL", "difficulty": "MEDIUM", "gap_it_targets": "MLOps"},
    {"question": "How would you design a RAG pipeline?", "type": "SYSTEM_DESIGN", "difficulty": "HARD", "gap_it_targets": "RAG systems"}
  ],
  "preparation_priorities": ["Topic A", "Topic B", "Topic C"]
}

Rules: exactly 3 questions with short strings, 3 preparation priorities.
"""

USER_PROMPT = """\
CANDIDATE PROFILE SUMMARY:
Overall fit score: {fit_score}/100
Critical gaps: {critical_gaps}
Key strengths: {strengths}

JOB ROLE: {jd_excerpt}

Generate 3 targeted interview questions. Mix technical and behavioral. \
Weight towards the identified critical gaps. Be concise.
Respond with JSON only.
"""


def run(state: ResumeCoachState) -> dict:
    tracer = get_tracer()
    session_id = state.get("session_id", "")
    gap = state.get("gap_analysis", {})

    with tracer.span("interview_generator", session_id=session_id,
                     input_data={"critical_gaps": gap.get("critical_gaps", [])}) as span:
        try:
            llm = get_llm()
            prompt = USER_PROMPT.format(
                fit_score=gap.get("overall_fit_score", "N/A"),
                critical_gaps=", ".join(gap.get("critical_gaps", [])[:5]),
                strengths=", ".join(gap.get("strengths", [])[:5]),
                jd_excerpt=state["jd_text"][:1500],
            )

            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]

            response = llm.invoke(messages)
            result = parse_llm_json(response.content)
            questions = result.get("questions", [])

            span.update(output={"questions_count": len(questions)})
            logger.info("Interview generator complete — %d questions", len(questions))

            # ── Return ONLY owned fields — no **state spread ──────────────
            return {
                "interview_questions": questions,
                "completed_nodes": ["interview_generator"],
            }

        except Exception as e:
            logger.error("interview_generator failed: %s", e)
            return {
                "interview_questions": [],
                "errors": [f"interview_generator: {e}"],
                "completed_nodes": ["interview_generator"],
            }
