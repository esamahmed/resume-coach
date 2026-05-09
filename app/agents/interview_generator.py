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
import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm import get_llm
from app.core.state import ResumeCoachState
from app.observability.langfuse_tracer import get_tracer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert technical interviewer for software engineering and AI/ML roles.

Generate targeted interview questions based on the candidate's identified gaps \
and strengths. Questions should prepare the candidate for what they're most \
likely to be asked given their specific profile vs. the role requirements.

Respond with ONLY valid JSON — no prose, no markdown:

{
  "questions": [
    {
      "question": "<interview question>",
      "type": "<TECHNICAL|BEHAVIORAL|SITUATIONAL|SYSTEM_DESIGN>",
      "difficulty": "<EASY|MEDIUM|HARD>",
      "gap_it_targets": "<which gap this question probes>",
      "why_they_will_ask": "<brief reason>",
      "sample_answer_hint": "<key points to hit in a strong answer>",
      "follow_up": "<likely follow-up question>"
    }
  ],
  "preparation_priorities": ["<ordered list of what to study first>"],
  "red_flags_to_address": ["<things the interviewer will likely probe given gaps>"]
}
"""

USER_PROMPT = """\
CANDIDATE PROFILE SUMMARY:
Overall fit score: {fit_score}/100
Critical gaps: {critical_gaps}
Key strengths: {strengths}

JOB ROLE: {jd_excerpt}

Generate 8 targeted interview questions. Mix technical, behavioral, and \
system design. Weight towards the identified critical gaps.
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
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            result = json.loads(raw)
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
