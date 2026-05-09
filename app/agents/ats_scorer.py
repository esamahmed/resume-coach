"""
app/agents/ats_scorer.py

LangGraph Node 3: ATS Scorer

Combines:
  - Classical ML scores already in state (TF-IDF + XGBoost)
  - LLM-based qualitative ATS assessment
  - Produces a composite ATS report

Reads ml scores from shared state — no re-computation needed.
"""
from __future__ import annotations
import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm import get_llm
from app.core.state import ResumeCoachState
from app.ml.ml_scorer import composite_score
from app.observability.langfuse_tracer import get_tracer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an ATS (Applicant Tracking System) expert who understands how \
enterprise recruiting software parses and scores resumes.

Given a resume, job description, and keyword analysis, produce a detailed \
ATS optimization report.

Respond with ONLY valid JSON — no prose, no markdown:

{
  "ats_pass_likelihood": "<HIGH|MEDIUM|LOW>",
  "ats_pass_likelihood_pct": <int 0-100>,
  "keyword_density_score": <int 0-100>,
  "format_score": <int 0-100>,
  "section_presence": {
    "contact_info": <bool>,
    "summary": <bool>,
    "skills": <bool>,
    "experience": <bool>,
    "education": <bool>,
    "certifications": <bool>
  },
  "top_missing_keywords": ["<keyword>"],
  "keyword_stuffing_risk": <bool>,
  "recommended_format_changes": ["<change>"],
  "ats_friendly_title_suggestion": "<job title string>"
}
"""

USER_PROMPT = """\
RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

CLASSICAL ML ATS SCORE (TF-IDF cosine similarity): {tfidf_score:.1%}
HIRE PROBABILITY (XGBoost): {xgb_score:.1%}
COMPOSITE SCORE: {composite:.1%}

MATCHED KEYWORDS: {matched}
MISSING KEYWORDS: {missing}

Produce the ATS report now. Respond with JSON only.
"""


def run(state: ResumeCoachState) -> ResumeCoachState:
    tracer = get_tracer()
    session_id = state.get("session_id", "")
    tfidf = state.get("tfidf_ats_score", 0.0)
    xgb = state.get("xgb_hire_prob", 0.0)
    comp = composite_score(tfidf, xgb)

    with tracer.span("ats_scorer", session_id=session_id,
                     input_data={"tfidf_score": tfidf, "xgb_score": xgb}) as span:
        try:
            llm = get_llm()
            prompt = USER_PROMPT.format(
                resume_text=state["resume_text"][:4000],
                jd_text=state["jd_text"][:2000],
                tfidf_score=tfidf,
                xgb_score=xgb,
                composite=comp,
                matched=", ".join(state.get("matched_keywords", [])[:15]),
                missing=", ".join(state.get("missing_keywords", [])[:15]),
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

            llm_ats = json.loads(raw)

            ats_report = {
                **llm_ats,
                "tfidf_ats_score": tfidf,
                "xgb_hire_probability": xgb,
                "composite_score": comp,
                "matched_keywords": state.get("matched_keywords", []),
                "missing_keywords": state.get("missing_keywords", []),
            }

            span.update(output={"composite_score": comp,
                                "ats_pass": llm_ats.get("ats_pass_likelihood")})
            logger.info(
                "ATS scorer complete — composite=%.3f pass=%s",
                comp, llm_ats.get("ats_pass_likelihood"),
            )

            return {
                **state,
                "ats_report": ats_report,
                "completed_nodes": ["ats_scorer"],
            }

        except Exception as e:
            logger.error("ats_scorer failed: %s", e)
            return {
                **state,
                "ats_report": {
                    "error": str(e),
                    "composite_score": comp,
                    "tfidf_ats_score": tfidf,
                    "xgb_hire_probability": xgb,
                },
                "errors": [f"ats_scorer: {e}"],
                "completed_nodes": ["ats_scorer"],
            }
