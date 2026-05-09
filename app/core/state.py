"""
app/core/state.py

ResumeCoachState — the single shared object that flows through every
LangGraph node. Every agent reads from AND writes to this state, so
the Q&A agent always has live access to gap findings, scores, etc.

This is the architectural fix for the SequentialChain context-loss problem:
instead of serialized string hand-offs between chains, all agents share
the same typed state object via LangGraph's StateGraph.
"""
from __future__ import annotations
from typing import Annotated, Any
from typing_extensions import TypedDict
import operator


def _last(a: Any, b: Any) -> Any:
    """Reducer: last writer wins (used for scalar fields)."""
    return b


def _extend(a: list, b: list) -> list:
    """Reducer: merge lists (used for messages/history)."""
    return a + b


class ResumeCoachState(TypedDict):
    # ── Raw inputs ──
    resume_text: Annotated[str, _last]           # full extracted resume text
    jd_text: Annotated[str, _last]               # full job description text
    resume_chunks: Annotated[list[str], _last]   # chunked for FAISS embedding
    jd_chunks: Annotated[list[str], _last]

    # ── Classical ML signals (fast pass — runs before LLM) ──
    tfidf_ats_score: Annotated[float, _last]     # 0.0–1.0 cosine similarity
    xgb_hire_prob: Annotated[float, _last]       # 0.0–1.0 hire probability
    matched_keywords: Annotated[list[str], _last]
    missing_keywords: Annotated[list[str], _last]

    # ── Gap Analyzer agent output ──
    gap_analysis: Annotated[dict, _last]         # structured JSON gap report
    # {skills_gap, experience_gap, education_gap, strengths, critical_gaps}

    # ── Coach Writer agent output ──
    coaching_report: Annotated[dict, _last]      # structured coaching report
    # {executive_summary, section_scores, recommendations, rewritten_bullets}

    # ── ATS Scorer agent output ──
    ats_report: Annotated[dict, _last]
    # {ats_score, ml_score, composite_score, keyword_analysis, pass_likelihood}

    # ── Interview Generator agent output ──
    interview_questions: Annotated[list[dict], _last]
    # [{question, type, gap_it_targets, sample_answer_hint}]

    # ── Conversational Q&A (accumulates across turns) ──
    chat_history: Annotated[list[dict], _extend]  # [{role, content}]
    current_question: Annotated[str, _last]
    current_answer: Annotated[str, _last]

    # ── Pipeline metadata ──
    session_id: Annotated[str, _last]
    trace_url: Annotated[str, _last]             # Langfuse trace URL
    errors: Annotated[list[str], _extend]
    completed_nodes: Annotated[list[str], _extend]
