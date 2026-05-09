"""
app/graph.py

LangGraph pipeline — the ResumeCoach StateGraph.

Topology:
    [ml_fast_pass] → [gap_analyzer] → [coach_writer]
                                    ↘ [ats_scorer]
                                    ↘ [interview_generator]

    All three downstream nodes run in parallel after gap_analyzer.
    Q&A runs on-demand (separate invocation with the same state + FAISS index).

Key design decision: LangGraph (not LangChain SequentialChain).
Every node reads/writes to shared ResumeCoachState — the Q&A agent
always has live access to all prior findings without context loss.
"""
from __future__ import annotations
import logging
import uuid
from typing import TYPE_CHECKING

from langgraph.graph import END, StateGraph

from app.core.state import ResumeCoachState
from app.agents import gap_analyzer, coach_writer, ats_scorer, interview_generator
from app.ml.ml_scorer import compute_ats_score, compute_hire_probability
from app.observability.langfuse_tracer import get_tracer

if TYPE_CHECKING:
    from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ML fast-pass node (runs before any LLM call)
# ---------------------------------------------------------------------------

def ml_fast_pass(state: ResumeCoachState) -> ResumeCoachState:
    """
    Classical ML pre-processing: TF-IDF ATS score + XGBoost hire probability.

    Runs synchronously before the LLM agents so that:
      - Gap analyzer gets keyword signals injected into its prompt
      - ATS scorer gets pre-computed scores without redundant LLM calls
      - We surface load-bearing signals cheaply before expensive tokens run
    """
    logger.info("Running ML fast pass")
    resume = state["resume_text"]
    jd = state["jd_text"]

    ats_result = compute_ats_score(resume, jd)
    hire_prob = compute_hire_probability(resume, jd)

    return {
        **state,
        "tfidf_ats_score": ats_result["ats_score"],
        "xgb_hire_prob": hire_prob,
        "matched_keywords": ats_result["matched_keywords"],
        "missing_keywords": ats_result["missing_keywords"],
        "completed_nodes": ["ml_fast_pass"],
    }


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(ResumeCoachState)

    graph.add_node("ml_fast_pass", ml_fast_pass)
    graph.add_node("gap_analyzer", gap_analyzer.run)
    graph.add_node("coach_writer", coach_writer.run)
    graph.add_node("ats_scorer", ats_scorer.run)
    graph.add_node("interview_generator", interview_generator.run)

    graph.set_entry_point("ml_fast_pass")

    graph.add_edge("ml_fast_pass", "gap_analyzer")
    graph.add_edge("gap_analyzer", "coach_writer")
    graph.add_edge("gap_analyzer", "ats_scorer")
    graph.add_edge("gap_analyzer", "interview_generator")

    graph.add_edge("coach_writer", END)
    graph.add_edge("ats_scorer", END)
    graph.add_edge("interview_generator", END)

    return graph


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph().compile()
    return _compiled_graph


# ---------------------------------------------------------------------------
# Main entry point for the pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    resume_text: str,
    jd_text: str,
    session_id: str | None = None,
) -> tuple[ResumeCoachState, "FAISS | None"]:
    """
    Run the full analysis pipeline and return final state + FAISS index.

    The FAISS index is built AFTER the pipeline so it can index
    gap findings and coaching report in addition to raw text.
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    tracer = get_tracer()
    trace_url = tracer.get_trace_url(session_id)

    initial_state: ResumeCoachState = {
        "resume_text": resume_text,
        "jd_text": jd_text,
        "resume_chunks": [],
        "jd_chunks": [],
        "tfidf_ats_score": 0.0,
        "xgb_hire_prob": 0.0,
        "matched_keywords": [],
        "missing_keywords": [],
        "gap_analysis": {},
        "coaching_report": {},
        "ats_report": {},
        "interview_questions": [],
        "chat_history": [],
        "current_question": "",
        "current_answer": "",
        "session_id": session_id,
        "trace_url": trace_url,
        "errors": [],
        "completed_nodes": [],
    }

    logger.info("Starting pipeline — session_id=%s", session_id)
    graph = get_compiled_graph()
    final_state = graph.invoke(initial_state)
    logger.info(
        "Pipeline complete — completed_nodes=%s errors=%s",
        final_state.get("completed_nodes"),
        final_state.get("errors"),
    )

    # Build FAISS index after pipeline — can now include gap findings
    faiss_index = None
    try:
        from app.rag.vector_store import build_session_index
        faiss_index = build_session_index(
            resume_text=resume_text,
            jd_text=jd_text,
            gap_analysis=final_state.get("gap_analysis"),
            coaching_report=final_state.get("coaching_report"),
        )
        logger.info("FAISS session index built for Q&A")
    except Exception as e:
        logger.warning("FAISS index build failed (Q&A will use state-only): %s", e)

    tracer.flush()
    return final_state, faiss_index
