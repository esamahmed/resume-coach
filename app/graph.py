"""
app/graph.py

LangGraph pipeline for Resume Coach — with supervisor orchestration.

Graph structure:
  supervisor_entry
       ↓
  ml_fast_pass
       ↓
  gap_analyzer
       ↓ (parallel)
  ┌────────────────────────────────┐
  coach_writer  ats_scorer  interview_generator
  └────────────────────────────────┘
       ↓
  supervisor_exit  (validation + budget summary)
       ↓
  END

Supervisor responsibilities:
  - Pre-flight routing decision (full vs quick)
  - Budget guard passed through state
  - Post-run validation and summary report
"""
from __future__ import annotations

import logging
import time
import uuid

from langgraph.graph import END, StateGraph

from app.agents import (
    ats_scorer,
    coach_writer,
    gap_analyzer,
    interview_generator,
)
from app.agents.supervisor import SupervisorAgent, make_budget_guard
from app.core.config import get_settings
from app.core.state import ResumeCoachState
from app.ml.ml_scorer import compute_ats_score, compute_hire_probability
from app.observability.langfuse_tracer import get_tracer

logger = logging.getLogger(__name__)


# ── Supervisor nodes ──────────────────────────────────────────────────────────

def supervisor_entry(state: ResumeCoachState) -> dict:
    """
    Entry node: initialise budget guard, decide routing, log decision.
    Attaches supervisor_routing and budget_guard_state to state.
    """
    settings = get_settings()
    supervisor = SupervisorAgent(provider=settings.llm_provider)
    guard      = make_budget_guard()
    routing    = supervisor.decide_routing(state)

    logger.info(
        "Supervisor entry — routing=%s budget_tokens=%d budget_cost=$%.2f",
        routing, guard.max_tokens, guard.max_cost_usd,
    )

    return {
        "supervisor_routing":   routing,
        "supervisor_report":    {},
        "budget_elapsed_tokens": 0,
        "budget_elapsed_cost":   0.0,
        "budget_start_time":     time.time(),
        "completed_nodes":       ["supervisor_entry"],
    }


def supervisor_exit(state: ResumeCoachState) -> dict:
    """
    Exit node: validate outputs, build budget summary, log final report.
    """
    settings   = get_settings()
    supervisor = SupervisorAgent(provider=settings.llm_provider)
    guard      = make_budget_guard()

    # Reconstruct elapsed values from state
    guard.elapsed_tokens = state.get("budget_elapsed_tokens", 0)
    guard.elapsed_cost   = state.get("budget_elapsed_cost", 0.0)
    guard.start_time     = state.get("budget_start_time", time.time())

    routing    = state.get("supervisor_routing", "full")
    validation = supervisor.validate_output(state, routing)
    report     = supervisor.build_supervisor_report(state, guard, routing, validation)

    logger.info(
        "Supervisor exit — routing=%s tokens=%d cost=$%.4f elapsed=%.1fs validation=%s",
        routing,
        guard.elapsed_tokens,
        guard.elapsed_cost,
        time.time() - guard.start_time,
        "PASS" if validation["passed"] else "FAIL",
    )

    if not validation["passed"]:
        logger.warning("Supervisor: missing outputs — %s", validation["missing"])

    return {
        "supervisor_report":  report,
        "completed_nodes":    ["supervisor_exit"],
    }


# ── Budget-wrapped agent nodes ────────────────────────────────────────────────

def _make_budget_node(agent_name: str, agent_run_fn):
    """
    Wraps an agent run function with pre-flight budget check.
    If budget is exhausted, skips the agent and logs a warning.
    """
    def node(state: ResumeCoachState) -> dict:
        guard = make_budget_guard()
        guard.elapsed_tokens = state.get("budget_elapsed_tokens", 0)
        guard.elapsed_cost   = state.get("budget_elapsed_cost", 0.0)
        guard.start_time     = state.get("budget_start_time", time.time())

        allowed, reason = guard.can_run(agent_name)
        if not allowed:
            logger.warning("Budget blocked %s: %s", agent_name, reason)
            return {
                "errors":          [f"{agent_name}: budget_blocked — {reason}"],
                "completed_nodes": [agent_name],
            }

        # Run the agent
        result = agent_run_fn(state)

        # Update budget in state after run
        guard.consume(agent_name)
        result["budget_elapsed_tokens"] = guard.elapsed_tokens
        result["budget_elapsed_cost"]   = guard.elapsed_cost

        return result

    node.__name__ = agent_name
    return node


# ── ML fast pass (no LLM — budget exempt) ────────────────────────────────────

def ml_fast_pass(state: ResumeCoachState) -> dict:
    """
    Runs synchronously before LLM agents.
    No LLM call — exempt from budget tracking.
    """
    logger.info("Running ML fast pass")
    resume = state["resume_text"]
    jd     = state["jd_text"]

    ats_result = compute_ats_score(resume, jd)
    hire_prob  = compute_hire_probability(resume, jd)

    return {
        "tfidf_ats_score":  ats_result["ats_score"],
        "xgb_hire_prob":    hire_prob,
        "matched_keywords": ats_result["matched_keywords"],
        "missing_keywords": ats_result["missing_keywords"],
        "completed_nodes":  ["ml_fast_pass"],
    }


# ── Routing function ──────────────────────────────────────────────────────────

def route_after_gap(state: ResumeCoachState) -> list[str]:
    """
    Conditional edge: determines which parallel agents run after gap_analyzer.
    Quick mode skips coach_writer and interview_generator.
    """
    routing = state.get("supervisor_routing", "full")

    if routing == "quick":
        logger.info("Supervisor: quick mode — running ats_scorer only")
        return ["ats_scorer"]

    if routing == "coach_only":
        logger.info("Supervisor: coach_only mode — running coach_writer only")
        return ["coach_writer"]

    logger.info("Supervisor: full mode — running all parallel agents")
    return ["coach_writer", "ats_scorer", "interview_generator"]


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(ResumeCoachState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    graph.add_node("supervisor_entry",      supervisor_entry)
    graph.add_node("ml_fast_pass",          ml_fast_pass)
    graph.add_node("gap_analyzer",          _make_budget_node("gap_analyzer",          gap_analyzer.run))
    graph.add_node("ats_scorer",            _make_budget_node("ats_scorer",            ats_scorer.run))
    graph.add_node("coach_writer",          _make_budget_node("coach_writer",          coach_writer.run))
    graph.add_node("interview_generator",   _make_budget_node("interview_generator",   interview_generator.run))
    graph.add_node("supervisor_exit",       supervisor_exit)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("supervisor_entry")

    # ── Sequential edges ──────────────────────────────────────────────────────
    graph.add_edge("supervisor_entry", "ml_fast_pass")
    graph.add_edge("ml_fast_pass",     "gap_analyzer")

    # ── Conditional parallel fan-out after gap_analyzer ───────────────────────
    graph.add_conditional_edges(
        "gap_analyzer",
        route_after_gap,
        {
            "coach_writer":        "coach_writer",
            "ats_scorer":          "ats_scorer",
            "interview_generator": "interview_generator",
        },
    )

    # ── Fan-in: all parallel branches → supervisor_exit ───────────────────────
    graph.add_edge("coach_writer",        "supervisor_exit")
    graph.add_edge("ats_scorer",          "supervisor_exit")
    graph.add_edge("interview_generator", "supervisor_exit")

    # ── Final edge ────────────────────────────────────────────────────────────
    graph.add_edge("supervisor_exit", END)

    return graph


# ── Compiled graph singleton ──────────────────────────────────────────────────

_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph().compile()
    return _compiled_graph


# ── Main pipeline entry point ─────────────────────────────────────────────────

def run_pipeline(
    resume_text: str,
    jd_text: str,
    session_id: str | None = None,
) -> tuple[ResumeCoachState, "FAISS | None"]:
    """
    Run the full analysis pipeline and return final state + FAISS index.
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    tracer    = get_tracer()
    trace_url = tracer.get_trace_url(session_id)

    initial_state: ResumeCoachState = {
        "resume_text":      resume_text,
        "jd_text":          jd_text,
        "resume_chunks":    [],
        "jd_chunks":        [],
        "tfidf_ats_score":  0.0,
        "xgb_hire_prob":    0.0,
        "matched_keywords": [],
        "missing_keywords": [],
        "gap_analysis":     {},
        "coaching_report":  {},
        "ats_report":       {},
        "interview_questions": [],
        "chat_history":     [],
        "current_question": "",
        "current_answer":   "",
        "session_id":       session_id,
        "trace_url":        trace_url,
        "errors":           [],
        "completed_nodes":  [],
        # Supervisor fields
        "supervisor_routing": "full",
        "supervisor_report":  {},
        "budget_elapsed_tokens": 0,
        "budget_elapsed_cost":   0.0,
        "budget_start_time":     time.time(),
    }

    logger.info("Starting pipeline — session_id=%s", session_id)
    graph       = get_compiled_graph()
    final_state = graph.invoke(initial_state)

    logger.info(
        "Pipeline complete — completed_nodes=%s errors=%s",
        final_state.get("completed_nodes"),
        final_state.get("errors"),
    )

    # Log supervisor report
    report = final_state.get("supervisor_report", {})
    if report:
        budget = report.get("budget_summary", {})
        logger.info(
            "Supervisor report — routing=%s tokens=%d cost=$%.4f elapsed=%.1fs validation=%s",
            report.get("routing_mode"),
            budget.get("elapsed_tokens", 0),
            budget.get("elapsed_cost_usd", 0),
            budget.get("elapsed_secs", 0),
            "PASS" if report.get("validation", {}).get("passed") else "FAIL",
        )

    # Build FAISS index after pipeline
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
