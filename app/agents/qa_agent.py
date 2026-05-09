"""
app/agents/qa_agent.py

LangGraph Node 5: Conversational Q&A Agent

The key fix over SequentialChain: this agent reads LIVE from shared state
(gap_analysis, coaching_report, ats_report) — not from a serialized string.

It also uses FAISS RAG to retrieve the most relevant resume/JD/coaching
chunks for the user's question, giving grounded, specific answers.

"Why didn't you flag my Python skills?" → retrieves Python-related chunks
from the FAISS index → answers from actual evidence, not hallucination.
"""
from __future__ import annotations
import json
import logging
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.llm import get_llm
from app.core.state import ResumeCoachState
from app.observability.langfuse_tracer import get_tracer

if TYPE_CHECKING:
    from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert resume coach helping a candidate improve their application \
for a specific role.

You have full access to:
- Their gap analysis findings
- The coaching report with recommendations
- ATS score details
- Interview preparation questions
- Relevant excerpts from their resume and the job description (via retrieval)

Answer the candidate's question specifically and concisely.
Ground your answer in the actual findings — do not make up information.
If you reference a recommendation, cite which one (e.g., "Recommendation #3").
Keep responses under 300 words unless the candidate asks for more detail.
"""

USER_TEMPLATE = """\
RETRIEVED CONTEXT (most relevant to your question):
{retrieved_context}

SESSION SUMMARY:
- Overall fit score: {fit_score}/100
- ATS composite score: {composite_score:.1%}
- Critical gaps: {critical_gaps}
- Recommendations count: {recs_count}

CONVERSATION HISTORY:
{chat_history}

CANDIDATE QUESTION: {question}
"""


def run(
    state: ResumeCoachState,
    faiss_index: "FAISS | None" = None,
) -> ResumeCoachState:
    """
    Run the Q&A agent for a single conversational turn.

    Args:
        state: Full shared state (live, not serialized)
        faiss_index: Optional FAISS index for RAG retrieval.
                     If None, falls back to state context only.
    """
    tracer = get_tracer()
    session_id = state.get("session_id", "")
    question = state.get("current_question", "")

    if not question:
        return state

    with tracer.span("qa_agent", session_id=session_id,
                     input_data={"question": question[:200]}) as span:
        try:
            # RAG retrieval
            retrieved_chunks: list[str] = []
            if faiss_index is not None:
                from app.rag.vector_store import retrieve
                retrieved_chunks = retrieve(faiss_index, question, k=4)
                logger.debug("Q&A RAG: retrieved %d chunks", len(retrieved_chunks))

            retrieved_context = "\n---\n".join(retrieved_chunks) if retrieved_chunks \
                else "No additional context retrieved."

            # Read live from state — this is the key fix
            gap = state.get("gap_analysis", {})
            ats = state.get("ats_report", {})
            coaching = state.get("coaching_report", {})
            history = state.get("chat_history", [])

            history_text = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in history[-6:]
            ) or "No prior conversation."

            prompt = USER_TEMPLATE.format(
                retrieved_context=retrieved_context[:2000],
                fit_score=gap.get("overall_fit_score", "N/A"),
                composite_score=ats.get("composite_score", 0.0),
                critical_gaps=", ".join(gap.get("critical_gaps", [])[:3]),
                recs_count=len(coaching.get("recommendations", [])),
                chat_history=history_text,
                question=question,
            )

            llm = get_llm()
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
            response = llm.invoke(messages)
            answer = response.content.strip()

            # Append to chat history
            new_history = list(history) + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]

            span.update(output={"answer_len": len(answer)})
            logger.info("Q&A agent answered question (%d chars)", len(answer))

            return {
                **state,
                "current_answer": answer,
                "chat_history": new_history,
                "completed_nodes": ["qa_agent"],
            }

        except Exception as e:
            logger.error("qa_agent failed: %s", e)
            return {
                **state,
                "current_answer": f"I encountered an error answering your question: {e}",
                "errors": [f"qa_agent: {e}"],
            }
