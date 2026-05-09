"""
app/api/main.py

FastAPI application — exposes the Resume Coach pipeline via HTTP.

Endpoints:
  POST /analyze          — upload resume + JD, run full pipeline
  POST /chat             — ask a question about the analysis
  GET  /health           — health check
"""
from __future__ import annotations
import logging
import uuid
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.core.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Resume Coach",
    description="LangGraph + FAISS + Langfuse powered resume coaching API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (replace with Redis/DynamoDB for production)
_sessions: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def verify_api_key(x_api_key: str = Header(...)):
    settings = get_settings()
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class AnalyzeResponse(BaseModel):
    session_id: str
    trace_url: str
    tfidf_ats_score: float
    xgb_hire_prob: float
    composite_score: float
    overall_fit_score: int | None
    fit_summary: str
    gap_analysis: dict
    coaching_report: dict
    ats_report: dict
    interview_questions: list
    errors: list[str]


class ChatRequest(BaseModel):
    session_id: str
    question: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    chat_history: list[dict]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "resume-coach"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    resume_file: UploadFile = File(...),
    jd_text: str = Form(...),
    _: str = Depends(verify_api_key),
):
    """
    Upload a resume (PDF or DOCX) and paste a job description.
    Returns a full coaching session.
    """
    from app.parsers.parser import extract_text
    from app.graph import run_pipeline
    from app.ml.ml_scorer import composite_score

    # Parse resume
    try:
        resume_bytes = await resume_file.read()
        resume_text = extract_text(resume_bytes, resume_file.filename or "resume.pdf")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Resume parsing failed: {e}")

    if len(resume_text.strip()) < 100:
        raise HTTPException(status_code=400, detail="Resume text too short — check file format")
    if len(jd_text.strip()) < 50:
        raise HTTPException(status_code=400, detail="Job description too short")

    session_id = str(uuid.uuid4())

    try:
        final_state, faiss_index = run_pipeline(
            resume_text=resume_text,
            jd_text=jd_text,
            session_id=session_id,
        )
    except Exception as e:
        logger.error("Pipeline failed for session %s: %s", session_id, e)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    # Store session for Q&A
    _sessions[session_id] = {
        "state": final_state,
        "faiss_index": faiss_index,
    }

    gap = final_state.get("gap_analysis", {})
    ats = final_state.get("ats_report", {})
    tfidf = final_state.get("tfidf_ats_score", 0.0)
    xgb = final_state.get("xgb_hire_prob", 0.0)

    return AnalyzeResponse(
        session_id=session_id,
        trace_url=final_state.get("trace_url", ""),
        tfidf_ats_score=tfidf,
        xgb_hire_prob=xgb,
        composite_score=composite_score(tfidf, xgb),
        overall_fit_score=gap.get("overall_fit_score"),
        fit_summary=gap.get("fit_summary", ""),
        gap_analysis=gap,
        coaching_report=final_state.get("coaching_report", {}),
        ats_report=ats,
        interview_questions=final_state.get("interview_questions", []),
        errors=final_state.get("errors", []),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    _: str = Depends(verify_api_key),
):
    """
    Ask a question about your coaching session.
    The Q&A agent reads live from state + FAISS index — no context loss.
    """
    from app.agents.qa_agent import run as qa_run
    from app.core.state import ResumeCoachState

    session = _sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Run /analyze first.")

    state: ResumeCoachState = {
        **session["state"],
        "current_question": request.question,
    }

    updated_state = qa_run(state, faiss_index=session.get("faiss_index"))

    # Persist updated state (chat history appended)
    _sessions[request.session_id]["state"] = updated_state

    return ChatResponse(
        session_id=request.session_id,
        answer=updated_state.get("current_answer", ""),
        chat_history=updated_state.get("chat_history", []),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
