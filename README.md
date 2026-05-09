# AI Resume Coach — Capstone Project
### Advanced Machine Learning Program | Interview Kickstart

End-to-end Generative AI application that accepts a resume and job description,
performs multi-stage gap analysis, and delivers a structured coaching report,
ATS optimization, and interview preparation — powered by LangGraph, FAISS,
Langfuse, and classical ML.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI  (POST /analyze, /chat)         │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│              LangGraph — ResumeCoachState                   │
│                                                             │
│  [ml_fast_pass]                                             │
│    TF-IDF ATS score + XGBoost hire-probability             │
│         │                                                   │
│  [gap_analyzer]   ← Langfuse span                          │
│    Structured gap analysis (JSON schema enforced)           │
│         │                                                   │
│    ┌────┴──────────────────┐                               │
│    ▼                       ▼            ▼                  │
│  [coach_writer]   [ats_scorer]  [interview_generator]      │
│    Coaching report  Composite     Gap-aware questions       │
│    + bullet rewrites  ATS report                           │
└─────────────────────────────────────────────────────────────┘
                           │
                    FAISS Index built
                 (resume + JD + gap findings)
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  [qa_agent] — RAG-powered conversational Q&A                │
│  Reads LIVE from shared state (no context loss)             │
│  FAISS retrieval grounds answers in actual evidence         │
└─────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Langfuse — span-level tracing per agent node               │
│  DVC — eval corpus versioning + corpus_hash                 │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

**LangGraph over LangChain SequentialChain**
SequentialChain passes serialized strings between chains — the Q&A agent
loses access to the reasoning behind earlier outputs. LangGraph uses a shared
`ResumeCoachState` TypedDict: every node reads/writes to the same typed object,
so the Q&A agent always has live access to gap findings, ATS scores, and
recommendations without context loss.

**Classical ML fast pass before LLM**
TF-IDF cosine similarity + XGBoost hire-probability run synchronously before
any LLM call. This surfaces load-bearing signals cheaply — the gap analyzer
gets keyword context injected into its prompt, and the ATS scorer gets
pre-computed scores without redundant LLM calls.

**Langfuse tracing from line one**
Every agent node emits a Langfuse span: input, output, token usage, errors.
Not just CloudWatch logs — span-level tracing per agent.

**DVC + corpus_hash for eval**
Golden eval fixtures (resume+JD pairs with expected outputs) are DVC-tracked.
Every eval run stamps a `corpus_hash` — if fixtures change, the hash changes
and scores are not comparable across runs.

## Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph |
| LLM inference | AWS Bedrock (Claude Sonnet) / OpenAI |
| Embeddings | Amazon Titan Embed v2 / OpenAI |
| Vector store | FAISS (session-scoped) |
| Classical ML | TF-IDF (scikit-learn) + XGBoost |
| Observability | Langfuse (span-level) |
| API | FastAPI + Pydantic v2 |
| Eval | DVC + LLM-as-judge + heuristic harness |
| Deployment | Docker → ECR → EC2 / SageMaker |

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/your-org/resume-coach
cd resume-coach
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Fill in your Bedrock or OpenAI credentials + Langfuse keys

# 3. Run API
python -m app.api.main

# 4. Analyze a resume
curl -X POST http://localhost:8000/analyze \
  -H "X-API-Key: dev-key-change-in-production" \
  -F "resume_file=@my_resume.pdf" \
  -F "jd_text=<job description text here>"

# 5. Ask a follow-up question
curl -X POST http://localhost:8000/chat \
  -H "X-API-Key: dev-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<from analyze response>", "question": "Which bullet should I rewrite first?"}'
```

## Eval harness

```bash
# Validate fixture structure (no LLM needed)
python -m eval.harness --dry-run

# Run full eval on all fixtures
python -m eval.harness --llm-judge

# Run a single fixture
python -m eval.harness --fixture RC-001-java-to-ai-engineer
```

## DVC setup (eval corpus versioning)

```bash
dvc init
dvc add eval/fixtures/
dvc remote add -d s3remote s3://your-bucket/resume-coach/eval/
dvc push
```

## Project structure

```
resume-coach/
├── app/
│   ├── core/
│   │   ├── config.py        # All settings from env vars
│   │   ├── state.py         # ResumeCoachState TypedDict
│   │   └── llm.py           # LLM factory (Bedrock/OpenAI)
│   ├── agents/
│   │   ├── gap_analyzer.py       # Node 1 — JSON gap analysis
│   │   ├── coach_writer.py       # Node 2 — coaching report
│   │   ├── ats_scorer.py         # Node 3 — composite ATS report
│   │   ├── interview_generator.py # Node 4 — gap-aware questions
│   │   └── qa_agent.py           # Node 5 — RAG Q&A (on-demand)
│   ├── ml/
│   │   └── ml_scorer.py     # TF-IDF + XGBoost classical ML
│   ├── rag/
│   │   └── vector_store.py  # FAISS session index
│   ├── parsers/
│   │   └── parser.py        # PDF + DOCX extraction
│   ├── observability/
│   │   └── langfuse_tracer.py # Span-level Langfuse tracing
│   └── api/
│       └── main.py          # FastAPI endpoints
├── eval/
│   ├── harness.py           # DVC + corpus_hash + LLM-as-judge
│   ├── fixtures/            # Golden resume+JD test pairs (DVC-tracked)
│   └── scores/              # Per-run eval results
├── requirements.txt
├── .env.example
└── README.md
```
