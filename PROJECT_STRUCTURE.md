# AI Resume Coach — Full Project Structure & Setup Commands
### P5 Capstone | Interview Kickstart Advanced Machine Learning Program

---

## Complete File Tree

```
resume-coach/
│
├── .env.example                          # Environment variable template
├── Dockerfile                            # Backend container (FastAPI)
├── docker-compose.yml                    # Backend + Frontend (nginx)
├── nginx.conf                            # Reverse proxy config
├── requirements.txt                      # Python dependencies
├── README.md                             # Project overview & architecture
│
├── app/                                  # FastAPI backend
│   ├── __init__.py
│   ├── graph.py                          # LangGraph pipeline (entry point)
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                     # Pydantic settings (env-driven)
│   │   ├── llm.py                        # LLM factory (Bedrock / OpenAI)
│   │   └── state.py                      # ResumeCoachState TypedDict
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── gap_analyzer.py               # Node 1 — structured gap analysis
│   │   ├── coach_writer.py               # Node 2 — coaching report + bullet rewrites
│   │   ├── ats_scorer.py                 # Node 3 — composite ATS report
│   │   ├── interview_generator.py        # Node 4 — gap-targeted interview questions
│   │   └── qa_agent.py                   # Node 5 — RAG-powered Q&A (on-demand)
│   │
│   ├── ml/
│   │   ├── __init__.py
│   │   └── ml_scorer.py                  # TF-IDF ATS score + XGBoost hire probability
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   └── vector_store.py               # FAISS session index (resume + gap findings)
│   │
│   ├── parsers/
│   │   ├── __init__.py
│   │   └── parser.py                     # PDF + DOCX text extraction
│   │
│   ├── observability/
│   │   ├── __init__.py
│   │   └── langfuse_tracer.py            # Langfuse span-level tracing
│   │
│   └── api/
│       ├── __init__.py
│       └── main.py                       # FastAPI routes: /analyze  /chat  /health
│
├── eval/                                 # Eval harness (mentor pattern)
│   ├── __init__.py
│   ├── harness.py                        # DVC + corpus_hash + LLM-as-judge scorer
│   └── fixtures/
│       └── RC-001-java-to-ai-engineer.json   # Golden test fixture
│
├── scripts/
│   └── deploy_sagemaker.py               # Deploy Llama-2 / Mistral on SageMaker
│
├── docs/
│   ├── prompt_experiment_log.md          # 7 documented prompt experiments (25% rubric)
│   └── finetuning.md                     # QLoRA fine-tuning code + hyperparameter rationale
│
└── frontend/                             # React + TypeScript + SCSS frontend
    ├── index.html
    ├── package.json
    ├── tsconfig.json
    ├── tsconfig.node.json
    ├── vite.config.ts
    ├── .env.example
    │
    └── src/
        ├── main.tsx                      # React entry point
        ├── App.tsx                       # Root component (orchestration only)
        ├── App.module.scss
        ├── vite-env.d.ts                 # VITE_ env var type declarations
        │
        ├── types/
        │   └── index.ts                  # All domain types (AnalyzeResult, GapAnalysis…)
        │
        ├── api/
        │   └── client.ts                 # Typed fetch wrappers (analyzeResume, sendChat)
        │
        ├── hooks/
        │   └── useAnalyze.ts             # All analyze state + logic extracted from App
        │
        ├── utils/
        │   └── formatters.ts             # pct(), score(), TABS, LOADING_STEPS, constants
        │
        ├── styles/
        │   ├── _variables.scss           # Single source of truth: colors, fonts, radii, mixins
        │   └── global.scss               # CSS reset, :root tokens, animations, utility classes
        │
        └── components/
            ├── Header/
            │   ├── Header.tsx
            │   └── Header.module.scss
            │
            ├── Hero/
            │   ├── Hero.tsx
            │   └── Hero.module.scss
            │
            ├── UploadSection/
            │   ├── UploadSection.tsx
            │   └── UploadSection.module.scss
            │
            ├── LoadingState/
            │   ├── LoadingState.tsx
            │   └── LoadingState.module.scss
            │
            ├── ScoreRow/
            │   ├── ScoreRow.tsx
            │   └── ScoreRow.module.scss
            │
            ├── TabNav/
            │   ├── TabNav.tsx
            │   └── TabNav.module.scss
            │
            ├── Results/
            │   ├── Results.tsx            # Orchestrates all tabs + ScoreRow + TabNav
            │   └── Results.module.scss
            │
            └── tabs/
                ├── OverviewTab/
                │   └── OverviewTab.tsx    # Strengths, gaps, keyword chips
                │
                ├── CoachingTab/
                │   ├── CoachingTab.tsx    # Prioritised recs + rewritten bullets
                │   └── CoachingTab.module.scss
                │
                ├── ATSTab/
                │   ├── ATSTab.tsx         # ATS meter, composite scores, keyword list
                │   └── ATSTab.module.scss
                │
                ├── InterviewTab/
                │   ├── InterviewTab.tsx   # Gap-targeted questions with hints
                │   └── InterviewTab.module.scss
                │
                └── ChatTab/
                    ├── ChatTab.tsx        # Conversational Q&A with suggestion pills
                    └── ChatTab.module.scss
```

---

## One-Command Project Bootstrap

Run this from an empty directory to recreate the entire folder structure:

```bash
# ── Backend directories ───────────────────────────────────────────────
mkdir -p resume-coach/{app/{agents,api,core,ml,observability,parsers,rag},\
eval/fixtures,\
scripts,\
docs,\
data/{ml,faiss_index},\
tests}

# ── Frontend directories ──────────────────────────────────────────────
mkdir -p resume-coach/frontend/src/{api,hooks,utils,types,styles,\
components/{Header,Hero,UploadSection,LoadingState,ScoreRow,TabNav,Results},\
components/tabs/{OverviewTab,CoachingTab,ATSTab,InterviewTab,ChatTab}}

# ── Python __init__.py files ──────────────────────────────────────────
touch resume-coach/app/__init__.py
touch resume-coach/app/{agents,api,core,ml,observability,parsers,rag}/__init__.py
touch resume-coach/eval/__init__.py

echo "✅ Directory structure created"
```

---

## Backend Setup

```bash
cd resume-coach

# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env — fill in Bedrock or OpenAI keys + Langfuse keys

# 4. Run API locally
python -m app.api.main
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

---

## Frontend Setup

```bash
cd resume-coach/frontend

# 1. Install dependencies (includes TypeScript, Vite, Sass)
npm install

# 2. Configure environment
cp .env.example .env
# Edit .env:
#   VITE_API_URL=http://localhost:8000
#   VITE_API_KEY=dev-key-change-in-production

# 3. Type-check only (no build)
npm run type-check

# 4. Dev server (hot reload, proxies /api → localhost:8000)
npm run dev
# → http://localhost:5173

# 5. Production build (runs tsc first, then Vite)
npm run build
# → dist/ folder ready for nginx / EC2
```

---

## Docker (Full Stack)

```bash
cd resume-coach

# Build and start both services
docker-compose up --build

# Backend:  http://localhost:8000
# Frontend: http://localhost:3000

# Stop (preserve data)
docker-compose stop

# Teardown
docker-compose down
```

---

## AWS SageMaker — Deploy LLM

```bash
# Deploy Mistral-7B-Instruct via JumpStart (recommended)
python scripts/deploy_sagemaker.py --model mistral --method jumpstart

# Deploy Llama-2-7B-Chat via HuggingFace DLC (4-bit quantized)
python scripts/deploy_sagemaker.py --model llama2 --method huggingface

# Test deployed endpoint
python scripts/deploy_sagemaker.py --model mistral --test

# ⚠️  Delete endpoint when not in use (stops billing)
python scripts/deploy_sagemaker.py --delete
```

---

## Eval Harness

```bash
# Validate fixture structure (no LLM, no cost)
python -m eval.harness --dry-run

# Full eval run — heuristic scoring only
python -m eval.harness

# Full eval run — with LLM-as-judge scoring
python -m eval.harness --llm-judge

# Run a single fixture
python -m eval.harness --fixture RC-001-java-to-ai-engineer

# DVC setup (version-control the eval corpus)
dvc init
dvc add eval/fixtures/
dvc remote add -d s3remote s3://your-bucket/resume-coach/eval/
dvc push
```

---

## EC2 Deployment

```bash
# On your EC2 instance (Ubuntu 24):

# 1. Install Docker
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-plugin
sudo systemctl start docker

# 2. Clone repo and configure
git clone https://github.com/your-org/resume-coach.git
cd resume-coach
cp .env.example .env && nano .env        # fill in credentials

# 3. Build React frontend
cd frontend && npm ci && npm run build && cd ..

# 4. Start services
docker-compose up -d --build

# 5. Check health
curl http://localhost:8000/health
```

---

## Key Architecture Decisions

| Decision | Choice | Why |
|---|---|---|
| Agent orchestration | LangGraph (not SequentialChain) | Shared `ResumeCoachState` — no context loss between agents |
| LLM inference | AWS Bedrock (Claude Sonnet) / SageMaker (Mistral-7B) | Capstone requirement + cost control |
| Embeddings | Amazon Titan Embed v2 | Native Bedrock, no separate hosting |
| Vector store | FAISS (session-scoped) | No server needed, rebuilt per session with gap findings included |
| Classical ML pre-pass | TF-IDF + XGBoost | Surfaces load-bearing signals before expensive LLM tokens run |
| Observability | Langfuse (span per node) | Mentor requirement: span-level tracing, not just CloudWatch |
| Eval | DVC + corpus_hash + LLM-as-judge | Mentor pattern: prevents cross-run score drift |
| Frontend | React + TypeScript + SCSS Modules | TypeScript strict mode; SCSS modules for scoped, variable-driven styles |

---

## Rubric Coverage

| Criterion | Weight | How Covered |
|---|---|---|
| Prompt Engineering | 25% | 7 documented experiments in `docs/prompt_experiment_log.md`; JSON schema enforcement; ATS injection; temperature tuning table |
| Model Deployment | 25% | `scripts/deploy_sagemaker.py` (JumpStart + HuggingFace DLC); Dockerfile; docker-compose; EC2 steps above |
| Web Application | 25% | React + TypeScript; drag-and-drop upload; 5-tab results dashboard; chatbot with suggestion pills |
| Innovation | 10% | Classical ML fast-pass; FAISS RAG; Langfuse tracing; DVC eval harness |
| Model + Docs | 10% | `docs/finetuning.md` — QLoRA code, dataset prep, hyperparameter rationale |
| EDA + Data Prep | 5% | Golden fixture set; LinkedIn/Indeed Kaggle dataset integration in fine-tuning script |
