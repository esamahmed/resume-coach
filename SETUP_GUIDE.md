# Resume Coach — File-by-File Setup Guide
## Download every file from Claude outputs and place it here

---

## Step 1 — Create the full folder structure first

Run this ONE command from inside your `resume-coach/` folder:

```bash
mkdir -p app/{agents,api,core,ml,observability,parsers,rag} \
         eval/fixtures \
         scripts \
         docs \
         data/ml \
         frontend/src/{api,hooks,utils,types,styles} \
         frontend/src/components/{Header,Hero,UploadSection,LoadingState,ScoreRow,TabNav,Results} \
         frontend/src/components/tabs/{OverviewTab,CoachingTab,ATSTab,InterviewTab,ChatTab}
```

Then create all Python `__init__.py` files:

```bash
touch app/__init__.py \
      app/agents/__init__.py \
      app/api/__init__.py \
      app/core/__init__.py \
      app/ml/__init__.py \
      app/observability/__init__.py \
      app/parsers/__init__.py \
      app/rag/__init__.py \
      eval/__init__.py
```

---

## Step 2 — Download and place each file

### 📁 ROOT  `resume-coach/`
| Download from Claude outputs | Save as |
|---|---|
| `requirements.txt` | `resume-coach/requirements.txt` |
| `Dockerfile` | `resume-coach/Dockerfile` |
| `docker-compose.yml` | `resume-coach/docker-compose.yml` |
| `nginx.conf` | `resume-coach/nginx.conf` |
| `.env.example` | `resume-coach/.env.example` |
| `.gitignore` | `resume-coach/.gitignore` |
| `README.md` | `resume-coach/README.md` |
| `PROJECT_STRUCTURE.md` | `resume-coach/PROJECT_STRUCTURE.md` |

---

### 📁 `app/`
| Download | Save as |
|---|---|
| `app/graph.py` | `resume-coach/app/graph.py` |

---

### 📁 `app/core/`
| Download | Save as |
|---|---|
| `app/core/config.py` | `resume-coach/app/core/config.py` |
| `app/core/llm.py` | `resume-coach/app/core/llm.py` |
| `app/core/state.py` | `resume-coach/app/core/state.py` |

---

### 📁 `app/agents/`  ← The 5 LangGraph nodes
| Download | Save as |
|---|---|
| `app/agents/gap_analyzer.py` | `resume-coach/app/agents/gap_analyzer.py` |
| `app/agents/coach_writer.py` | `resume-coach/app/agents/coach_writer.py` |
| `app/agents/ats_scorer.py` | `resume-coach/app/agents/ats_scorer.py` |
| `app/agents/interview_generator.py` | `resume-coach/app/agents/interview_generator.py` |
| `app/agents/qa_agent.py` | `resume-coach/app/agents/qa_agent.py` |

---

### 📁 `app/ml/`
| Download | Save as |
|---|---|
| `app/ml/ml_scorer.py` | `resume-coach/app/ml/ml_scorer.py` |

---

### 📁 `app/rag/`
| Download | Save as |
|---|---|
| `app/rag/vector_store.py` | `resume-coach/app/rag/vector_store.py` |

---

### 📁 `app/parsers/`
| Download | Save as |
|---|---|
| `app/parsers/parser.py` | `resume-coach/app/parsers/parser.py` |

---

### 📁 `app/observability/`
| Download | Save as |
|---|---|
| `app/observability/langfuse_tracer.py` | `resume-coach/app/observability/langfuse_tracer.py` |

---

### 📁 `app/api/`
| Download | Save as |
|---|---|
| `app/api/main.py` | `resume-coach/app/api/main.py` |

---

### 📁 `eval/`
| Download | Save as |
|---|---|
| `eval/harness.py` | `resume-coach/eval/harness.py` |
| `eval/fixtures/RC-001-java-to-ai-engineer.json` | `resume-coach/eval/fixtures/RC-001-java-to-ai-engineer.json` |

---

### 📁 `scripts/`
| Download | Save as |
|---|---|
| `scripts/deploy_sagemaker.py` | `resume-coach/scripts/deploy_sagemaker.py` |

---

### 📁 `docs/`
| Download | Save as |
|---|---|
| `docs/prompt_experiment_log.md` | `resume-coach/docs/prompt_experiment_log.md` |
| `docs/finetuning.md` | `resume-coach/docs/finetuning.md` |

---

### 📁 `frontend/`  ← Use the `frontend_ts` folder (TypeScript version)
| Download from `frontend_ts/` | Save as |
|---|---|
| `frontend_ts/index.html` | `resume-coach/frontend/index.html` |
| `frontend_ts/package.json` | `resume-coach/frontend/package.json` |
| `frontend_ts/tsconfig.json` | `resume-coach/frontend/tsconfig.json` |
| `frontend_ts/tsconfig.node.json` | `resume-coach/frontend/tsconfig.node.json` |
| `frontend_ts/vite.config.ts` | `resume-coach/frontend/vite.config.ts` |
| `frontend_ts/.env.example` | `resume-coach/frontend/.env.example` |

---

### 📁 `frontend/src/`
| Download from `frontend_ts/src/` | Save as |
|---|---|
| `src/main.tsx` | `frontend/src/main.tsx` |
| `src/App.tsx` | `frontend/src/App.tsx` |
| `src/App.module.scss` | `frontend/src/App.module.scss` |
| `src/vite-env.d.ts` | `frontend/src/vite-env.d.ts` |

---

### 📁 `frontend/src/types/`
| Download | Save as |
|---|---|
| `src/types/index.ts` | `frontend/src/types/index.ts` |

---

### 📁 `frontend/src/api/`
| Download | Save as |
|---|---|
| `src/api/client.ts` | `frontend/src/api/client.ts` |

---

### 📁 `frontend/src/hooks/`
| Download | Save as |
|---|---|
| `src/hooks/useAnalyze.ts` | `frontend/src/hooks/useAnalyze.ts` |

---

### 📁 `frontend/src/utils/`
| Download | Save as |
|---|---|
| `src/utils/formatters.ts` | `frontend/src/utils/formatters.ts` |

---

### 📁 `frontend/src/styles/`
| Download | Save as |
|---|---|
| `src/styles/_variables.scss` | `frontend/src/styles/_variables.scss` |
| `src/styles/global.scss` | `frontend/src/styles/global.scss` |

---

### 📁 `frontend/src/components/`  ← One folder per component, each with .tsx + .module.scss
| Download | Save as |
|---|---|
| `Header/Header.tsx` | `frontend/src/components/Header/Header.tsx` |
| `Header/Header.module.scss` | `frontend/src/components/Header/Header.module.scss` |
| `Hero/Hero.tsx` | `frontend/src/components/Hero/Hero.tsx` |
| `Hero/Hero.module.scss` | `frontend/src/components/Hero/Hero.module.scss` |
| `UploadSection/UploadSection.tsx` | `frontend/src/components/UploadSection/UploadSection.tsx` |
| `UploadSection/UploadSection.module.scss` | `frontend/src/components/UploadSection/UploadSection.module.scss` |
| `LoadingState/LoadingState.tsx` | `frontend/src/components/LoadingState/LoadingState.tsx` |
| `LoadingState/LoadingState.module.scss` | `frontend/src/components/LoadingState/LoadingState.module.scss` |
| `ScoreRow/ScoreRow.tsx` | `frontend/src/components/ScoreRow/ScoreRow.tsx` |
| `ScoreRow/ScoreRow.module.scss` | `frontend/src/components/ScoreRow/ScoreRow.module.scss` |
| `TabNav/TabNav.tsx` | `frontend/src/components/TabNav/TabNav.tsx` |
| `TabNav/TabNav.module.scss` | `frontend/src/components/TabNav/TabNav.module.scss` |
| `Results/Results.tsx` | `frontend/src/components/Results/Results.tsx` |
| `Results/Results.module.scss` | `frontend/src/components/Results/Results.module.scss` |

---

### 📁 `frontend/src/components/tabs/`
| Download | Save as |
|---|---|
| `tabs/OverviewTab/OverviewTab.tsx` | `frontend/src/components/tabs/OverviewTab/OverviewTab.tsx` |
| `tabs/CoachingTab/CoachingTab.tsx` | `frontend/src/components/tabs/CoachingTab/CoachingTab.tsx` |
| `tabs/CoachingTab/CoachingTab.module.scss` | `frontend/src/components/tabs/CoachingTab/CoachingTab.module.scss` |
| `tabs/ATSTab/ATSTab.tsx` | `frontend/src/components/tabs/ATSTab/ATSTab.tsx` |
| `tabs/ATSTab/ATSTab.module.scss` | `frontend/src/components/tabs/ATSTab/ATSTab.module.scss` |
| `tabs/InterviewTab/InterviewTab.tsx` | `frontend/src/components/tabs/InterviewTab/InterviewTab.tsx` |
| `tabs/InterviewTab/InterviewTab.module.scss` | `frontend/src/components/tabs/InterviewTab/InterviewTab.module.scss` |
| `tabs/ChatTab/ChatTab.tsx` | `frontend/src/components/tabs/ChatTab/ChatTab.tsx` |
| `tabs/ChatTab/ChatTab.module.scss` | `frontend/src/components/tabs/ChatTab/ChatTab.module.scss` |

---

## Step 3 — Configure .env

```bash
cd resume-coach
cp .env.example .env
# Edit .env and fill in your keys
```

Minimum required to run locally:
- `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` (or use OpenAI instead)
- `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` (free at cloud.langfuse.com)

To use OpenAI instead of Bedrock: set `LLM_PROVIDER=openai` and fill `OPENAI_API_KEY`

---

## Step 4 — Install and run

```bash
# Backend
cd resume-coach
source .venv/bin/activate
pip install -r requirements.txt
python -m app.api.main
# → http://localhost:8000/docs

# Frontend (new terminal)
cd resume-coach/frontend
cp .env.example .env          # set VITE_API_URL=http://localhost:8000
npm install
npm run dev
# → http://localhost:5173
```

---

## Step 5 — Verify it's working

```bash
# Health check
curl http://localhost:8000/health
# Expected: {"status":"ok","service":"resume-coach"}
```
