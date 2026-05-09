# Failure Mode Runbook — AI Resume Coach

**Version:** 1.0  
**Created:** 2026-05-09  
**Owner:** Resume Coach Engineering  
**Scope:** All agents in the LangGraph pipeline (gap_analyzer, ats_scorer, coach_writer, interview_generator, supervisor)

---

## How to Use This Runbook

Each entry follows this structure:
- **Trigger** — what input or condition causes this failure
- **Symptom** — what you observe in logs or UI
- **Detection** — how to confirm it programmatically
- **Blast Radius** — downstream impact if undetected
- **Mitigation** — fix the root cause
- **Recovery** — how to recover a failed run without full re-run
- **Status** — Open / Mitigated / Resolved

---

## FM-001: gap_analyzer hallucinates skill not in resume

**Trigger:** JD contains a niche skill with no semantic match in the resume. The model fills the gap with a plausible-sounding but incorrect assessment.

**Symptom:**
```
gap_analysis.skills_gap.present_and_strong contains a skill
that does not appear anywhere in the resume text
```
Example: resume has no mention of "Kubernetes" but `present_and_strong` lists it.

**Detection:**
- JudgeAgent `accuracy` dimension scores 0 → `critical_failure: true`
- eval harness exits with code 1
- LangFuse trace shows gap_analyzer output with ungrounded skill claims

**Blast Radius:**
- coach_writer builds rewrite recommendations around fabricated strength
- ats_scorer inflates keyword match score
- interview_generator skips probing a real gap
- User submits resume with false claim to recruiter → reputational risk

**Mitigation:**
1. Add citation-grounding instruction to gap_analyzer system prompt:
   ```
   For each skill in present_and_strong, you MUST quote the exact
   resume sentence that confirms it. If you cannot quote it, do not
   list it.
   ```
2. JudgeAgent `accuracy=0` blocks the run via CI gate

**Recovery:**
- Re-run gap_analyzer node only using LangGraph checkpointing (session_id preserved)
- Do not re-run full pipeline — other agents' outputs are valid

**Status:** Mitigated (JudgeAgent catches; prompt hardening pending)

---

## FM-002: Parallel agent state overwrite (race condition)

**Trigger:** coach_writer, ats_scorer, and interview_generator run in parallel and one agent's `{**state}` spread overwrites another's output with an empty dict.

**Symptom:**
```
ats_report: {}   ← empty despite ats_scorer completing successfully
coaching_report: {}  ← same
INFO app.agents.ats_scorer — ATS scorer complete  ← agent ran fine
```

**Detection:**
- API response shows `ats_report: {}` or `coaching_report: {}`
- UI ATS tab shows no data despite successful log line
- `completed_nodes` includes `ats_scorer` but state field is empty

**Blast Radius:**
- ATS tab blank in UI
- Coaching tab blank in UI
- JudgeAgent skips both agents ("no output found")
- eval harness average score artificially low

**Mitigation:**
- All agent `run()` functions must return ONLY owned fields — never `{**state, ...}`
- LangGraph's state reducers handle merging automatically
- This fix is already applied as of 2026-05-09

**Recovery:**
- Restart backend (clears `lru_cache` on compiled graph)
- Re-submit analysis — fixed agents will write correctly

**Status:** Resolved (2026-05-09 — removed `**state` spread from all agents)

---

## FM-003: Budget exhaustion mid-pipeline

**Trigger:** Token or cost budget is hit after gap_analyzer but before parallel agents complete. Happens on very long resumes (>8000 chars) or with expensive providers.

**Symptom:**
```
WARNING app.agents.supervisor — Budget blocked ats_scorer:
Token budget exhausted: 7800/8000 used, ats_scorer needs ~2000 more
```

**Detection:**
- `supervisor_report.budget_summary.token_budget_pct` > 90
- One or more parallel agents show `budget_blocked` in errors
- UI shows gap analysis but missing ATS / coaching tabs

**Blast Radius:**
- Partial pipeline output delivered to user
- No ATS or coaching data — reduced product value
- User may not notice if UI doesn't surface the error clearly

**Mitigation:**
1. Increase `BUDGET_MAX_TOKENS` in `.env` for long resumes
2. Add resume truncation at 5000 chars before pipeline entry
3. Surface budget warning in UI when `token_budget_pct > 80`

**Recovery:**
- Increase budget in `.env`, restart backend, re-submit
- Or switch to `quick` routing mode for immediate partial result

**Status:** Open — UI warning not yet implemented

---

## FM-004: Replicate rate limiting (429)

**Trigger:** Multiple parallel agent calls hit Replicate's rate limit simultaneously. Occurs with < $5 account credit or burst traffic.

**Symptom:**
```
ERROR app.agents.ats_scorer — ats_scorer failed: ReplicateError
status: 429
detail: Request was throttled. Rate limit resets in ~10s.
```

**Detection:**
- `errors` list in pipeline output contains `429` references
- Multiple agents fail simultaneously (not sequential)
- LangFuse traces show short-duration spans with error

**Blast Radius:**
- Multiple agents fail in same run
- `errors=[]` condition in logs is violated
- eval harness may pass (mock mode) but live run fails

**Mitigation:**
1. Add $10+ Replicate credit (removes burst limit)
2. Add retry with exponential backoff in `_make_budget_node`:
   ```python
   import time
   for attempt in range(2):
       try:
           return agent_run_fn(state)
       except Exception as e:
           if "429" in str(e) and attempt == 0:
               time.sleep(15)
               continue
           raise
   ```
3. Long-term: switch to SageMaker (no rate limits, self-hosted)

**Recovery:**
- Wait 60 seconds, re-submit
- Or switch `LLM_PROVIDER=ollama` for local fallback (slow but no rate limit)

**Status:** Mitigated ($10 credit added; retry logic pending)

---

## FM-005: JSON parse failure in agent output

**Trigger:** LLM returns malformed JSON — common with smaller models (Mistral-7B, Llama-3-8B) when context is long or prompt is complex.

**Symptom:**
```
ERROR app.agents.coach_writer — coach_writer failed:
json.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```
Or agent returns markdown-fenced JSON that wasn't stripped:
```json
```json
{"executive_summary": ...}
```  ← trailing fence not stripped
```

**Detection:**
- Agent error in `completed_nodes` with `json.JSONDecodeError`
- `coaching_report: {"error": "...JSONDecodeError..."}` in state
- LangFuse trace shows raw LLM output that starts with ` ```json`

**Blast Radius:**
- Agent output is `{"error": "..."}` — downstream agents may still run but on empty input
- coach_writer failure means no coaching recommendations in UI

**Mitigation:**
1. Improve markdown fence stripping (current code handles ` ```json` but not all variants):
   ```python
   import re
   raw = re.sub(r'```(?:json)?|```', '', raw).strip()
   ```
2. Add JSON repair fallback using `json-repair` library:
   ```python
   from json_repair import repair_json
   gap_analysis = json.loads(repair_json(raw))
   ```
3. Add explicit instruction in system prompt: `Return ONLY the raw JSON object. Do not use markdown fences.`

**Recovery:**
- Re-run failed agent only (LangGraph checkpoint)
- If persistent: switch to `bedrock` provider (Claude is more reliable at JSON schema adherence)

**Status:** Partially mitigated (basic fence stripping exists; json-repair not yet added)

---

## FM-006: FAISS index build failure

**Trigger:** Embedding model not available (Ollama not running, wrong model name, 404 on embed endpoint).

**Symptom:**
```
WARNING app.graph — FAISS index build failed (Q&A will use state-only):
model "nomic-embed-text" not found, try pulling it first (status code: 404)
```

**Detection:**
- `FAISS session index built for Q&A` log line absent after pipeline
- Q&A tab returns state-only answers (no RAG grounding)
- httpx shows 404 on `/api/embed` endpoint

**Blast Radius:**
- Q&A tab works but answers are ungrounded — just LLM memory
- No semantic search over resume/JD/gap content
- User gets lower-quality Q&A responses without noticing

**Mitigation:**
1. Pull missing model: `ollama pull nomic-embed-text`
2. Add startup health check that verifies embedding model availability
3. Surface warning in UI when RAG is unavailable

**Recovery:**
```bash
ollama pull nomic-embed-text
# Restart backend — FAISS will build on next analysis submission
```

**Status:** Mitigated (model pull documented; startup check pending)

---

## FM-007: Supervisor routing to wrong mode

**Trigger:** Resume or JD text is very short (test data, copy-paste error) causing supervisor to route to `quick` mode when user expects full analysis.

**Symptom:**
```
INFO app.agents.supervisor — Supervisor: routing=quick (resume=312 jd=95)
INFO app.graph — Supervisor: quick mode — running ats_scorer only
```
User sees gap analysis but no coaching or interview questions.

**Detection:**
- `supervisor_report.routing_mode = "quick"` in API response
- `completed_nodes` missing `coach_writer` and `interview_generator`
- UI coaching and interview tabs empty

**Blast Radius:**
- User gets incomplete analysis without clear explanation
- No coaching recommendations or interview prep

**Mitigation:**
1. Surface routing mode in UI with explanation ("Quick mode: short input detected")
2. Add minimum length validation at API entry point before pipeline starts:
   ```python
   if len(resume_text) < 200:
       raise HTTPException(400, "Resume text too short — minimum 200 characters")
   ```
3. Lower `quick` mode threshold or remove it for MVP

**Recovery:**
- Re-submit with full resume text
- Or force `full` mode by temporarily removing quick-mode routing logic

**Status:** Open — UI surfacing not yet implemented

---

## FM-008: SageMaker endpoint cold start timeout

**Trigger:** SageMaker endpoint was idle and needs to cold-start. First request times out before the instance is ready.

**Symptom:**
```
ERROR app.agents.gap_analyzer — gap_analyzer failed:
ModelError: An error occurred (ModelError) when calling the
InvokeEndpoint operation: Endpoint is in Creating state
```
Or request times out after 60s with no response.

**Detection:**
- Error references `Creating` or `InService` state mismatch
- AWS console shows endpoint status transitioning
- Only affects first request after long idle period

**Blast Radius:**
- First user request fails completely
- Subsequent requests succeed (endpoint warm)
- Demo scenario risk: first live demo request fails

**Mitigation:**
1. Send a warm-up request before demo:
   ```bash
   curl -X POST http://localhost:8000/health  # triggers endpoint check
   ```
2. Add keep-alive ping every 30 minutes via scheduled Lambda
3. Switch to `replicate` provider for demos (no cold start)

**Recovery:**
- Wait 2-3 minutes for endpoint to reach `InService` state
- Re-submit request
- Always verify endpoint status before graded demo:
  ```bash
  aws sagemaker describe-endpoint --endpoint-name resume-coach-mistral-7b \
    --query 'EndpointStatus'
  ```

**Status:** Open — keep-alive not yet implemented

---

## FM-009: LangFuse trace not captured

**Trigger:** `LANGFUSE_ENABLED=false` in `.env`, or LangFuse credentials are invalid, or network connectivity to `cloud.langfuse.com` is blocked.

**Symptom:**
```
INFO app.graph — Starting pipeline — session_id=abc123
# ... pipeline runs normally ...
trace_url: ""  ← empty in API response
```
No traces appear in LangFuse dashboard.

**Detection:**
- `trace_url` field empty in API response
- LangFuse dashboard shows no recent traces
- No error in logs (tracer fails silently by design)

**Blast Radius:**
- Pipeline runs correctly — no functional impact
- Observability lost: cannot debug agent behavior or score regression
- JudgeAgent cannot correlate scores with traces

**Mitigation:**
1. Set `LANGFUSE_ENABLED=true` and add valid credentials to `.env`
2. Add startup log that confirms LangFuse connection:
   ```python
   logger.info("LangFuse: enabled=%s host=%s", settings.langfuse_enabled, settings.langfuse_host)
   ```

**Recovery:**
- Fix credentials in `.env`, restart backend
- Historical traces cannot be recovered — observability is forward-only

**Status:** Open — currently disabled in dev (`LANGFUSE_ENABLED=false`)

---

## FM-010: XGBoost model missing or version mismatch

**Trigger:** `data/ml/xgb_model.joblib` missing (fresh clone, deleted by mistake) or trained with a different XGBoost version than the runtime.

**Symptom:**
```
WARNING app.ml.ml_scorer — XGBoost model not found at
data/ml/xgb_model.joblib — using heuristic fallback.
Run scripts/train_xgb.py to train.
```
Or pickle version warning:
```
UserWarning: If you are loading a serialized model generated by an
older version of XGBoost...
```

**Detection:**
- `xgb_hire_prob` in API response is suspiciously round (heuristic fallback returns fixed values)
- Warning log appears on every request
- `data/ml/xgb_model.joblib` absent from filesystem

**Blast Radius:**
- ML fast pass still runs but hire probability is heuristic not model-based
- ATS composite score less accurate
- gap_analyzer prompt receives degraded ML signals

**Mitigation:**
1. Run training script after every fresh clone:
   ```bash
   python scripts/train_xgb.py
   ```
2. Add model files to CI artifact cache so they persist across runs
3. Add startup assertion:
   ```python
   assert Path("data/ml/xgb_model.joblib").exists(), \
       "XGBoost model missing — run: python scripts/train_xgb.py"
   ```

**Recovery:**
```bash
python scripts/train_xgb.py
# Restart backend — model loads on next request
```

**Status:** Mitigated (train script exists; startup assertion pending)

---

## Quick Reference

| ID | Agent | Failure | Severity | Status |
|----|-------|---------|----------|--------|
| FM-001 | gap_analyzer | Hallucinated skill | 🔴 Critical | Mitigated |
| FM-002 | All parallel | State overwrite race | 🔴 Critical | Resolved |
| FM-003 | Supervisor | Budget exhaustion | 🟡 High | Open |
| FM-004 | All agents | Replicate 429 | 🟡 High | Mitigated |
| FM-005 | All agents | JSON parse failure | 🟡 High | Partial |
| FM-006 | RAG/Q&A | FAISS index failure | 🟠 Medium | Mitigated |
| FM-007 | Supervisor | Wrong routing mode | 🟠 Medium | Open |
| FM-008 | All agents | SageMaker cold start | 🟠 Medium | Open |
| FM-009 | Observability | LangFuse not tracing | 🟢 Low | Open |
| FM-010 | ML scorer | XGBoost missing | 🟢 Low | Mitigated |

---

## Runbook Maintenance

- Add a new entry for every production incident
- Update `Status` field when a mitigation is implemented
- Re-run eval harness after any mitigation to confirm fix
- Review and update quarterly or after major dependency upgrades
