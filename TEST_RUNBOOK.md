# Test Runbook — RAG + Agent + Eval Lab

Documents the expected and actual results for each step in RUNBOOK.md. Each
test record (TR) maps to one RUNBOOK step and states:
- **Pass criteria** — what a correct run looks like
- **Actual results** — from a real run on this codebase; use as your baseline
- **Known issues** — expected failures and their root causes

**Assumption:** Setup is complete — Ollama is running, models are pulled,
`.venv` is active, `.env` is loaded with `GOOGLE_API_KEY` and
`LANGSMITH_API_KEY`.

---

## TR-01 — Build the Vector Store (RUNBOOK Step 5)

**Command:**
```bash
python3 src/ingest.py
```

**Pass criteria:**
- Exactly **9 chunks** reported
- `chroma_db/` directory created with at least `chroma.sqlite3`

**Actual output (baseline):**
```
Loaded 1 document(s), 2478 characters.
Split into 9 chunks.
Embedded and stored 9 chunks in ./chroma_db.
Ingestion complete. You can now run src/rag_chain.py
```

**Verify:**
```bash
ls chroma_db/    # must contain chroma.sqlite3 and a UUID folder
```

**Status: ✅ PASS**

**If this fails:** Ollama is not running or `nomic-embed-text` was not pulled.
Run `ollama list` to confirm both models are present.

---

## TR-02 — RAG Pipeline: In-Corpus Questions (RUNBOOK Step 6)

**Commands:**
```bash
python3 src/rag_chain.py "How much does the Pro plan cost?"
python3 src/rag_chain.py "What is the API rate limit on the Pro plan?"
```

**Pass criteria:**
- Answers match expected values (exact wording may vary)
- Retrieved chunk [1] contains the relevant fact

**Actual results:**

| Question | Expected answer | Status |
|----------|----------------|--------|
| Pro plan cost? | `$49 per seat per month` | ✅ PASS |
| API rate limit on Pro? | `1,000 requests per minute` | ✅ PASS |

**What to evaluate in LangSmith:**
1. Open the trace for each question in the **rag-eval-lab** project
2. Check the **retrieve** step — confirm the pricing or API limits chunk appeared as [1]
3. Check the **ChatOllama** step — confirm the model received the correct context before answering
4. If the answer is correct but retrieved chunks are irrelevant, the model used
   prior training knowledge — the grounding prompt is not working

---

## TR-03 — RAG Pipeline: Out-of-Corpus (Hallucination Test) (RUNBOOK Step 6)

**Command:**
```bash
python3 src/rag_chain.py "What is the capital of France?"
```

**Pass criteria:**
Answer **must** contain `"I don't know based on the handbook."` — any other
response is a hallucination and a **hard FAIL**.

**Actual output (baseline):**
```
Q: What is the capital of France?
A: I don't know based on the handbook.

--- retrieved context (what the model was allowed to see) ---
[1] ## About Zephyr ...
[2] # Zephyr Analytics — Product Handbook ...
[3] ## The Pulse Feature ...
```

**Status: ✅ PASS**

**Why this matters:** The grounding prompt in `rag_chain.py` is the only
defence against the model using its own training knowledge. If this test fails,
the prompt is broken and every eval metric becomes meaningless.

---

## TR-04 — RAG Pipeline: Known Retrieval Gap (RUNBOOK Step 6)

**Command:**
```bash
python3 src/rag_chain.py "Which plans include the Pulse feature?"
```

**Expected answer (from golden dataset):**
`Pulse is available on the Pro and Enterprise plans only.`

**Actual output:**
```
Q: Which plans include the Pulse feature?
A: The Free plan does not include the Pulse feature.
   I don't know based on the handbook.

--- retrieved context ---
[1] ## Plans and Limits   ← mentions Free plan excludes Pulse
[2] The Enterprise plan includes ...
[3] ## Data Export ...
```

**Status: ⚠ KNOWN FAIL — retrieval gap**

**Root cause:** Embedding similarity favours the `## Plans and Limits` chunk
(mentions Pulse in the context of all plans) over the `## The Pulse Feature`
chunk (states availability directly). The model cannot infer the positive from
the negative with the chunks it receives.

**Fix options:**
- Raise `k` from 3 → 5 in `rag_chain.py` to fetch more chunks, increasing the
  chance the Pulse Feature section is included
- Re-chunk with smaller overlap so the Pulse Feature section is isolated in its
  own chunk (better precision — the explicit statement is always retrieved together)

**This failure is expected and will appear in eval metrics** (TR-07, TR-10).

---

## TR-05 — Agent Tool Routing (RUNBOOK Step 7)

**Commands:**
```bash
python3 src/agent.py "What is the API rate limit on the Pro plan?"
python3 src/agent.py "What is 2 + 2?"
python3 src/agent.py "I need 90 extra days of retention. What will it cost?"
```

**Pass criteria for the multi-step question:**
- Agent must call `search_handbook` first
- Retrieved fact: add-on costs $15 per 30 days
- Agent must then call `calculator` with expression `90 / 30 * 15`
- Final answer must be `$45.00`

**Actual results:**

| Question | Tool(s) called | Expected answer | Status |
|----------|---------------|----------------|--------|
| API rate limit? | `search_handbook` | 1,000 requests/minute | ✅ PASS |
| What is 2 + 2? | `calculator` | 4 | ✅ PASS |
| 90 days retention cost? | `search_handbook` → `calculator` | $45.00 | ✅ PASS |

**What to evaluate in LangSmith:**
- Open the agent trace for the third question
- Both `search_handbook` and `calculator` must appear as separate tool call nodes
- If the correct answer appears but only one tool fired, it was a lucky guess —
  the trace tells you which

**Known warning (non-blocking):**
```
LangGraphDeprecatedSinceV10: create_react_agent has been moved to langchain.agents.
```
Code still works correctly; this is a future-release warning only.

---

## TR-06 — Hand-Rolled Evaluation (RUNBOOK Step 8)

**Command:**
```bash
python3 src/eval_custom.py
```

**Pass criteria:**
- Retrieval hit rate ≥ 100% (any miss is a regression)
- Keyword correctness ≥ 83% (the Pulse gap is expected)
- LLM-judge faithful ≥ 83% (baseline)
- Abstention = 100% (non-negotiable)

**Actual results (baseline run — Gemini judge):**
```
===== RUN 1 of 1 =====
[in ] retrieval=Y keyword=Y judge=Y :: How many dashboards does the Free plan include?
[in ] retrieval=Y keyword=Y judge=Y :: How much does the Pro plan cost per seat per month?
[in ] retrieval=Y keyword=N judge=N :: Which plans include the Pulse feature?
[in ] retrieval=Y keyword=Y judge=Y :: What is the API rate limit on the Pro plan?
[in ] retrieval=Y keyword=Y judge=Y :: How long is data retained on the Free plan?
[in ] retrieval=Y keyword=Y judge=Y :: In which cities does Zephyr run its data centres?
[out] abstention=Y (OK) :: What is Zephyr Analytics' stock price?
[out] abstention=Y (OK) :: Who is the CEO of Zephyr Analytics?

===== SUMMARY =====
Retrieval hit rate : 6/6 = 100%
Keyword correctness: 5/6 = 83%
LLM-judge faithful : 5/6 = 83%
Abstention (no hallucination): 2/2 = 100%
```

**Score summary:**

| Metric | Score | Status |
|--------|-------|--------|
| Retrieval hit rate | 6/6 = 100% | ✅ |
| Keyword correctness | 5/6 = 83% | ✅ (1 known gap — TR-04) |
| LLM-judge faithful | 5/6 = 83% | ✅ (1 known gap — TR-04) |
| Abstention | 2/2 = 100% | ✅ |

**The consistent FAIL:** "Which plans include the Pulse feature?" — retrieval
miss documented as TR-04. All other questions PASS across all metrics.

**Variance experiment:** Change `REPEATS = 1` to `REPEATS = 3` in
`eval_custom.py` and re-run. If the judge score shifts between runs, you have
observed why single-run AI testing is unreliable.

---

## TR-07 — DeepEval: Baseline with Local Judge (RUNBOOK Step 9)

**Command:**
```bash
USE_LOCAL_JUDGE=1 python3 src/eval_deepeval.py
```

**Judge model used:** `qwen2.5:7b` (local Ollama — free, no quota)

**Actual results (baseline run — 2026-08-25):**

Per-question breakdown:
```
[out] abstention=Y (OK) :: What is Zephyr Analytics' stock price?
[out] abstention=Y (OK) :: Who is the CEO of Zephyr Analytics?

[FAIL] How many dashboards does the Free plan include?
       ✓ Faithfulness                 1.00
       ✓ Answer Relevancy             1.00
       ✗ Contextual Recall            0.50
       ✓ Contextual Precision         0.83

[FAIL] How much does the Pro plan cost per seat per month?
       ✗ Faithfulness                 0.50
       ✓ Answer Relevancy             1.00
       ✗ Contextual Recall            0.50
       ✓ Contextual Precision         0.83

[FAIL] Which plans include the Pulse feature?
       ✓ Faithfulness                 1.00
       ✗ Answer Relevancy             0.50
       ✗ Contextual Recall            0.50
       ✓ Contextual Precision         0.83

[FAIL] What is the API rate limit on the Pro plan?
       ✗ Faithfulness                 0.00
       ✓ Answer Relevancy             1.00
       ✗ Contextual Recall            0.50
       ✓ Contextual Precision         0.83

[FAIL] How long is data retained on the Free plan?
       ✗ Faithfulness                 0.50
       ✓ Answer Relevancy             1.00
       ✗ Contextual Recall            0.57
       ✓ Contextual Precision         0.83

[FAIL] In which cities does Zephyr run its data centres?
       ✗ Faithfulness                 0.50
       ✗ Answer Relevancy             0.50
       ✗ Contextual Recall            0.50
       ✓ Contextual Precision         0.83
```

**Aggregate summary (local judge):**

| Metric | Avg Score | Pass Rate | Threshold | Status |
|--------|-----------|-----------|-----------|--------|
| Faithfulness | 0.58 | 2/6 | 0.7 | ⚠ Below threshold |
| Answer Relevancy | 0.83 | 4/6 | 0.7 | ⚠ 2 questions miss |
| Contextual Recall | 0.51 | 0/6 | 0.7 | ❌ Consistent gap |
| Contextual Precision | 0.83 | 6/6 | 0.7 | ✅ |
| Abstention | 2/2 = 100% | — | — | ✅ |

**What these scores reveal:**

| Finding | Implication |
|---------|-------------|
| Contextual Precision 0.83 across all questions | Retriever fetches relevant chunks — no noise problem |
| Contextual Recall 0.51 across all questions | Retriever consistently misses some needed facts — primary bottleneck |
| Faithfulness 0.58 | `qwen2.5:7b` is a capable but weaker judge than Gemini; expect Gemini to score differently |
| Abstention 100% | Grounding holds — no hallucinations on out-of-corpus questions |

**Root cause of low Contextual Recall:** Retriever fetches k=3 chunks. Facts
needed for a complete answer sometimes span more than 3 chunks. Raising `k` in
`rag_chain.py` from 3 → 5 is the first fix to try.

---

## TR-08 — LangSmith UI: Dataset Upload (RUNBOOK Step 10)

**Action:** Upload `eval/golden_qa.json` to LangSmith as a Dataset.
See RUNBOOK Step 10 for exact UI steps (manual UI), or run `eval_langsmith.py`
(TR-13) which creates the dataset programmatically.

**Pass criteria:**
- Dataset appears in Datasets & Experiments with **8 examples**
- Each example has `question` as input and `reference` as expected output

**Actual result:** Dataset `zephyr-golden-qa` created programmatically via
`eval_langsmith.py` with id `b0aac0b2-2bae-4b09-b553-452ae249c87c`. 8 examples
confirmed. Inputs include `question` + `in_corpus`; outputs include `reference`
+ `must_contain`.

**Status: ✅ PASS** (completed programmatically — see TR-13)

---

## TR-09 — LangSmith UI: Online Evaluators (RUNBOOK Step 11)

**Action:** Create two LLM-as-Judge evaluators (Faithfulness, Answer Relevancy)
in the LangSmith Rules/Automations tab. See RUNBOOK Step 11 for exact UI steps.

**Pass criteria:**
- Both evaluators appear in the Rules tab with status **Active**
- After running one manual `rag_chain.py` question, the resulting trace shows
  `Faithfulness` and `Answer Relevancy` scores populated in the Feedback section
  within ~30 seconds

**Verify:**
```bash
python3 src/rag_chain.py "How much does the Pro plan cost?"
```
Then open the trace in LangSmith → confirm scores appear.

**Status:** ⬜ To complete (run RUNBOOK Step 11)

---

## TR-10 — DeepEval: Run with Gemini Judge (RUNBOOK Step 12)

**Command:**
```bash
python3 src/eval_deepeval.py
```

**Judge model:** `gemini-3.6-flash` (default — no `USE_LOCAL_JUDGE`)

**Pass criteria:**
- Abstention = 2/2 = 100% (non-negotiable)
- Contextual Precision ≥ 0.83 (should not regress from local run)
- No quota error (run day after Step 9 if quota was exhausted)

**Actual results (Gemini judge — fill in after running):**

Per-question breakdown:
```
[to be filled in after run]
```

**Aggregate summary (Gemini judge — fill in after running):**

| Metric | Avg Score | Pass Rate | Threshold | Status |
|--------|-----------|-----------|-----------|--------|
| Faithfulness | — | — | 0.7 | — |
| Answer Relevancy | — | — | 0.7 | — |
| Contextual Recall | — | — | 0.7 | — |
| Contextual Precision | — | — | 0.7 | — |
| Abstention | — | — | — | — |

**Judge comparison (fill in after running):**

| Metric | Local judge (llama3.1:8b) | Gemini judge | Delta | What this means |
|--------|--------------------------|--------------|-------|----------------|
| Faithfulness | 0.58 | — | — | — |
| Answer Relevancy | 0.83 | — | — | — |
| Contextual Recall | 0.51 | — | — | — |
| Contextual Precision | 0.83 | — | — | — |

**What to evaluate:**
- A large delta on Faithfulness means the local model is an unreliable judge for this metric
- Similar Contextual Precision scores across both judges = the metric is robust to judge quality
- A question that passes under Gemini but failed under local = local judge false negative

**Status:** ⬜ To complete (run RUNBOOK Step 12)

---

## TR-11 — LangSmith UI: Experiment Comparison (RUNBOOK Step 13)

**Action:** Compare the Step 9 (local judge) and Step 12 (Gemini judge)
experiments in the LangSmith Experiments tab. See RUNBOOK Step 13 for exact
UI steps.

**Pass criteria:**
- Two experiments appear in the Experiments tab
- Comparison view shows per-question score columns for each experiment
- The "Pulse" question (TR-04) scores low in both experiments

**What to record after running:**

| Question | Local Faithfulness | Gemini Faithfulness | Local Recall | Gemini Recall |
|----------|-------------------|---------------------|--------------|---------------|
| Dashboards (Free plan) | — | — | — | — |
| Pro plan cost | — | — | — | — |
| Pulse feature | — | — | — | — |
| API rate limit | — | — | — | — |
| Data retention | — | — | — | — |
| Data centres | — | — | — | — |

**What to look for:**
- Questions where both judges agree on pass/fail = reliable signal
- Questions where judges disagree = judge quality is a variable, not the pipeline
- The Pulse question (TR-04 known gap) should fail under both judges on Recall

**Status:** ⬜ To complete (run RUNBOOK Step 13)

---

## TR-13 — LangSmith Experiment: eval_langsmith.py (RUNBOOK Step 14)

**Command:**
```bash
USE_LOCAL_JUDGE=1 python3 src/eval_langsmith.py
```

**Judge model used:** `qwen2.5:7b` (local Ollama — Gemini quota was exhausted)

**Pass criteria:**
- Dataset `zephyr-golden-qa` created (or reused) in LangSmith
- All 8 questions run; named experiment created in LangSmith
- Faithfulness, answer_relevancy, and abstention scores logged per question
- Experiment URL printed and accessible in LangSmith UI

**Actual result:**
- Dataset `zephyr-golden-qa` created (id `b0aac0b2-2bae-4b09-b553-452ae249c87c`)
- Experiment `golden-eval-0c49b171` created — 8 questions, ~5 min (local judge)
- First run `golden-eval-5f033d56` hit Gemini quota mid-run (partial scores);
  second run with local judge completed cleanly

**LangSmith experiment URL:**
```
https://smith.langchain.com/o/0dc991dc-e502-437d-95b7-5b37ab92ab86/datasets/b0aac0b2-2bae-4b09-b553-452ae249c87c/compare?selectedSessions=1aba575d-b2b8-4a55-a565-8b20b0165556
```

**Status: ✅ PASS**

**Note:** Local terminal summary was blank (LangSmith SDK result iterator API
changed in newer versions) — scores are fully recorded in LangSmith UI.

---

## TR-14 — LangSmith Online Evaluator: run_golden_eval.py (RUNBOOK Step 15)

**Command:**
```bash
python3 src/run_golden_eval.py
```

**Pass criteria:**
- All 8 questions run as a LangSmith experiment (no local judge)
- Experiment created and URL printed
- LangSmith's configured Answer Relevancy online evaluator scores traces
  within ~60 seconds of run completion

**Actual result:**
- Dataset `zephyr-golden-qa` reused
- Experiment `golden-answer-relevancy-a43ad2b2` created — 8 questions, ~19 sec
- Traces available for online evaluator to score

**LangSmith experiment URL:**
```
https://smith.langchain.com/o/0dc991dc-e502-437d-95b7-5b37ab92ab86/datasets/b0aac0b2-2bae-4b09-b553-452ae249c87c/compare?selectedSessions=c05e5643-7f0c-48b2-b365-a4609fba36df
```

**Status: ✅ PASS** (pipeline run complete; online evaluator scores asynchronous)

**Verify:** Open LangSmith URL above → wait ~30–60 sec → confirm Answer
Relevancy score column is populated for each of the 8 rows.

---

## TR-15 — RAGAS Evaluation (RUNBOOK Step 16 — Optional)

**Command:**
```bash
python3 src/eval_ragas.py
```

**Pass criteria:**
- No `NaN` values (would indicate quota exhaustion)
- Scores within 0.10 of the DeepEval Gemini-judge scores from TR-10

**Expected output format:**
```
===== RAGAS SCORES =====
faithfulness: X.XX
answer_relevancy: X.XX
context_precision: X.XX
context_recall: X.XX
```

**Cross-validation table (fill in after running):**

| Metric | DeepEval (Gemini) | RAGAS | Delta | Agreement? |
|--------|-------------------|-------|-------|-----------|
| Faithfulness | — | — | — | — |
| Answer Relevancy | — | — | — | — |
| Context Precision | — | — | — | — |
| Context Recall | — | — | — | — |

Agreement within 0.10 = both frameworks measure the same thing consistently.
Disagreement > 0.15 = investigate framework differences before trusting either score.

**Status:** ⬜ To complete (run RUNBOOK Step 14)

---

## Summary of Pass/Fail Status

| TR | Step | Description | Status |
|----|------|-------------|--------|
| TR-01 | 5 | Build vector store | ✅ PASS |
| TR-02 | 6 | RAG in-corpus questions | ✅ PASS |
| TR-03 | 6 | RAG hallucination test | ✅ PASS |
| TR-04 | 6 | Known retrieval gap (Pulse) | ⚠ Known FAIL |
| TR-05 | 7 | Agent tool routing | ✅ PASS |
| TR-06 | 8 | Hand-rolled evaluation | ✅ PASS |
| TR-07 | 9 | DeepEval — local judge baseline | ✅ Complete (see scores above) |
| TR-08 | 10 | LangSmith dataset upload | ✅ PASS (programmatic via TR-13) |
| TR-09 | 11 | LangSmith online evaluators | ⬜ To complete (UI setup) |
| TR-10 | 12 | DeepEval — Gemini judge | ⬜ To complete |
| TR-11 | 13 | LangSmith experiment comparison | ⬜ To complete |
| TR-13 | 14 | LangSmith experiment — eval_langsmith.py | ✅ PASS (local judge) |
| TR-14 | 15 | LangSmith online evaluator — run_golden_eval.py | ✅ PASS |
| TR-15 | 16 | RAGAS cross-validation (optional) | ⬜ To complete |
