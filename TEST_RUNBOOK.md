# Test Runbook — RAG + Agent + Eval Lab

**Assumption:** Setup is complete — Ollama is running, models are pulled, `.venv`
is active, and `.env` is loaded with `GOOGLE_API_KEY` and `LANGSMITH_API_KEY`.

All results below are from an actual run on this codebase. Use them as your
baseline: if your numbers differ significantly, something in the pipeline
has changed or degraded.

---

## TR-01 — Build the Vector Store

**Command:**
```bash
python3 src/ingest.py
```

**Expected output:**
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

**Pass criteria:**
- Exactly 9 chunks reported
- `chroma_db/` directory created with at least `chroma.sqlite3`

**If this fails:** Ollama is not running or `nomic-embed-text` model was not
pulled. Run `ollama list` to confirm.

---

## TR-02 — RAG Pipeline: In-Corpus Questions

**Command:**
```bash
python3 src/rag_chain.py "How much does the Pro plan cost?"
python3 src/rag_chain.py "What is the API rate limit on the Pro plan?"
```

**Actual results:**

| Question | Expected answer | Retrieved chunk [1] |
|----------|----------------|---------------------|
| Pro plan cost? | `The Pro plan costs $49 per seat per month.` | Plans and Limits |
| API rate limit on Pro? | `1,000 requests per minute.` | API Access |

**Pass criteria:**
- Answer matches expected (exact wording may vary)
- Retrieved context [1] contains the relevant section

**LangSmith:** Each run creates a trace in the **Rag-Eval-Lab** project showing
retrieve → prompt → generate steps. Open the trace to confirm chunk [1] held
the fact that produced the answer.

---

## TR-03 — RAG Pipeline: Out-of-Corpus (Hallucination Test)

**Command:**
```bash
python3 src/rag_chain.py "What is the capital of France?"
```

**Actual result:**
```
Q: What is the capital of France?
A: I don't know based on the handbook.

--- retrieved context ---
[1] ## About Zephyr ...
[2] # Zephyr Analytics — Product Handbook ...
[3] ## The Pulse Feature ...
```

**Pass criteria:**
- Answer must contain `"I don't know based on the handbook"` — any other
  response is a hallucination and a **FAIL**
- Retrieved chunks will be irrelevant (correct — the question is not in the corpus)

**Why this matters:** The grounding prompt is your only defence against the
model using its own training knowledge. If this test fails, the prompt is broken.

---

## TR-04 — RAG Pipeline: Known Retrieval Gap

**Command:**
```bash
python3 src/rag_chain.py "Which plans include the Pulse feature?"
```

**Actual result:**
```
Q: Which plans include the Pulse feature?
A: The Free plan does not include the Pulse feature.
   I don't know based on the handbook.

--- retrieved context ---
[1] ## Plans and Limits   ← mentions Free plan excludes Pulse
[2] The Enterprise plan includes ...
[3] ## Data Export ...
```

**Expected answer (per golden dataset):** `Pulse is available on the Pro and
Enterprise plans only.`

**Status: KNOWN FAIL — retrieval gap**

The explicit answer lives in the `## The Pulse Feature` section of the handbook.
The retriever consistently fetches the `## Plans and Limits` chunk instead, which
only states that the Free plan does *not* include Pulse. The model cannot infer
the positive from the negative with the chunks it receives.

**Root cause:** Embedding similarity favours the Plans section (it mentions Pulse
in context of all plans) over the Pulse Feature section (which states availability
directly). This is a **retrieval precision issue**, not a generation issue.

**Fix options:** Raise `k` from 3 to 4 to 5 (more chunks fetched, higher chance the pulse section is included), or re-chunk with a smaller overlap so the Pulse Feature section
is isolated in its own chunk (Better Precision- the explicit statement is always retrieved together).

---

## TR-05 — Agent: Tool Routing

**Commands:**
```bash
python3 src/agent.py "What is the API rate limit on the Pro plan?"
python3 src/agent.py "What is 2 + 2?"
python3 src/agent.py "I need 90 extra days of retention. What will it cost?"
```

**Actual results:**

| Question | Tool(s) called | Expected answer | Result |
|----------|---------------|----------------|--------|
| API rate limit on Pro? | `search_handbook` | 1,000 requests per minute | ✅ PASS |
| What is 2 + 2? | `calculator` | 4 | ✅ PASS |
| 90 extra days retention cost? | `search_handbook` → `calculator` | $45.00 | ✅ PASS |

**Pass criteria for the multi-step question:**
- Agent must call `search_handbook` first (retrieves: add-on costs $15 per 30 days)
- Then call `calculator` with expression `90 / 30 * 15`
- Final answer must be `$45`

**LangSmith:** The agent trace shows each tool invocation as a separate node.
If the answer is correct but only one tool fired, it was a lucky guess — the
trace tells you which.

**Known warning (non-blocking):**
```
LangGraphDeprecatedSinceV10: create_react_agent has been moved to langchain.agents.
```
This is a deprecation notice for a future LangGraph release. Code still works correctly.

---

## TR-06 — Hand-Rolled Evaluation (`eval_custom.py`)

**Command:**
```bash
python3 src/eval_custom.py
```

**Actual results (baseline run):**

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

**Baseline scores:**

| Metric | Score | Status |
|--------|-------|--------|
| Retrieval hit rate | 6/6 = 100% | ✅ |
| Keyword correctness | 5/6 = 83% | ✅ (1 known gap — TR-04) |
| LLM-judge faithful | 5/6 = 83% | ✅ (1 known gap — TR-04) |
| Abstention | 2/2 = 100% | ✅ |

**The one consistent FAIL:** "Which plans include the Pulse feature?" — retrieval
miss documented in TR-04. All other questions PASS across all metrics.

**Pass criteria for a healthy run:**
- Retrieval hit rate ≥ 100% (must not drop — any miss is a regression)
- Keyword correctness ≥ 83% (baseline; the Pulse gap is expected)
- LLM-judge faithful ≥ 83% (baseline; note: judge uses fixed sampling on
  `gemini-3.6-flash` so scores may vary slightly run to run)
- Abstention = 100% (non-negotiable — any hallucination is a **FAIL**)

**Variance experiment:** Change `REPEATS = 1` to `REPEATS = 3` in
`eval_custom.py` and re-run. If the judge score moves between runs, you have
observed why single-run AI testing is unreliable.

---

## TR-07 — RAGAS Evaluation (`eval_ragas.py`)

*(Pending — blocked by Gemini free-tier quota; run tomorrow)*

**Command:**
```bash
python3 src/eval_ragas.py
```

**Metrics computed:** faithfulness, answer_relevancy, context_precision,
context_recall

**Expected score range (0–1, higher is better):** to be filled in after first run.

**How to read scores:**

| Score | If low, it means... | Action |
|-------|---------------------|--------|
| `faithfulness` | Model is generating beyond the context | Tighten grounding prompt |
| `answer_relevancy` | Answer wanders off the question | Review prompt or raise `k` |
| `context_precision` | Retriever pulls irrelevant chunks | Lower `k` or improve chunking |
| `context_recall` | Retriever misses needed facts | Raise `k` or re-chunk |

**Cross-check:** Compare `faithfulness` against the LLM-judge score from TR-06.
They measure the same thing with different methods. If they agree, the signal
is reliable. If they disagree by more than 15%, investigate which one is wrong.

---

## TR-08 — DeepEval Evaluation (`eval_deepeval.py`)

**Command (local judge — no quota needed):**
```bash
USE_LOCAL_JUDGE=1 python3 src/eval_deepeval.py
```

**Command (Gemini judge — requires quota):**
```bash
python3 src/eval_deepeval.py
```

**Judge model used for baseline below:** `llama3.1:8b` (local Ollama)
Set `USE_LOCAL_JUDGE=1` in `.env` to use local judge when Gemini free-tier
quota (20 req/day for `gemini-3.6-flash`) is exhausted.

**Actual results — baseline run (local judge `llama3.1:8b`, 2026-08-25):**

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

**Baseline scores (local judge):**

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
| Faithfulness 0.58 | `llama3.1:8b` adds embellishments beyond the context; tighter prompt may help |
| Abstention 100% | Grounding holds — no hallucination on out-of-corpus questions |

**Root cause of low Contextual Recall:** Retriever fetches k=3 chunks.
Facts needed for full answers sometimes span more than 3 chunks.
Try raising `k` in [`src/rag_chain.py`](src/rag_chain.py) from 3 → 5.

**Note on local vs Gemini judge:** `llama3.1:8b` is a weaker judge than
Gemini — scores may be noisier and thresholds harder to pass. Re-run with
Gemini judge to get the authoritative baseline (see pending task below).

---

## PENDING — Re-run TR-08 with Gemini Judge

**Status:** ⏳ Blocked — Gemini free-tier quota (`gemini-3.6-flash`, 20 req/day)
exhausted on 2026-08-25. Quota resets daily.

**Action (run on 2026-08-26 or later):**
```bash
python3 src/eval_deepeval.py
```

**What to do after running:**
1. Fill in the Gemini-judge scores in the table below
2. Compare Faithfulness score against TR-06 LLM-judge (83%) — if they agree
   within 15%, the signal is reliable
3. Update the "baseline" note in TR-08 to reflect the Gemini run

**Gemini-judge results (to be filled in):**

| Metric | Avg Score | Pass Rate | Status |
|--------|-----------|-----------|--------|
| Faithfulness | — | — | — |
| Answer Relevancy | — | — | — |
| Contextual Recall | — | — | — |
| Contextual Precision | — | — | — |
| Abstention | — | — | — |
