# Execution Runbook

Step-by-step guide to running the RAG Evaluation Lab from scratch. Each step
states its **purpose**, exact **action**, **expected output**, and how to
**evaluate / verify** success.

Work through steps in order — each depends on the one before it. Steps 10, 11,
and 13 are performed in the LangSmith web UI (no code required).

---

## Step 0 — Prerequisites Check

**Purpose:** Confirm the required tools are installed before starting anything else.

**Action:**
```bash
ollama --version
python3 --version
```

**Expected output:**
- `ollama` prints a version string, e.g. `ollama version 0.x.x`
- `python3` prints `Python 3.10.x` or higher

**Evaluate / Verify:**
Both commands return version numbers with no errors.

**If this fails:**
- Ollama missing → download from https://ollama.com
- Python < 3.10 → install from https://python.org

---

## Step 1 — Start Ollama

**Purpose:** Ollama must be running as a background HTTP server on port 11434
before any script can use it for embeddings or generation.

**Action:**
```bash
ollama serve
```

Leave this terminal open. Open a **new terminal** for all remaining steps.

**Expected output:**
```
Listening on 127.0.0.1:11434
```

**Evaluate / Verify:**
```bash
curl http://localhost:11434
# Expected response: Ollama is running
```

**If this fails:** Port 11434 may already be in use by another Ollama process.
Run `pkill ollama` then retry.

---

## Step 2 — Pull Required Models

**Purpose:** Download the two models this project uses. One converts text into
vectors (embedding model); the other writes answers (generation model).
These are downloaded once and cached locally.

**Action:**
```bash
ollama pull llama3.1:8b         # ~4.7 GB — generation model
ollama pull nomic-embed-text    # ~275 MB — embedding model
```

Each command shows a download progress bar.

**Expected output:**
Both commands end with `success`.

**Evaluate / Verify:**
```bash
ollama list
# Both models must appear in the list
```

**Key concept:** RAG uses *two* separate models — an embedding model to index
and search the vector store, and a generation model to write the final answer.
Confusing these two is the most common beginner mistake.

---

## Step 3 — Python Environment

**Purpose:** Create an isolated Python environment and install all project
dependencies so scripts run without version conflicts.

**Action:**
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Your terminal prompt should now show `(.venv)`.

**Expected output:**
`pip install` ends with `Successfully installed ...` followed by a list of packages.

**Evaluate / Verify:**
```bash
python3 -c "import langchain, ragas, langgraph, deepeval; print('OK')"
# Expected: OK
```

**If this fails:** Run `pip install --upgrade pip` first, then retry the install.

---

## Step 4 — Configure API Keys

**Purpose:** Set credentials for Gemini (LLM judge for evaluations) and
LangSmith (tracing + evaluation UI). LangSmith is optional but strongly
recommended — it makes every pipeline step visible and enables the UI
comparison in Steps 10–13.

**Action:**
```bash
cp .env.example .env
```

Open `.env` in a text editor and fill in the following:

| Variable | Where to get it | Required? |
|----------|----------------|-----------|
| `GOOGLE_API_KEY` | https://aistudio.google.com/apikey | **Yes** — eval scripts use Gemini as judge |
| `LANGSMITH_API_KEY` | https://smith.langchain.com → Settings → API Keys | **Recommended** — enables tracing and UI comparison |
| `LANGSMITH_TRACING` | Set to `true` | Recommended — activates automatic tracing |
| `LANGSMITH_PROJECT` | Set to `rag-eval-lab` | Recommended — keeps all traces in one project |

Load the file into your current shell session:
```bash
set -a && source .env && set +a
```

**Evaluate / Verify:**
```bash
echo $GOOGLE_API_KEY        # should print your key (not blank)
echo $LANGSMITH_API_KEY     # should print your key (not blank)
```

**If either prints blank:** The `.env` file did not load. Re-run the
`set -a && source .env && set +a` command and verify again.

---

## Step 5 — Build the Vector Store

**Purpose:** Chunk the handbook into searchable pieces, embed each chunk, and
save to ChromaDB on disk. This is the "index" that all retrieval steps search
against. Run once; re-run only if you change the source data or chunk settings.

**Action:**
```bash
python3 src/ingest.py
```

**What happens inside:**
1. Loads `data/zephyr_handbook.md` — a fictional product handbook the LLM has
   never seen in training, so all correct answers *must* come from retrieval
2. Splits the document into 400-character chunks with 80-character overlap
3. Embeds each chunk using `nomic-embed-text` via Ollama
4. Saves the vector store to `./chroma_db/`

**Expected output:**
```
Loaded 1 document(s), 2478 characters.
Split into 9 chunks.
Embedded and stored 9 chunks in ./chroma_db.
Ingestion complete. You can now run src/rag_chain.py
```

**Evaluate / Verify:**
```bash
ls chroma_db/
# Must contain chroma.sqlite3 and at least one UUID-named folder
```

**If this fails:** Ollama is not running or `nomic-embed-text` was not pulled.
Check Steps 1 and 2.

**Tuning experiment (try later):** Change `chunk_size=400` to `chunk_size=100`
in `ingest.py`, re-run, then re-run `rag_chain.py` on the same question. Answer
quality typically drops — this demonstrates chunk size as a tunable variable.
Reset to 400 when done.

---

## Step 6 — Test the RAG Pipeline

**Purpose:** Manually verify the end-to-end RAG pipeline before running
automated evaluations. Builds intuition for what retrieval + generation looks
like, and what correct vs. incorrect behaviour feels like.

**Action:**
```bash
python3 src/rag_chain.py "How much does the Pro plan cost?"
python3 src/rag_chain.py "What is the API rate limit on the Pro plan?"
python3 src/rag_chain.py "What is the capital of France?"
```

**Expected output per question:**

| Question | Expected answer | Why this matters |
|----------|----------------|-----------------|
| Pro plan cost? | `$49 per seat per month` | Basic in-corpus retrieval |
| API rate limit on Pro? | `1,000 requests per minute` | Distractor test — Free=100, Enterprise=10,000 |
| Capital of France? | `I don't know based on the handbook.` | Grounding test — not in corpus |

Below each answer the script prints the 3 retrieved chunks the model was given.

**Evaluate / Verify:**
1. The France question **must** say `"I don't know based on the handbook."` —
   any other response means the grounding prompt is broken; investigate before continuing
2. Read the printed chunks for each in-corpus question — if the right fact
   appears in the chunks but the answer is wrong, the generation model is at
   fault; if the right fact is absent from all chunks, retrieval is at fault

**View the trace in LangSmith (if LANGSMITH_API_KEY is set):**
1. Go to https://smith.langchain.com → **Projects** → **rag-eval-lab**
2. Click the most recent trace (one per question you just ran)
3. Expand the trace tree:
   ```
   ▶ RunnableSequence
     ├── retrieve          → shows the 3 chunks fetched from chroma_db
     ├── ChatPromptTemplate → the grounding prompt with context filled in
     ├── ChatOllama        → the LLM call (exact input prompt + raw output)
     └── StrOutputParser   → the final answer text
   ```
4. Click the **retrieve** step → confirm the correct fact chunk is present
5. Click the **ChatOllama** step → see the exact prompt the model received

This trace view is the most important debugging tool in RAG systems.

---

## Step 7 — Test the Agent

**Purpose:** Verify the LangGraph ReAct agent correctly routes questions to the
right tool, including multi-step reasoning that combines retrieval and
calculation in sequence.

**Action:**
```bash
python3 src/agent.py "What is the API rate limit on the Pro plan?"
python3 src/agent.py "What is 2 + 2?"
python3 src/agent.py "I need 90 extra days of retention. What will it cost?"
```

**Expected output per question:**

| Question | Tool(s) called | Expected answer |
|----------|---------------|----------------|
| API rate limit? | `search_handbook` | 1,000 requests per minute |
| What is 2 + 2? | `calculator` | 4 |
| 90 days retention cost? | `search_handbook` → `calculator` | $45.00 |

For the third (multi-step) question, the agent must:
1. Call `search_handbook` → retrieves: add-on costs $15 per 30 days
2. Call `calculator` with `90 / 30 * 15` → returns `45.0`
3. Report the final answer as `$45.00`

**Evaluate / Verify:**
- Correct final answer for all three questions
- In LangSmith, open the agent trace → confirm both `search_handbook` AND
  `calculator` appear as separate tool call nodes for the third question
- If the agent only called one tool and still got the right answer, it may have
  guessed — the trace will reveal which tools actually fired

**Known warning (non-blocking):**
```
LangGraphDeprecatedSinceV10: create_react_agent has been moved to langchain.agents.
```
This deprecation warning does not affect correctness.

---

## Step 8 — Hand-Rolled Evaluation

**Purpose:** Run a custom evaluation harness that measures four metrics without
any external framework. Establishes a baseline and teaches the core ideas behind
automated evaluation before introducing heavier tools.

**Action:**
```bash
python3 src/eval_custom.py
```

**What it measures:**

| Metric | What it checks |
|--------|---------------|
| Retrieval hit rate | Did any retrieved chunk contain the expected answer keyword? |
| Keyword correctness | Does the generated answer contain the expected keywords? |
| Abstention | Did the model say "I don't know" for out-of-corpus questions? |
| LLM-as-judge | Does Gemini grade the answer as faithful to the retrieved context? |

**Expected output:**
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

**Evaluate / Verify:**
- `Abstention = 2/2` is non-negotiable — any hallucination on an out-of-corpus
  question is a FAIL requiring investigation
- `Retrieval hit rate = 6/6` should be 100%; a miss indicates a broken vector
  store or embedding problem
- `Retrieval hit rate > Keyword correctness` → retriever found the right chunk
  but generator produced a wrong answer
- `Retrieval hit rate = Keyword correctness` but LLM-judge fails → judge
  disagrees with the keyword check; check the judge prompt
- The "Pulse" question (`Which plans include the Pulse feature?`) is an
  expected FAIL — documented as a known retrieval gap in TEST_RUNBOOK TR-04

**LangSmith traces:** Each of the 8 questions creates a trace in the
`rag-eval-lab` project. You can inspect which chunks each question retrieved
without changing any code.

**Variance experiment:** Change `REPEATS = 1` to `REPEATS = 3` at the top of
`eval_custom.py` and re-run. If the LLM-judge score shifts between runs, you
have observed why a single eval run is not enough to trust.

---

## Step 9 — DeepEval: Baseline Run with Local Judge

**Purpose:** Run DeepEval's four RAG metrics using the local `llama3.1:8b`
model as judge (free, no quota). This is your baseline — you will compare it
against the Gemini-judged run in Step 12 to understand how much judge quality
affects scores.

**Action:**
```bash
USE_LOCAL_JUDGE=1 python3 src/eval_deepeval.py
```

**What it measures:**

| Metric | What it checks |
|--------|---------------|
| Faithfulness | Is every claim in the answer supported by the retrieved context? |
| Answer Relevancy | Does the answer actually address the question that was asked? |
| Contextual Recall | Does the retrieved context contain all facts needed for the reference answer? |
| Contextual Precision | Are the retrieved chunks on-topic (no irrelevant noise)? |

**Expected output (baseline — your numbers may differ slightly):**
```
[judge] Using local Ollama model: llama3.1:8b
Running RAG pipeline over golden dataset...

[out] abstention=Y (OK) :: What is Zephyr Analytics' stock price?
[out] abstention=Y (OK) :: Who is the CEO of Zephyr Analytics?

===== PER-QUESTION RESULTS =====
[FAIL] How many dashboards does the Free plan include?
       ✓ Faithfulness                 1.00
       ✓ Answer Relevancy             1.00
       ✗ Contextual Recall            0.50
       ✓ Contextual Precision         0.83
...

===== SUMMARY =====
Faithfulness          avg: 0.58  pass: 2/6  (threshold: 0.7)
Answer Relevancy      avg: 0.83  pass: 4/6  (threshold: 0.7)
Contextual Recall     avg: 0.51  pass: 0/6  (threshold: 0.7)
Contextual Precision  avg: 0.83  pass: 6/6  (threshold: 0.7)
Abstention (no hallucination): 2/2 = 100%
```

**Evaluate / Verify:**
- **Contextual Precision ≥ 0.83** — retriever fetches relevant chunks; this is good
- **Contextual Recall ≈ 0.51** — retriever consistently misses some needed
  facts; this is the primary problem (root cause: k=3 may not fetch enough chunks)
- **Faithfulness ≈ 0.58** — local model adds embellishments beyond the context;
  expect Gemini judge to score this metric differently in Step 12
- **Record these numbers** in TEST_RUNBOOK TR-07 — you will compare them
  against the Gemini run in Step 12

**Note on local judge quality:** `llama3.1:8b` is a weaker judge than Gemini.
It may evaluate faithfulness inconsistently. Steps 10–13 use LangSmith to get
authoritative, Gemini-judged scores.

---

## Step 10 — LangSmith UI: Upload the Golden Dataset

**Purpose:** Upload the 8 golden QA pairs into LangSmith so every future eval
run is tracked as a named experiment and can be compared side by side in the UI.
You only do this once — the dataset persists in LangSmith.

**Prerequisite:** LANGSMITH_API_KEY set in `.env` (Step 4). Logged in at
https://smith.langchain.com.

**Action (UI steps):**

1. Go to https://smith.langchain.com
2. In the left sidebar, click **Datasets & Experiments**
3. Click **+ New Dataset** (top right button)
4. Fill in:
   - **Name:** `rag-eval-golden`
   - **Description:** `Golden QA pairs for Zephyr Analytics RAG evaluation`
   - **Dataset type:** leave as default (key-value)
5. Click **Create Dataset**
6. On the dataset detail page, click **+ Add examples** → **Upload file**
7. Upload the file `eval/golden_qa.jsonl` from your local machine
8. In the field mapping step:
   - Set **Input** to `question`
   - Set **Output** (expected) to `reference`
   - Other fields (`in_corpus`, `must_contain`, `note`) set as metadata
9. Click **Submit** / **Confirm**

**Evaluate / Verify:**
- The dataset page shows **8 examples**
- Click any example row — the `question` field shows as input and `reference`
  shows as expected output
- The `in_corpus` flag is visible in the metadata panel

**What this unlocks:** Every eval run (local judge, Gemini judge, k=5 tuning run)
can now be run as a separate "Experiment" against this dataset and compared side
by side in the LangSmith Experiments tab — no extra code required.

---

## Step 11 — LangSmith UI: Set Up Online Evaluators

**Purpose:** Configure LangSmith to automatically score every RAG trace that
lands in your project using an LLM judge. Scores appear alongside traces
within ~30 seconds of each run — no code changes needed.

**Prerequisite:** Steps 4 and 10 complete. LangSmith must have access to your
Gemini key (it uses your connected account or you paste the key in the evaluator
settings).

**Action (UI steps):**

1. Go to https://smith.langchain.com → **Projects** → **rag-eval-lab**
2. Click the **Rules** tab (may appear as **Automations** or **Evaluators**
   depending on your LangSmith version)
3. Click **+ New Rule** → choose **LLM-as-Judge**
4. Configure the first evaluator — **Faithfulness**:
   - **Name:** `Faithfulness`
   - **Prompt template:**
     ```
     You are evaluating a RAG system answer.

     Question: {{input}} 
     Answer: {{output}}

     Score from 0.0 to 1.0: is every claim in the answer directly supported
     by the retrieved context, with nothing added beyond what the context says?

     Respond with only a JSON object: {{"Score": <number between 0 and 1>}}
     ```
   - **Model:** select a Gemini model (e.g. Gemini 2.0 Flash)
   - **Response Format field:** `score`
   - **Sampling rate:** 100% (score every trace)
   - **Note:Map the Question to the input field (based on Dataset Config) and the Answer to the reference output field (based on Dataset Config)and score to the Score field from Feedback configuration.**
5. Click **Save**
6. Repeat to add a second evaluator — **Answer Relevancy**:
   - **Name:** `Answer Relevancy`
   - **Prompt template:**
     ```
     Question: {{input}} 
     Answer: {{output}}

     Score from 0.0 to 1.0: does the answer directly and completely address
     the question? A score of 1.0 means fully on-topic and complete.

     Respond with only a JSON object: {{"score": <number between 0 and 1>}}
     ```
   - Same model and sampling settings
   - **Model:** select a Gemini model (e.g. Gemini 2.0 Flash)
   - **Response Format field:** `score`
   - **Sampling rate:** 100% (score every trace)
   - **Note:Map the Question to the input field (based on Dataset Config) and the Answer to the reference output field (based on Dataset Config)and score to the Score field from Feedback configuration.**
   - Click **Save**
**Evaluate / Verify:**
- Both evaluators appear in the Rules/Automations tab with status **Active**
- Run one manual question to trigger a trace:
  ```bash
  python3 src/rag_chain.py "How much does the Pro plan cost?"
  ```
- Go to **Projects** → **rag-eval-lab** → **Traces**
- Open the new trace → scroll to the **Feedback** or **Scores** section
- Within ~30 seconds you should see `Faithfulness` and `Answer Relevancy`
  scores populated by the evaluators

**If scores don't appear:** Check that the evaluator is set to **Active** and
that your Gemini credentials are connected in LangSmith settings.

---

## Step 12 — DeepEval: Run with Gemini Judge

**Purpose:** Re-run the same DeepEval metrics from Step 9, this time using
Gemini as the judge. This is the authoritative baseline — Gemini's evaluation
is more consistent than the local model. Compare the results against Step 9
to understand how judge quality affects scores.

**Prerequisite:** `GOOGLE_API_KEY` set in `.env` and Gemini free-tier quota
available (20 requests/day on `gemini-3.6-flash`). If quota was exhausted
during Step 8, run this the next day.

**Action:**
```bash
python3 src/eval_deepeval.py
```

(No `USE_LOCAL_JUDGE=1` — Gemini is the default.)

**Expected output format:**
```
[judge] Using Gemini: gemini-3.6-flash
Running RAG pipeline over golden dataset...
...
===== SUMMARY =====
Faithfulness          avg: X.XX  pass: N/6  (threshold: 0.7)
Answer Relevancy      avg: X.XX  pass: N/6  (threshold: 0.7)
Contextual Recall     avg: X.XX  pass: N/6  (threshold: 0.7)
Contextual Precision  avg: X.XX  pass: N/6  (threshold: 0.7)
Abstention (no hallucination): 2/2 = 100%
```

**Evaluate / Verify:**
1. Record the summary scores in TEST_RUNBOOK TR-10
2. Compare each metric against the Step 9 (local judge) results:
   - **Faithfulness:** expect Gemini to score differently from local — a large
     gap means local judge is unreliable for this metric
   - **Contextual Precision:** expect similar results — this metric is more
     objective and less sensitive to judge quality
   - **Contextual Recall:** may shift — Gemini understands nuanced gaps better
3. Go to LangSmith → **Projects** → **rag-eval-lab** → **Traces** — the 6
   in-corpus questions each created a new trace; the online evaluators from
   Step 11 score them automatically

**If quota error appears:**
```
ResourceExhausted: 429 Resource has been exhausted
```
Set `USE_LOCAL_JUDGE=1` temporarily and try again tomorrow for the Gemini run.

---

## Step 13 — LangSmith UI: Compare Experiments

**Purpose:** View the local-judge run (Step 9) and the Gemini-judge run (Step 12)
side by side in LangSmith. The comparison view shows per-question scores for
each metric across both runs — this is where you identify which failures are
pipeline problems vs. judge quality problems.

**Prerequisite:** Steps 9 and 12 both complete, LANGSMITH_API_KEY set.

**Action (UI steps):**

1. Go to https://smith.langchain.com → **Projects** → **rag-eval-lab**
2. Click the **Experiments** tab in the left sidebar or project nav
3. You should see at least two experiment rows — one from each DeepEval run
4. Check the checkbox next to both experiments
5. Click **Compare** (button appears in the top right when two are selected)

**What to look for in the comparison view:**

| Thing to check | What it means |
|---------------|---------------|
| "Pulse" question scores low in both runs | Real pipeline problem — retrieval gap (see TEST_RUNBOOK TR-04) |
| Faithfulness differs significantly between runs | Judge quality is the variable, not the pipeline |
| Contextual Precision similar in both runs | This metric is objective; good signal regardless of judge |
| A question passes in one run but fails in the other | Local judge is unreliable for that metric |

**Evaluate / Verify:**
- At least two experiments appear in the list
- The comparison table shows per-question scores in columns, one per experiment
- You can click any row to drill into the individual trace for that question

**What you are learning:** When two judges disagree on a score, the
higher-quality judge (Gemini) is more likely correct. The comparison tells you
which metrics to trust from the local-judge run and which required a stronger judge
to evaluate reliably.

---

## Step 14 — RAGAS Evaluation (Optional / Cross-Validation)

**Purpose:** Run a second framework (RAGAS) on the same four metrics to
cross-validate the DeepEval scores from Steps 9 and 12. If RAGAS and DeepEval
agree within 0.10, the signal is reliable. If they disagree by more than 0.15,
investigate — different frameworks define metrics subtly differently.

**Action:**
```bash
python3 src/eval_ragas.py
```

**Expected output (scores between 0 and 1):**
```
===== RAGAS SCORES =====
faithfulness: X.XX
answer_relevancy: X.XX
context_precision: X.XX
context_recall: X.XX
```

**Evaluate / Verify:**
- Compare each score against the corresponding DeepEval metric from Step 12
- Agreement within 0.10 → signal is reliable for that metric
- Disagreement > 0.15 → one framework or its judge calls are hitting a limit or
  interpreting the metric differently; check for `NaN` values (quota issue)

**If you see `NaN`:** Gemini quota is exhausted. Set `USE_LOCAL_JUDGE=1` in
`.env` or wait for the quota to reset.

---

## Step 15 — Extend the Golden Dataset

**Purpose:** Practice writing your own evaluation cases — the skill that
transfers to every AI system you will test in future.

**Action:**
Open `eval/golden_qa.json` and add at least three new entries:

1. An **in-corpus** question about a fact in the handbook not yet tested
2. A **multi-fact** question that requires two separate handbook sections to
   answer correctly
3. An **out-of-corpus trap** — a plausible question with no answer in the handbook

Then re-run Steps 8 and 9 and observe how scores change.

**Evaluate / Verify:**
- New in-corpus questions: retrieval should succeed if the fact is clearly
  stated in the handbook
- New out-of-corpus questions: model must abstain (`"I don't know based on the
  handbook."`) — if it does not, the grounding prompt needs investigation
- Update the LangSmith dataset (Step 10) with the new examples so future
  experiments run against the full set

**What you are learning:** Deciding *what* to test is harder than running the
metrics. An eval suite is only as good as its cases.

---

## Quick Reference

| Step | Action | Runs in | Requires |
|------|--------|---------|---------|
| 0 | Check tools | Terminal | — |
| 1 | `ollama serve` | Terminal (keep open) | — |
| 2 | `ollama pull llama3.1:8b && ollama pull nomic-embed-text` | Terminal | Ollama running |
| 3 | `pip install -r requirements.txt` | Terminal | Python 3.10+ |
| 4 | Fill in `.env` | Text editor | Google + LangSmith accounts |
| 5 | `python3 src/ingest.py` | Terminal | Ollama running |
| 6 | `python3 src/rag_chain.py "..."` | Terminal | `chroma_db/` built |
| 7 | `python3 src/agent.py "..."` | Terminal | `chroma_db/` built |
| 8 | `python3 src/eval_custom.py` | Terminal | Ollama + `GOOGLE_API_KEY` |
| 9 | `USE_LOCAL_JUDGE=1 python3 src/eval_deepeval.py` | Terminal | Ollama |
| 10 | Upload `eval/golden_qa.json` | LangSmith UI | `LANGSMITH_API_KEY` |
| 11 | Create online evaluators | LangSmith UI | `LANGSMITH_API_KEY` + `GOOGLE_API_KEY` |
| 12 | `python3 src/eval_deepeval.py` | Terminal | `GOOGLE_API_KEY` (quota needed) |
| 13 | Compare experiments | LangSmith UI | Steps 9 + 12 done |
| 14 | `python3 src/eval_ragas.py` (optional) | Terminal | `GOOGLE_API_KEY` |
| 15 | Edit `eval/golden_qa.json` | Text editor | — |
