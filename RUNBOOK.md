# Execution Runbook

A step-by-step guide to running the project. Each step tells you what to run,
what to expect, what to check, and what you are learning.

Work through these in order — each step depends on the one before it.

---

## Step 0 — Prerequisites check

Before anything else, confirm your tools are in place.

```bash
ollama --version          # should print a version number
python3 --version          # should be 3.10+
gh --version              # optional, only needed if you want to push to GitHub
```

**What to expect:** Three version strings, no errors.

**If Ollama is missing:** Download from https://ollama.com and install.

---

## Step 1 — Start Ollama

Ollama must be running as a background server before any script can use it.

```bash
ollama serve
```

Leave this terminal open. Open a new terminal for everything else.

**What to expect:** A log line like `Listening on 127.0.0.1:11434`.

**Verify:**
```bash
curl http://localhost:11434
# should print: Ollama is running
```

---

## Step 2 — Pull the models

```bash
ollama pull llama3.1:8b        # ~4.7 GB — the generation model
ollama pull nomic-embed-text   # ~275 MB — the embedding model
```

These are one-time downloads. Each prints a progress bar.

**What to expect:** Both end with `success`.

**Verify:**
```bash
ollama list
# should show both llama3.1:8b and nomic-embed-text
```

**Learning note:** RAG uses *two* models — one to turn text into vectors
(embedding), one to generate answers. Confusing these is the #1 beginner error.

---

## Step 3 — Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**What to expect:** `pip install` prints a list of packages ending with
`Successfully installed ...`. Your prompt should show `(.venv)`.

**Verify:**
```bash
python3 -c "import langchain, ragas, langgraph; print('OK')"
# should print: OK
```

---

## Step 4 — API keys

```bash
cp .env.example .env
```

Open `.env` and fill in:

| Key | Where to get it | Required? |
|-----|----------------|-----------|
| `GOOGLE_API_KEY` | https://aistudio.google.com/apikey | **Yes** — needed for eval steps |
| `LANGSMITH_API_KEY` | https://smith.langchain.com → Settings → API Keys | No — but gives you traces |

Load the file into your shell:

```bash
set -a && source .env && set +a
```

**Verify:**
```bash
echo $GOOGLE_API_KEY    # should print your key, not blank
```

**What breaks without it:** `eval_custom.py` and `eval_ragas.py` will throw an
authentication error when they try to call the Gemini judge.

---

## Step 5 — Build the vector store (`ingest.py`)

```bash
python3 src/ingest.py
```

**What happens inside:**
1. Loads `data/zephyr_handbook.md` (the fictional product handbook)
2. Splits it into 400-character chunks with 80-character overlap
3. Embeds each chunk with `nomic-embed-text` via Ollama
4. Saves everything to `./chroma_db/`

**What to expect:**
```
Loaded 1 document(s).
Split into N chunks.
Embeddings stored in ./chroma_db/
```

**Verify:**
```bash
ls chroma_db/    # should contain files (sqlite db + vector data)
```

**What breaks if you skip this:** Every other script fails because the vector
store doesn't exist yet.

**Experiment to try (Day 3):** Change `chunk_size=400` to `chunk_size=100` in
`ingest.py`, re-run, then re-run `rag_chain.py` with the same question. Watch
quality drop. This is a controlled experiment showing chunking is a tuning knob.
Reset to 400 when done.

---

## Step 6 — Test the RAG pipeline (`rag_chain.py`)

```bash
python3 src/rag_chain.py "How much does the Pro plan cost?"
```

Try a few questions:
```bash
python3 src/rag_chain.py "What is the API rate limit on the Pro plan?"
python3 src/rag_chain.py "What is the capital of France?"   # out-of-corpus test
```

**What to expect:**
- In-corpus questions: a grounded answer drawn from the handbook
- Out-of-corpus question: `"I don't know based on the handbook."`
- Printed below the answer: the 3 retrieved chunks the model was allowed to see

**What you are learning:**
- The answer is only as good as what was retrieved
- The grounding prompt (`"use ONLY the context below"`) is what forces the
  model to say "I don't know" — not the model itself
- Reading the retrieved chunks tells you *why* an answer is right or wrong

**If the answer hallucinates on the France question:** The grounding prompt is
not working as intended — this is worth investigating before moving on.

---

## Step 7 — Test the agent (`agent.py`)

```bash
python3 src/agent.py "What is the API rate limit on the Pro plan?"
python3 src/agent.py "What is 2 + 2?"
python3 src/agent.py "I need 90 extra days of retention. What will it cost?"
```

**What to expect:**
- Question 1: agent calls `search_handbook` tool, returns retrieved answer
- Question 2: agent calls `calculator` tool, returns `4`
- Question 3 (multi-step): agent calls `search_handbook` to find add-on pricing
  (15 dollars per 30 days), then calls `calculator` to compute 90/30*15 = 45

**What you are learning:**
- The agent *decides* which tool to call based on the tool's docstring
- Multi-step reasoning: the model can chain tool calls
- New failure modes vs. a plain RAG chain: wrong tool choice, wrong arguments,
  infinite loops — these require different testing strategies

**If the multi-step question gives the wrong answer:** Check whether the agent
called both tools (it should). If it only called one, the tool descriptions may
need to be clearer — this is itself a lesson in how tool docstrings drive routing.

---

## Step 8 — Hand-rolled evaluation (`eval_custom.py`)

```bash
python3 src/eval_custom.py
```

**What happens inside:**
Runs all 8 golden questions through the RAG chain and measures 4 metrics:

| Metric | What it checks |
|--------|---------------|
| Retrieval hit rate | Did the retriever fetch a chunk containing the answer? |
| Keyword correctness | Does the answer contain the expected keywords? |
| Abstention | Did the model say "I don't know" for out-of-corpus questions? |
| LLM-as-judge | Does Gemini 2.5 Flash grade the answer as faithful to the context? |

**What to expect:**
```
===== RUN 1 of 1 =====
[in ] retrieval=Y keyword=Y judge=Y :: How much does the Pro plan cost?
...
[out] abstention=Y (OK) :: What is the Zephyr mobile app called?
...
===== SUMMARY (pass rates across all runs) =====
Retrieval hit rate : 6/6 = 100%
Keyword correctness: 5/6 = 83%
LLM-judge faithful : 6/6 = 100%
Abstention (no hallucination): 2/2 = 100%
```

Your numbers will likely differ — that is expected.

**Key things to check:**
- Any `[out]` row marked `HALLUCINATED` is a problem — the model invented an
  answer for a question that isn't in the handbook
- Retrieval hit rate lower than keyword correctness means the retriever is
  failing, not the generator
- Retrieval hit rate higher than keyword correctness means the generator is
  failing even with good context

**Experiment to try:** Change `REPEATS = 1` to `REPEATS = 3` at the top of
`eval_custom.py` and re-run. Watch the summary numbers move between runs. This
variance is the core reason AI testing differs from traditional QA — a single
run proves nothing.

---

## Step 9 — RAGAS evaluation (`eval_ragas.py`)

```bash
python3 src/eval_ragas.py
```

**What happens inside:**
Runs the same in-corpus questions through RAGAS, using Gemini 2.5 Flash as the
judge and `nomic-embed-text` for embeddings.

**What to expect** (scores between 0 and 1, higher is better):
```
===== RAGAS SCORES =====
{'faithfulness': 0.92, 'answer_relevancy': 0.88, 'context_precision': 0.75, 'context_recall': 0.83}
```

**How to read the scores:**

| Score | Low means... | Fix |
|-------|-------------|-----|
| `faithfulness` | Model is inventing beyond the context | Tighten the grounding prompt |
| `answer_relevancy` | Answer wanders off the question | Review prompt or raise `k` |
| `context_precision` | Retriever pulls irrelevant chunks (noise) | Lower `k` or improve chunking |
| `context_recall` | Retriever misses needed facts | Raise `k` or re-chunk |

**What you are learning:** Compare these scores against your hand-rolled numbers
from Step 8. Where they agree, you can trust the signal. Where they disagree,
investigate — one of the two methods is wrong, and figuring out which one is a
real testing skill.

**If you see `NaN`:** Check that `GOOGLE_API_KEY` is set and you haven't hit
the free-tier rate limit. Re-run after a minute.

---

## Step 10 — Extend the golden dataset

Open `eval/golden_qa.json` and add at least three new entries:

- One in-corpus question about a fact in the handbook you haven't tested yet
- One multi-fact question that requires two separate chunks to answer correctly
- One out-of-corpus trap (a plausible-sounding question with no answer in the handbook)

Then re-run Steps 8 and 9 and observe how your scores change.

**What you are learning:** You have now authored an eval suite. The discipline of
deciding *what* to test — not just running someone else's metrics — is the skill
that transfers to every AI system you will test in the future.

---

## Quick reference

| Script | Depends on | Requires |
|--------|-----------|---------|
| `ingest.py` | `data/zephyr_handbook.md`, Ollama running | — |
| `rag_chain.py` | `chroma_db/` (run ingest first) | Ollama running |
| `agent.py` | `chroma_db/` (run ingest first) | Ollama running |
| `eval_custom.py` | `chroma_db/`, `eval/golden_qa.json` | Ollama + `GOOGLE_API_KEY` |
| `eval_ragas.py` | `chroma_db/`, `eval/golden_qa.json` | Ollama + `GOOGLE_API_KEY` |
