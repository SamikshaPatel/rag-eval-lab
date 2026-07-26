# RAG + Agent + Eval Lab — a fully free learning project

One tiny project that combines **Step 5 (RAG)**, **Step 6 (Agents & tool use)**,
and **Step 7 (Evaluation)** from your AI-testing plan. Every tool is free and
runs on your own machine — no API bills, no cloud keys except an optional free
LangSmith account.

The thing you are really learning: **how to test a system that gives a
different answer every time you run it.** That is the discipline that separates
AI QA from traditional QA, and evaluation (step 7) is where your existing
testing instincts transfer directly.

---

## The free stack

| Layer            | Tool                    | Cost | Role in the project |
|------------------|-------------------------|------|---------------------|
| Local LLM        | **Ollama** (llama3.1)   | Free | Generation + LLM-as-judge |
| Embeddings       | **nomic-embed-text**    | Free | Turns text into vectors |
| Vector store     | **Chroma**              | Free | Stores + searches vectors, on disk |
| Orchestration    | **LangChain**           | Free | Wires the RAG pipeline |
| Agent framework  | **LangGraph**           | Free | The decision-making loop |
| Tracing          | **LangSmith**           | Free tier | See each step of a run |
| Eval library     | **RAGAS**               | Free | RAG triad metrics |

---

## Where each concept lives, and what you learn from it

Read the files in this order. Each one is heavily commented with the "why".

### Step 5 — Retrieval-Augmented Generation

| Concept | File | What you learn |
|---|---|---|
| Loading | `src/ingest.py` | Raw text is where RAG begins |
| **Chunking** | `src/ingest.py` | Chunk size is a tuning knob *and* a common bug source — facts split across chunks vanish from retrieval |
| **Embeddings** | `src/ingest.py` | Meaning becomes geometry: similar text → nearby vectors |
| **Vector store** | `src/ingest.py` | Chroma answers "which chunks are nearest this query?" — that search *is* retrieval |
| **Retriever + `k`** | `src/rag_chain.py` | How many chunks you fetch trades misses against noise |
| **Prompt grounding** | `src/rag_chain.py` | The "use only the context / else say I don't know" instruction is your main anti-hallucination lever |
| **LCEL pipeline** | `src/rag_chain.py` | RAG is a data-flow graph; reading it shows every place a bug can hide |

### Step 6 — Agents & tool use

| Concept | File | What you learn |
|---|---|---|
| **Tools** | `src/agent.py` | A tool is a function + a docstring the model reads to decide when to call it |
| **Agentic routing** | `src/agent.py` | The model *chooses* the tool — the test surface becomes a decision tree |
| **Multi-step reasoning** | `src/agent.py` | "90 extra days of retention → cost?" needs retrieve **then** calculate |
| **New failure modes** | `src/agent.py` | Wrong tool, wrong arguments, loops — why agents need more testing than chains |

### Step 7 — Evaluation (your home turf)

| Concept | File | What you learn |
|---|---|---|
| **Golden dataset** | `eval/golden_qa.json` | The reference set you test against — including out-of-corpus traps |
| **Retrieval hit rate** | `src/eval_custom.py` | Isolate retrieval from generation before blaming the LLM |
| **Reference-based metrics** | `src/eval_custom.py` | Keyword matching: cheap, deterministic, brittle |
| **Abstention / hallucination test** | `src/eval_custom.py` | Out-of-corpus questions *must* get "I don't know" |
| **LLM-as-judge** | `src/eval_custom.py` | A model grades a model — powerful, but it can be confidently wrong |
| **Pass rates vs single runs** | `src/eval_custom.py` | Set `REPEATS=3` and watch scores wobble — the core reason AI testing differs |
| **Faithfulness** | `src/eval_ragas.py` | Formal grounding metric = your hallucination test, standardised |
| **Answer relevancy** | `src/eval_ragas.py` | Does the answer address the question asked? |
| **Context precision** | `src/eval_ragas.py` | Retriever *noise* — too much junk fetched |
| **Context recall** | `src/eval_ragas.py` | Retriever *misses* — needed facts not fetched |
| **Tracing / observability** | any script + LangSmith | For agents, the trace is the only way to tell a lucky answer from a correct process |

---

## Suggested learning path (about a week, an hour a day)

1. **Day 1** — Do `SETUP.md`. Run `ingest.py`. Open the `chroma_db` folder and
   confirm something was written. You now have a searchable knowledge base.
2. **Day 2** — Run `rag_chain.py` with several questions. Then open the run in
   LangSmith and read the retrieve → prompt → generate trace.
3. **Day 3** — Break it on purpose: set `chunk_size=100` in `ingest.py`,
   re-ingest, re-ask. Watch quality drop. This is a controlled experiment.
4. **Day 4** — Run `agent.py`. Try a lookup question, a math question, and the
   multi-step retention question. Read the trace to see the tool choices.
5. **Day 5** — Run `eval_custom.py`. Read every metric's comment block. Set
   `REPEATS=3` and observe variance.
6. **Day 6** — Run `eval_ragas.py`. Line its scores up against your hand-rolled
   numbers. Where they disagree, figure out who is right.
7. **Day 7** — Add three of your own questions to `golden_qa.json`, including
   one more out-of-corpus trap. You have now authored an eval suite.

---

## The mental shift to internalise

Traditional QA: **same input → same output**, so one run proves correctness.

AI QA: **same input → varying output**, so you test *distributions* — pass
rates over many runs, with metrics that tolerate wording differences, and
explicit hallucination/abstention checks. This project makes that shift
concrete. Everything else in AI testing is a variation on what you do here.

---

*All facts about "Zephyr Analytics" are fictional by design — it forces the
model to rely on retrieval, so you can actually catch it hallucinating.*
