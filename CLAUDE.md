# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

An educational project teaching RAG (Retrieval-Augmented Generation), Agents, and AI Evaluation using a fully local, free stack. The fictional Zephyr Analytics corpus is intentional—the model has never seen it in training, so any correct answer *must* come from retrieval.

## Setup & Commands

```bash
# One-time setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Prerequisites: Ollama must be running locally (http://localhost:11434)
# Pull required models: ollama pull llama3.1:8b && ollama pull nomic-embed-text

# Run in order
python3 src/ingest.py                        # Build vector store (run once, or after changing data/chunking)
python3 src/rag_chain.py "Your question"     # Test RAG pipeline
python3 src/agent.py "Multi-step question"   # Test agent tool routing
python3 src/eval_custom.py                   # Run hand-rolled eval harness
python3 src/eval_ragas.py                    # Run RAGAS evaluation
USE_LOCAL_JUDGE=1 python3 src/eval_deepeval.py  # Run DeepEval (local Ollama judge)
python3 src/eval_deepeval.py                 # Run DeepEval (Gemini judge)
USE_LOCAL_JUDGE=1 python3 src/eval_langsmith.py  # Run all golden questions as a LangSmith experiment (local judge)
python3 src/eval_langsmith.py               # Run LangSmith experiment (Gemini judge)
python3 src/run_golden_eval.py              # Run golden dataset to trigger LangSmith online evaluator

# Required for eval: copy .env.example → .env and add GOOGLE_API_KEY (Gemini judge)
# Optional: add USE_LOCAL_JUDGE=1 to .env to use Ollama when Gemini quota is exhausted
# Optional: also add LangSmith keys for tracing
```

## Architecture

Seven modular scripts form a data-flow pipeline:

### Data Flow
```
data/zephyr_handbook.md
    → ingest.py (chunk + embed)
    → chroma_db/ (persisted vector store)
    → rag_chain.py (retrieve + generate)
    → eval_custom.py / eval_ragas.py / eval_deepeval.py (evaluate)
    → eval_langsmith.py / run_golden_eval.py (LangSmith experiments)
         ↑ also tested via agent.py (tool routing)
```

### Key Components

**`src/config.py`** — Single source of truth for all tunable parameters: model names, chunk size/overlap, retrieval k, temperature, prompt filenames, and all metric thresholds. Every other script imports constants from here. To change any knob, edit this file only.

**`src/ingest.py`** — One-time ingestion: loads `data/zephyr_handbook.md`, splits into 400-char chunks (80-char overlap), embeds with `nomic-embed-text`, persists to `./chroma_db/`. Chunk size is a primary tuning knob—change `CHUNK_SIZE`/`CHUNK_OVERLAP` in `config.py` (single source of truth), then re-run this script.

**`src/rag_chain.py`** — Core RAG pipeline. Exposes `answer_with_context(question, k)` returning both answer and retrieved contexts (required for eval). `k=3` retrieved chunks by default; temperature=0 for repeatability. Grounding prompt instructs the model to say "I don't know based on the handbook" for out-of-corpus questions.

**`src/agent.py`** — LangGraph ReAct agent with two tools: `search_handbook()` (wraps retriever) and `calculator()` (whitelist-validated arithmetic). Handles multi-step questions that require retrieval then calculation.

**`src/eval_custom.py`** — Hand-rolled eval harness measuring: retrieval hit rate, keyword correctness, abstention (hallucination test), and LLM-as-judge (Gemini 2.5 Flash via `ChatGoogleGenerativeAI`). The `REPEATS` parameter (default 1) controls multi-run variance measurement.

**`src/eval_ragas.py`** — RAGAS evaluation using `gemini-3.6-flash` as the judge via `LangchainLLMWrapper(ChatGoogleGenerativeAI(...))`. Computes faithfulness, answer relevancy, context precision, and context recall. Embeddings still use local Ollama (`nomic-embed-text`). Set `USE_LOCAL_JUDGE=1` to use Ollama instead.

**`src/eval_deepeval.py`** — DeepEval evaluation: same four metrics as RAGAS but pytest-style (each question is an `LLMTestCase`). Makes fewer LLM sub-calls per metric than RAGAS. Judge is `gemini-3.6-flash` by default; set `USE_LOCAL_JUDGE=1` to fall back to `llama3.1:8b` via `OllamaModel`. Baseline results (local judge): Contextual Precision 0.83, Answer Relevancy 0.83, Faithfulness 0.58, Contextual Recall 0.51.

**`src/eval_langsmith.py`** — LangSmith experiment runner. Creates the `zephyr-golden-qa` dataset in LangSmith from `eval/golden_qa.json` (idempotent — skips if already exists), then runs all 8 questions as a named experiment, logging faithfulness, answer relevancy, and abstention scores per question via DeepEval. Each run creates a new experiment so scores can be compared across pipeline changes in the LangSmith UI. Uses Gemini judge by default; set `USE_LOCAL_JUDGE=1` for Ollama.

**`src/run_golden_eval.py`** — Minimal golden-dataset runner for LangSmith's online evaluator. Runs all 8 questions through `answer_with_context` as a LangSmith experiment with no local judge — relies on LangSmith's configured Answer Relevancy online evaluator (set up in the UI) to score traces automatically. Use this when you want LangSmith to handle scoring rather than a local DeepEval/RAGAS judge.

### Golden Dataset

`eval/golden_qa.json` — 8 QA pairs: 6 in-corpus, 2 out-of-corpus (abstention tests). Each entry has `question`, `reference`, `must_contain` keywords, and `in_corpus` flag.

## Critical Design Decisions

- **Fictional corpus**: Forces retrieval dependency; model cannot rely on training data
- **Temperature=0**: Reduces variance in generation for more reproducible evals
- **`answer_with_context()`**: Returns contexts alongside answers—RAGAS and custom eval both need the retrieved chunks, not just the final answer
- **Gemini 3.6 Flash as judge**: All three eval scripts (`eval_custom.py`, `eval_ragas.py`, `eval_deepeval.py`) use `JUDGE_MODEL = "gemini-3.6-flash"` (defined in `config.py`) via `ChatGoogleGenerativeAI`. Requires `GOOGLE_API_KEY` in `.env`. Free tier is 20 req/day — set `USE_LOCAL_JUDGE=1` in `.env` to fall back to `LOCAL_JUDGE_MODEL = "qwen2.5:7b"` via Ollama when quota is exhausted. Ollama is always used for RAG generation and embeddings; only the judge role uses Gemini.
- **Calculator whitelist**: `re.fullmatch(r'[\d\s\+\-\*/\(\)\.]+', expression)` prevents code injection via the tool
