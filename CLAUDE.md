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

# Required for eval: copy .env.example → .env and add GOOGLE_API_KEY (Gemini judge)
# Optional: also add LangSmith keys for tracing
```

## Architecture

Five modular scripts form a data-flow pipeline:

### Data Flow
```
data/zephyr_handbook.md
    → ingest.py (chunk + embed)
    → chroma_db/ (persisted vector store)
    → rag_chain.py (retrieve + generate)
    → eval_custom.py / eval_ragas.py (evaluate)
         ↑ also tested via agent.py (tool routing)
```

### Key Components

**`src/ingest.py`** — One-time ingestion: loads `data/zephyr_handbook.md`, splits into 400-char chunks (80-char overlap), embeds with `nomic-embed-text`, persists to `./chroma_db/`. Chunk size is a primary tuning knob—changing it requires re-running this script.

**`src/rag_chain.py`** — Core RAG pipeline. Exposes `answer_with_context(question, k)` returning both answer and retrieved contexts (required for eval). `k=3` retrieved chunks by default; temperature=0 for repeatability. Grounding prompt instructs the model to say "I don't know based on the handbook" for out-of-corpus questions.

**`src/agent.py`** — LangGraph ReAct agent with two tools: `search_handbook()` (wraps retriever) and `calculator()` (whitelist-validated arithmetic). Handles multi-step questions that require retrieval then calculation.

**`src/eval_custom.py`** — Hand-rolled eval harness measuring: retrieval hit rate, keyword correctness, abstention (hallucination test), and LLM-as-judge (Gemini 2.5 Flash via `ChatGoogleGenerativeAI`). The `REPEATS` parameter (default 1) controls multi-run variance measurement.

**`src/eval_ragas.py`** — RAGAS evaluation using Gemini 2.5 Flash as the judge via `LangchainLLMWrapper(ChatGoogleGenerativeAI(...))`. Computes faithfulness, answer relevancy, context precision, and context recall. Embeddings still use local Ollama (`nomic-embed-text`).

### Golden Dataset

`eval/golden_qa.json` — 8 QA pairs: 6 in-corpus, 2 out-of-corpus (abstention tests). Each entry has `question`, `reference`, `must_contain` keywords, and `in_corpus` flag.

## Critical Design Decisions

- **Fictional corpus**: Forces retrieval dependency; model cannot rely on training data
- **Temperature=0**: Reduces variance in generation for more reproducible evals
- **`answer_with_context()`**: Returns contexts alongside answers—RAGAS and custom eval both need the retrieved chunks, not just the final answer
- **Gemini 2.5 Flash as judge**: Both `eval_custom.py` and `eval_ragas.py` use `JUDGE_MODEL = "gemini-2.5-flash"` (defined in `rag_chain.py`) via `ChatGoogleGenerativeAI`. Requires `GOOGLE_API_KEY` in `.env`. Ollama is still used for RAG generation and embeddings; only the judge role uses Gemini.
- **Calculator whitelist**: `re.fullmatch(r'[\d\s\+\-\*/\(\)\.]+', expression)` prevents code injection via the tool
