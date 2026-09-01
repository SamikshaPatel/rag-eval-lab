# Setup — one-time, ~15 minutes

The only large download is the Ollama model weights (a few GB). Evaluation also
requires a free Google API key for the Gemini 2.5 Flash judge (step 4).

## 1. Install Ollama (your local, free LLM engine)

Download from https://ollama.com and install. Then start the background server:

```bash
ollama serve      # leave this running in its own terminal
```

## 2. Pull the two models

RAG uses **two separate models** — one to embed, one to chat. Forgetting the
embedding model is the #1 beginner error.

```bash
ollama pull llama3.1:8b        # the chat / generation model (~4.7 GB)
ollama pull nomic-embed-text   # the embedding model (~275 MB)
```

Low on RAM? Swap `llama3.1:8b` for a smaller model like `qwen2.5:3b` in
`src/config.py` (the `CHAT_MODEL` constant). Smaller models are worse at
tool-calling — which, for the agent step, is itself interesting to observe.

## 3. Python environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 4. API keys — Gemini (required) and LangSmith (optional)

```bash
cp .env.example .env
```

**Gemini 3.6 Flash (required for eval judge, or use local fallback)**

`eval_custom.py`, `eval_ragas.py`, and `eval_deepeval.py` all use
`gemini-3.6-flash` as the LLM judge by default.

1. Get a free key at https://aistudio.google.com/apikey (no card required).
2. Paste it into `.env` as `GOOGLE_API_KEY=your_key_here`.
3. Load it: `set -a && source .env && set +a`

**Free-tier limit:** 20 requests/day. If you hit the quota, set
`USE_LOCAL_JUDGE=1` in `.env` to fall back to `llama3.1:8b` (already
installed via Ollama). Scores will be noisier but the scripts still run.

Without a key and without `USE_LOCAL_JUDGE=1`, the eval scripts will error.

**LangSmith (optional — for tracing and eval UI)**

1. Sign up at https://smith.langchain.com (free Developer tier, no card).
2. Create an API key under Settings → API Keys.
3. Paste it into `.env` as `LANGSMITH_API_KEY=your_key_here`.

Skipping LangSmith is fine — the code still runs, you just won't get traces or the
Experiments comparison UI. Steps 10-13 in RUNBOOK.md cover the LangSmith eval workflow.

## 5. Run the project in order

```bash
# Zephyr corpus (fictional analytics handbook — run first)
python3 src/ingest_zephyr.py          # build the Zephyr vector store (run once)
python3 src/rag_chain.py "How much is the Pro plan?"   # try RAG
python3 src/agent.py "How much extra would 90-day retention cost for 500 users?"  # watch the agent route between tools

# Evaluation (Zephyr)
python3 src/eval_custom.py            # your hand-built eval harness
python3 src/eval_ragas.py             # RAGAS library eval
USE_LOCAL_JUDGE=1 python3 src/eval_deepeval.py   # DeepEval (local judge, no quota)
python3 src/eval_deepeval.py          # DeepEval with Gemini judge (requires quota)

# Northstar corpus (fictional banking handbook — optional second corpus)
python3 src/ingest_northstar.py       # build the Northstar vector store (run once)
USE_LOCAL_JUDGE=1 python3 src/eval_langsmith_northstar.py  # run 44-question Northstar eval
```

**Rate limit tip:** All three eval scripts share the same Gemini free-tier
quota (20 req/day). If you hit the limit, add `USE_LOCAL_JUDGE=1` to `.env`
and re-run — all scripts will switch to the local Ollama judge automatically.
