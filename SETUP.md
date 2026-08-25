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
`src/rag_chain.py` (the `CHAT_MODEL` constant). Smaller models are worse at
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

**Gemini 2.5 Flash (required for eval)**

Both `eval_custom.py` and `eval_ragas.py` use Gemini 2.5 Flash as the LLM judge.

1. Get a free key at https://aistudio.google.com/apikey (no card required).
2. Paste it into `.env` as `GOOGLE_API_KEY=your_key_here`.
3. Load it: `set -a && source .env && set +a`

Without this key the eval scripts will error when they try to call the judge.

**LangSmith (optional — for tracing)**

1. Sign up at https://smith.langchain.com (free Developer tier, no card).
2. Create an API key under Settings → API Keys.
3. Paste it into `.env` as `LANGSMITH_API_KEY=your_key_here`.

Skipping LangSmith is fine — the code still runs, you just won't get traces.

## 5. Run the project in order

```bash
python src/ingest.py                 # build the vector store (run once)
python src/rag_chain.py "How much is the Pro plan?"   # try RAG
python src/agent.py                  # watch the agent route between tools
python src/eval_custom.py            # your hand-built eval harness
python src/eval_ragas.py             # the same, via the RAGAS library
```

If `eval_ragas.py` prints `NaN` for a metric, check that `GOOGLE_API_KEY` is
set correctly and that you have not hit the free-tier rate limit. The `RunConfig`
timeout is already raised and parallelism is limited to avoid this.
