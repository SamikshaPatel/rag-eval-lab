# Setup — one-time, ~15 minutes

Everything below is free. The only real download is the model weights (a few GB).

## 1. Install Ollama (your local, free LLM engine)

Download from https://ollama.com and install. Then start the background server:

```bash
ollama serve      # leave this running in its own terminal
```

## 2. Pull the two models

RAG uses **two separate models** — one to embed, one to chat. Forgetting the
embedding model is the #1 beginner error.

```bash
ollama pull llama3.1:8b        # the chat / generation / judge model (~4.7 GB)
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

## 4. LangSmith (free tracing) — optional but recommended

1. Sign up at https://smith.langchain.com (free Developer tier, no card).
2. Create an API key under Settings → API Keys.
3. `cp .env.example .env` and paste your key in.
4. Load it: `set -a && source .env && set +a`

Skipping this is fine — the code still runs, you just won't get traces.

## 5. Run the project in order

```bash
python src/ingest.py                 # build the vector store (run once)
python src/rag_chain.py "How much is the Pro plan?"   # try RAG
python src/agent.py                  # watch the agent route between tools
python src/eval_custom.py            # your hand-built eval harness
python src/eval_ragas.py             # the same, via the RAGAS library
```

If `eval_ragas.py` prints `NaN` for a metric, that is almost always an Ollama
timeout, not a real zero. Try a smaller model or rerun; the `RunConfig` timeout
is already raised for you.
