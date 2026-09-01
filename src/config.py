"""
Central configuration for the RAG eval lab.
=========================================
All tunable parameters live here. Import constants from this module;
never hardcode values in individual scripts.

To change any knob, edit this file only — every other script picks it up
automatically on the next run.

Prompt versioning: to upgrade a prompt, copy prompts/X_v1.txt →
prompts/X_v2.txt, edit the body, then update the filename constant below.
No other .py file needs to change.
"""

from pathlib import Path

# --- Zephyr corpus -----------------------------------------------------------
ZEPHYR_DB_PATH      = "./chroma_db_zephyr"
ZEPHYR_DATA_PATH    = "data/zephyr_handbook.md"
ZEPHYR_GOLDEN_PATH  = "eval/golden_qa_zephyr.json"
ZEPHYR_DATASET_NAME = "zephyr-golden-qa"

# --- Northstar corpus --------------------------------------------------------
NORTHSTAR_DB_PATH      = "./chroma_db_northstar"
NORTHSTAR_DATA_PATH    = "data/Northstar_Digital_Bank.docx"
NORTHSTAR_GOLDEN_PATH  = "eval/golden_qa_northstar.jsonl"
NORTHSTAR_DATASET_NAME = "northstar-golden-qa"

# --- Models ------------------------------------------------------------------
EMBED_MODEL       = "nomic-embed-text"   # local Ollama embedding model
CHAT_MODEL        = "llama3.1:8b"        # local Ollama generation model
JUDGE_MODEL       = "gemini-3.6-flash"   # cloud judge (default; 20 req/day free)
LOCAL_JUDGE_MODEL = "qwen2.5:7b"         # local fallback judge (USE_LOCAL_JUDGE=1)

# --- Chunking ----------------------------------------------------------------
# Zephyr corpus
CHUNK_SIZE    = 400   # chars per chunk — re-run ingest_zephyr.py after changing
CHUNK_OVERLAP = 80    # overlap between chunks

# Northstar corpus (tuned separately so Zephyr baseline is preserved)
NORTHSTAR_CHUNK_SIZE    = 200
NORTHSTAR_CHUNK_OVERLAP = 40

# --- Retrieval & generation --------------------------------------------------
RETRIEVAL_K = 3     # chunks kept after reranking (or fetched if no reranker)
TEMPERATURE = 0.0   # generation temperature (0 = most deterministic)
REPEATS     = 1     # eval_custom: bump to 3+ to measure run-to-run variance

# --- Reranking ---------------------------------------------------------------
# Cross-encoder reranker: fetch RERANKER_FETCH_K candidates, rerank, keep RETRIEVAL_K.
# Model downloads automatically from HuggingFace on first use (~85 MB).
RERANKER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_FETCH_K = 10   # initial candidate pool before reranking

# --- Prompt filenames --------------------------------------------------------
# Update the string here when you version a prompt; no logic files change.
PROMPT_RAG_GROUNDING      = "rag_grounding_v1.txt"
PROMPT_JUDGE_FAITHFULNESS = "judge_faithfulness_v1.txt"
PROMPT_JUDGE_CORRECTNESS  = "judge_correctness_v1.txt"
PROMPT_JUDGE_COMPLETENESS = "judge_completeness_v1.txt"

# --- Metric thresholds (pass/fail boundary; raw scores are always logged) ----
FAITHFULNESS_THRESHOLD         = 0.7
RELEVANCY_THRESHOLD            = 0.7
CONTEXTUAL_PRECISION_THRESHOLD = 0.7
CONTEXTUAL_RECALL_THRESHOLD    = 0.7
CONTEXTUAL_RELEVANCY_THRESHOLD = 0.7
HALLUCINATION_THRESHOLD        = 0.8   # 1=no hallucination → higher = stricter
BIAS_THRESHOLD                 = 0.8   # 1=no bias          → higher = stricter
TOXICITY_THRESHOLD             = 0.8   # 1=no toxicity      → higher = stricter
CORRECTNESS_THRESHOLD          = 0.7
COMPLETENESS_THRESHOLD         = 0.7


# ---------------------------------------------------------------------------
# Prompt loader
# Strips the header comment block (everything up to and including the first
# '---' separator line) so only the prompt body is returned.
# ---------------------------------------------------------------------------
def _load_prompt(filename: str) -> str:
    """Load a prompt from the prompts/ directory."""
    path = Path(__file__).parent.parent / "prompts" / filename
    content = path.read_text()
    if "---\n" in content:
        return content.split("---\n", 1)[1]
    return content
