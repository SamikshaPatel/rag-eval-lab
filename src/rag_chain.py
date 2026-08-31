"""
STEP 5 (RAG) — PART 2: RETRIEVE + GENERATE
==========================================
This is the actual RAG pipeline. It answers a question by first RETRIEVING
relevant chunks, then asking the LLM to GENERATE an answer grounded in them.

Concepts you learn here:
  1. RETRIEVER        — query -> nearest chunks
  2. PROMPT GROUNDING — instructing the model to use ONLY the retrieved context
  3. LCEL             — LangChain's way of wiring steps into a pipeline
  4. LANGSMITH        — seeing, step by step, what actually happened (tracing)

This file exposes build_rag_chain() and get_retriever() so the agent and the
eval scripts can reuse the exact same pipeline. Reusing one definition is a
testing principle too: you evaluate the thing you ship, not a lookalike.

Run directly to try a question:  python src/rag_chain.py "How much is Pro?"
"""

import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()


def _load_prompt(filename: str) -> str:
    """Load a prompt from the prompts/ directory. Strips the header comment
    block (everything up to and including the first '---' separator line)."""
    path = Path(__file__).parent.parent / "prompts" / filename
    content = path.read_text()
    if "---\n" in content:
        return content.split("---\n", 1)[1]
    return content

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

# =============================================================================
# ALL TUNABLE PARAMETERS — this is the single file to edit for any config change.
# Every other script imports from here; nothing is hardcoded elsewhere.
# =============================================================================

# --- Models ------------------------------------------------------------------
DB_PATH = "./chroma_db"
EMBED_MODEL    = "nomic-embed-text"   # local Ollama embedding model
CHAT_MODEL     = "llama3.1:8b"        # local Ollama generation model
JUDGE_MODEL    = "gemini-3.6-flash"   # cloud judge (default; 20 req/day free)
LOCAL_JUDGE_MODEL = "qwen2.5:7b"      # local fallback judge (USE_LOCAL_JUDGE=1)

# --- Chunking ----------------------------------------------------------------
CHUNK_SIZE    = 400   # chars per chunk — re-run ingest.py after changing
CHUNK_OVERLAP = 80    # overlap between chunks

# --- Retrieval & generation --------------------------------------------------
RETRIEVAL_K = 3     # chunks fetched per query
TEMPERATURE = 0.0   # generation temperature (0 = most deterministic)
REPEATS     = 1     # eval_custom: bump to 3+ to measure run-to-run variance

# --- Prompt filenames --------------------------------------------------------
# To upgrade a prompt: copy prompts/X_v1.txt → prompts/X_v2.txt, edit the
# body, update the version string below. No other .py file needs to change.
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

# =============================================================================
# ---------------------------------------------------------------------------
# 1. RETRIEVER
# What you learn: k is how many chunks you fetch. Another tuning knob and
# another testable variable. Low k can miss the answer; high k floods the
# prompt with noise and can *cause* hallucination.
# ---------------------------------------------------------------------------
def get_retriever(k: int = 3):
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": k})


def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


# ---------------------------------------------------------------------------
# 2. PROMPT GROUNDING
# What you learn: this prompt is your main defence against hallucination. The
# instruction to say "I don't know" when the context is silent is what makes
# the abstention tests in the golden set pass or fail. Editing this one string
# changes your faithfulness scores — a direct, testable cause and effect.
# ---------------------------------------------------------------------------
PROMPT = ChatPromptTemplate.from_template(_load_prompt(PROMPT_RAG_GROUNDING))


# ---------------------------------------------------------------------------
# 3. LCEL — compose retriever -> prompt -> model -> text
# What you learn: RAG is a data-flow pipeline. Reading this graph tells you
# every place a failure can hide: bad retrieval, bad prompt, or bad generation.
# ---------------------------------------------------------------------------
def build_rag_chain(k: int = 3):
    retriever = get_retriever(k=k)
    llm = ChatOllama(model=CHAT_MODEL, temperature=0)  # temp 0 = as repeatable as possible
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


# ---------------------------------------------------------------------------
# 4. LANGSMITH-TRACED ANSWER FUNCTION
# @traceable creates a named span whose input is the question string and
# whose output is the answer string — no nesting, no {"output": ...} wrapper.
# This is what LangSmith online evaluators target for field mapping.
# No-op when LANGSMITH_TRACING is not set; safe to call without LangSmith.
# ---------------------------------------------------------------------------
@traceable(run_type="chain", name="rag-answer")
def _generate_answer(question: str, k: int = 3) -> str:
    chain = build_rag_chain(k=k)
    return chain.invoke(question)


# A helper that returns BOTH the answer and the retrieved contexts.
# The eval scripts need the contexts to measure retrieval quality, not just
# the final text. Always instrument the middle of the pipeline, not only the end.
def answer_with_context(question: str, k: int = 3):
    retriever = get_retriever(k=k)
    contexts = retriever.invoke(question)
    answer = _generate_answer(question, k=k)
    return answer, [c.page_content for c in contexts]


if __name__ == "__main__":
    # 4. LANGSMITH tracing is automatic when the env vars in .env are set.
    # After running this, open smith.langchain.com to see the retrieve -> prompt
    # -> generate steps laid out. Seeing the trace is how you debug RAG.
    q = sys.argv[1] if len(sys.argv) > 1 else "How much does the Pro plan cost?"
    ans, ctx = answer_with_context(q)
    print(f"\nQ: {q}\nA: {ans}\n")
    print("--- retrieved context (what the model was allowed to see) ---")
    for i, c in enumerate(ctx, 1):
        print(f"[{i}] {c[:150]}...")
