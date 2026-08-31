"""
Shared judge utilities for all eval scripts.
=============================================
Centralises the three things every evaluator needs:

  - GeminiJudge   : DeepEvalBaseLLM wrapper around ChatGoogleGenerativeAI
  - make_judge()  : factory that returns either GeminiJudge or OllamaModel
                    based on the USE_LOCAL_JUDGE env var
  - ABSTENTION_MARKERS / abstained() : rule-based out-of-corpus detection

Nothing here depends on the RAG pipeline. Import config constants from
config.py; import runtime functions (answer_with_context etc.) from
rag_chain.py.
"""

import os

from langchain_google_genai import ChatGoogleGenerativeAI
from deepeval.models import DeepEvalBaseLLM, OllamaModel

from config import JUDGE_MODEL, LOCAL_JUDGE_MODEL


# ---------------------------------------------------------------------------
# GeminiJudge — wraps ChatGoogleGenerativeAI for DeepEval's interface.
# DeepEval expects generate() to return a plain str; Gemini's content field
# can be a list of parts, so we normalise it here.
# ---------------------------------------------------------------------------
class GeminiJudge(DeepEvalBaseLLM):
    def __init__(self):
        self._llm = ChatGoogleGenerativeAI(model=JUDGE_MODEL, temperature=0)

    def load_model(self):
        return self._llm

    def get_model_name(self) -> str:
        return JUDGE_MODEL

    def generate(self, prompt: str) -> str:
        result = self._llm.invoke(prompt).content
        if isinstance(result, list):
            result = " ".join(
                p.get("text", str(p)) if isinstance(p, dict) else str(p)
                for p in result
            )
        return result

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)


# ---------------------------------------------------------------------------
# make_judge() — single entry point for all eval scripts.
# Returns OllamaModel when USE_LOCAL_JUDGE=1 (free, no quota limit);
# otherwise returns GeminiJudge (cloud, 20 req/day free tier).
# ---------------------------------------------------------------------------
def make_judge():
    if os.getenv("USE_LOCAL_JUDGE") == "1":
        print(f"[judge] Using local Ollama model: {LOCAL_JUDGE_MODEL}")
        return OllamaModel(model=LOCAL_JUDGE_MODEL, temperature=0)
    print(f"[judge] Using Gemini: {JUDGE_MODEL}")
    return GeminiJudge()


# ---------------------------------------------------------------------------
# Abstention detection — shared by eval_custom, eval_deepeval, eval_langsmith.
# Out-of-corpus questions should produce one of these refusal phrases.
# ---------------------------------------------------------------------------
ABSTENTION_MARKERS = [
    "don't know", "do not know", "not in", "no information",
    "cannot", "can't", "not contain", "not available",
]


def abstained(answer: str) -> bool:
    """Return True if the answer is a recognised refusal (correct for out-of-corpus Qs)."""
    low = answer.lower()
    return any(m in low for m in ABSTENTION_MARKERS)
