"""
STEP 7 (EVALUATION) — PART 2: RAGAS, THE INDUSTRY-STANDARD LIBRARY
=================================================================
Now that you have hand-built the metrics, here is the library everyone uses.
RAGAS computes the "RAG triad" plus more, using LLM-as-judge internally:

  - faithfulness       : is the answer grounded in the retrieved context?
                         (your hallucination metric, formalised)
  - answer_relevancy   : does the answer actually address the question?
  - context_precision  : of the chunks retrieved, how many were relevant?
                         (measures retriever NOISE)
  - context_recall     : of the facts needed, how many were retrieved?
                         (measures retriever MISSES — needs a reference answer)

What you learn: precision vs recall on the RETRIEVAL side is the single most
useful diagnostic in RAG. Low precision -> tighten/lower k. Low recall ->
fix chunking or raise k. You are localising the fault before touching the LLM.

THE JUDGE LLM:
RAGAS defaults to OpenAI. Here we point it at Gemini 2.5 Flash via
LangchainLLMWrapper, which requires a GOOGLE_API_KEY in your .env file.
Gemini is faster and more reliable than local Ollama judges for RAGAS.
If a metric returns NaN, check that GOOGLE_API_KEY is set correctly.

Run (after ingest.py):  python src/eval_ragas.py
"""

import json
from langchain_ollama import OllamaEmbeddings
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from config import JUDGE_MODEL, LOCAL_JUDGE_MODEL, EMBED_MODEL
from rag_chain import answer_with_context

# RAGAS 0.2.x imports. If these fail, check `pip show ragas` — the API changed
# across versions. Your hand-rolled eval_custom.py is the version-proof backup.
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig


def build_dataset():
    """Run our RAG pipeline on the in-corpus questions and shape the results
    the way RAGAS expects: question, contexts, answer, and a reference."""
    with open("eval/golden_qa.json") as f:
        golden = [g for g in json.load(f) if g["in_corpus"]]

    samples = []
    for item in golden:
        answer, contexts = answer_with_context(item["question"])
        samples.append({
            "user_input": item["question"],
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": item["reference"],
        })
    return EvaluationDataset.from_list(samples)


def main():
    dataset = build_dataset()

    # USE_LOCAL_JUDGE=1 in .env uses Ollama instead of Gemini (free, no quota).
    if os.getenv("USE_LOCAL_JUDGE") == "1":
        print(f"[judge] Using local Ollama model: {LOCAL_JUDGE_MODEL}")
        _llm = ChatOllama(model=LOCAL_JUDGE_MODEL, temperature=0)
    else:
        print(f"[judge] Using Gemini: {JUDGE_MODEL}")
        _llm = ChatGoogleGenerativeAI(model=JUDGE_MODEL, temperature=0)
    evaluator_llm = LangchainLLMWrapper(_llm)
    evaluator_emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=EMBED_MODEL))

    # Longer timeout + no parallelism helps the judge model survive the run.
    # 600s per call accommodates slow Gemini responses under rate limits.
    run_config = RunConfig(timeout=600, max_workers=1)

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_emb,
        run_config=run_config,
    )

    print("\n===== RAGAS SCORES =====")
    print(result)
    print("\nHow to read these (0 to 1, higher is better):")
    print("  faithfulness      low -> the model is inventing beyond the context")
    print("  answer_relevancy  low -> answers wander off the question")
    print("  context_precision low -> retriever pulls junk; lower k or improve chunks")
    print("  context_recall    low -> retriever misses needed facts; raise k or re-chunk")
    print("\nCompare these against your hand-rolled numbers from eval_custom.py. "
          "When two independent methods agree, you can trust the result.")


if __name__ == "__main__":
    main()
