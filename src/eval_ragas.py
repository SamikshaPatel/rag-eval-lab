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

THE "NOT ACTUALLY FREE" TRAP (important):
RAGAS defaults to OpenAI, which costs money and needs an API key. To keep this
100% free we point RAGAS at your LOCAL Ollama model using LangchainLLMWrapper.
Watch out: local judges are slower and RAGAS has a known habit of TIMING OUT
against Ollama. Mitigations are applied below (low k, small dataset, longer
timeout). If a metric returns NaN, that is usually a timeout, not a zero.

Run (after ingest.py):  python src/eval_ragas.py
"""

import json
from langchain_ollama import ChatOllama, OllamaEmbeddings
from rag_chain import answer_with_context, CHAT_MODEL, EMBED_MODEL

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

    # Point RAGAS at LOCAL models — this is the line that keeps it free.
    evaluator_llm = LangchainLLMWrapper(ChatOllama(model=CHAT_MODEL, temperature=0))
    evaluator_emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=EMBED_MODEL))

    # Longer timeout + no parallelism helps local models survive the run.
    run_config = RunConfig(timeout=180, max_workers=1)

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
