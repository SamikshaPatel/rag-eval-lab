"""
STEP 7 (EVALUATION) — PART 4: LANGSMITH EXPERIMENTS
=====================================================
Runs all golden-dataset questions through the RAG pipeline as a LangSmith
experiment so faithfulness scores appear in the LangSmith UI and can be
compared across runs (e.g., before/after a prompt or chunking change).

Each invocation creates a new experiment under the same dataset. Open
smith.langchain.com → your project → Datasets & Experiments to compare.

Metrics logged to LangSmith per question:
  - faithfulness          (DeepEval FaithfulnessMetric)
  - answer_relevancy      (DeepEval AnswerRelevancyMetric)
  - abstention            (rule-based, for out-of-corpus questions)

Run:
    python src/eval_langsmith.py
    USE_LOCAL_JUDGE=1 python src/eval_langsmith.py   # Ollama judge
"""

import json
import os
import sys
from dotenv import load_dotenv
load_dotenv()

from langsmith import Client
from langsmith.evaluation import evaluate as ls_evaluate

from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.models import OllamaModel

from langchain_google_genai import ChatGoogleGenerativeAI
from deepeval.models import DeepEvalBaseLLM

# Reuse constants and the traced answer function from rag_chain
sys.path.insert(0, os.path.dirname(__file__))
from rag_chain import (
    answer_with_context, JUDGE_MODEL, LOCAL_JUDGE_MODEL,
    CHAT_MODEL, EMBED_MODEL,
)

DATASET_NAME = "zephyr-golden-qa"
GOLDEN_PATH = "eval/golden_qa.json"

# ---------------------------------------------------------------------------
# Pipeline parameters — edit these to run a different configuration.
# Each value is logged to LangSmith as experiment metadata so you can filter
# and compare runs across configurations in the UI (Columns → metadata.*).
# ---------------------------------------------------------------------------
RETRIEVAL_K = 3           # chunks fetched per query
CHUNK_SIZE = 400          # characters per chunk (must match ingest.py)
CHUNK_OVERLAP = 80        # overlap between chunks (must match ingest.py)
TEMPERATURE = 0.0         # generation temperature (0 = most deterministic)
FAITHFULNESS_THRESHOLD = 0.7
RELEVANCY_THRESHOLD = 0.7

ABSTENTION_MARKERS = [
    "don't know", "do not know", "not in", "no information",
    "cannot", "can't", "not contain", "not available",
]


# ---------------------------------------------------------------------------
# Judge model (mirrors eval_deepeval.py)
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


def make_judge():
    if os.getenv("USE_LOCAL_JUDGE") == "1":
        print(f"[judge] Using local Ollama model: {LOCAL_JUDGE_MODEL}")
        return OllamaModel(model=LOCAL_JUDGE_MODEL, temperature=0)
    print(f"[judge] Using Gemini: {JUDGE_MODEL}")
    return GeminiJudge()


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def ensure_dataset(client: Client, golden: list) -> str:
    """Create the LangSmith dataset if it doesn't exist; return its name."""
    existing = [d.name for d in client.list_datasets()]
    if DATASET_NAME in existing:
        print(f"[dataset] '{DATASET_NAME}' already exists — reusing.")
        return DATASET_NAME

    print(f"[dataset] Creating '{DATASET_NAME}' with {len(golden)} examples...")
    dataset = client.create_dataset(
        DATASET_NAME,
        description="8-question golden set for Zephyr RAG eval (6 in-corpus, 2 abstention).",
    )
    client.create_examples(
        inputs=[{"question": q["question"], "in_corpus": q["in_corpus"]} for q in golden],
        outputs=[{"reference": q["reference"], "must_contain": q["must_contain"]} for q in golden],
        dataset_id=dataset.id,
    )
    print(f"[dataset] Created with id={dataset.id}")
    return DATASET_NAME


# ---------------------------------------------------------------------------
# Target function — what LangSmith runs for each example
# ---------------------------------------------------------------------------
def rag_target(inputs: dict) -> dict:
    question = inputs["question"]
    answer, contexts = answer_with_context(question, k=RETRIEVAL_K)
    return {"answer": answer, "contexts": contexts}


# ---------------------------------------------------------------------------
# Evaluators — each returns a score dict that LangSmith records
# ---------------------------------------------------------------------------
def make_faithfulness_evaluator(judge):
    def faithfulness_evaluator(run, example) -> dict:
        if not example.inputs.get("in_corpus", True):
            return {"key": "faithfulness", "score": None, "comment": "skipped — out-of-corpus"}

        answer = run.outputs["answer"]
        contexts = run.outputs["contexts"]
        question = example.inputs["question"]
        reference = example.outputs["reference"]

        case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=reference,
            retrieval_context=contexts,
        )
        metric = FaithfulnessMetric(threshold=FAITHFULNESS_THRESHOLD, model=judge, verbose_mode=False)
        metric.measure(case)
        return {"key": "faithfulness", "score": metric.score}

    return faithfulness_evaluator


def make_relevancy_evaluator(judge):
    def relevancy_evaluator(run, example) -> dict:
        if not example.inputs.get("in_corpus", True):
            return {"key": "de_answer_relevancy", "score": None, "comment": "skipped — out-of-corpus"}

        answer = run.outputs["answer"]
        contexts = run.outputs["contexts"]
        question = example.inputs["question"]
        reference = example.outputs["reference"]

        case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=reference,
            retrieval_context=contexts,
        )
        metric = AnswerRelevancyMetric(threshold=RELEVANCY_THRESHOLD, model=judge, verbose_mode=False)
        metric.measure(case)
        return {"key": "de_answer_relevancy", "score": metric.score}

    return relevancy_evaluator


def abstention_evaluator(run, example) -> dict:
    if example.inputs.get("in_corpus", True):
        return {"key": "abstention", "score": None, "comment": "skipped — in-corpus"}
    answer = run.outputs["answer"].lower()
    passed = any(m in answer for m in ABSTENTION_MARKERS)
    return {"key": "abstention", "score": 1.0 if passed else 0.0}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    client = Client()
    ensure_dataset(client, golden)

    judge = make_judge()
    judge_label = "ollama" if os.getenv("USE_LOCAL_JUDGE") == "1" else "gemini"
    chat_label = CHAT_MODEL.replace(":", "-").replace(".", "_")

    # Experiment title encodes the key knobs so each run is self-describing in
    # the LangSmith UI: k=chunks, chunk size, generation model, judge model.
    experiment_prefix = (
        f"k{RETRIEVAL_K}"
        f"-chunk{CHUNK_SIZE}o{CHUNK_OVERLAP}"
        f"-{chat_label}"
        f"-judge-{judge_label}"
    )

    # All pipeline parameters are logged as metadata columns in LangSmith.
    # In the UI: open an experiment → Columns → metadata.<key> to add them.
    experiment_metadata = {
        "retrieval_k": RETRIEVAL_K,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embed_model": EMBED_MODEL,
        "chat_model": CHAT_MODEL,
        "temperature": TEMPERATURE,
        "judge_model": LOCAL_JUDGE_MODEL if os.getenv("USE_LOCAL_JUDGE") == "1" else JUDGE_MODEL,
        "faithfulness_threshold": FAITHFULNESS_THRESHOLD,
        "relevancy_threshold": RELEVANCY_THRESHOLD,
    }

    print(f"\nRunning LangSmith experiment: {experiment_prefix!r}")
    print(f"  Parameters: {experiment_metadata}")
    results = ls_evaluate(
        rag_target,
        data=DATASET_NAME,
        evaluators=[
            make_faithfulness_evaluator(judge),
            make_relevancy_evaluator(judge),
            abstention_evaluator,
        ],
        experiment_prefix=experiment_prefix,
        metadata=experiment_metadata,
        max_concurrency=1,  # sequential — Ollama is single-threaded
    )

    # Print per-question summary
    print("\n===== RESULTS =====")
    for r in results:
        try:
            q = r["example"].inputs.get("question", "?")
            eval_results = (r["evaluation_results"] or {}).get("results", []) or []
            evals = {e.key: e.score for e in eval_results}
            faith = evals.get("faithfulness")
            relev = evals.get("de_answer_relevancy")
            abst = evals.get("abstention")
            if faith is not None:
                faith_str = f"{faith:.2f}" if faith is not None else "n/a"
                relev_str = f"{relev:.2f}" if relev is not None else "n/a"
                print(f"  faithfulness={faith_str}  relevancy={relev_str}  | {q}")
            elif abst is not None:
                flag = "ABSTAINED" if abst == 1.0 else "HALLUCINATED"
                print(f"  abstention={flag}  | {q}")
            else:
                print(f"  (no scores — evaluator may have errored)  | {q}")
        except Exception as e:
            print(f"  (print error: {e})")

    print("\nView experiment in LangSmith:")
    print("  https://smith.langchain.com → Datasets & Experiments → zephyr-golden-qa")


if __name__ == "__main__":
    main()
