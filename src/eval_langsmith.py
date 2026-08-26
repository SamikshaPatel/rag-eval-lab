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
from rag_chain import answer_with_context, JUDGE_MODEL, LOCAL_JUDGE_MODEL

DATASET_NAME = "zephyr-golden-qa"
GOLDEN_PATH = "eval/golden_qa.json"

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
    answer, contexts = answer_with_context(question)
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
        metric = FaithfulnessMetric(threshold=0.7, model=judge, verbose_mode=False)
        metric.measure(case)
        return {"key": "faithfulness", "score": metric.score}

    return faithfulness_evaluator


def make_relevancy_evaluator(judge):
    def relevancy_evaluator(run, example) -> dict:
        if not example.inputs.get("in_corpus", True):
            return {"key": "answer_relevancy", "score": None, "comment": "skipped — out-of-corpus"}

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
        metric = AnswerRelevancyMetric(threshold=0.7, model=judge, verbose_mode=False)
        metric.measure(case)
        return {"key": "answer_relevancy", "score": metric.score}

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

    print("\nRunning LangSmith experiment...")
    results = ls_evaluate(
        rag_target,
        data=DATASET_NAME,
        evaluators=[
            make_faithfulness_evaluator(judge),
            make_relevancy_evaluator(judge),
            abstention_evaluator,
        ],
        experiment_prefix="golden-eval",
        max_concurrency=1,  # sequential — Ollama is single-threaded
    )

    # Print per-question summary
    print("\n===== RESULTS =====")
    for r in results:
        try:
            q = r.example.inputs.get("question", "?")
            eval_results = getattr(r.evaluation_results, "results", []) or []
            evals = {e.key: e.score for e in eval_results}
            faith = evals.get("faithfulness")
            relev = evals.get("answer_relevancy")
            abst = evals.get("abstention")
            if faith is not None:
                print(f"  faithfulness={faith:.2f}  relevancy={relev:.2f}  | {q}")
            elif abst is not None:
                flag = "ABSTAINED" if abst == 1.0 else "HALLUCINATED"
                print(f"  abstention={flag}  | {q}")
        except Exception:
            pass

    print("\nView experiment in LangSmith:")
    print("  https://smith.langchain.com → Datasets & Experiments → zephyr-golden-qa")


if __name__ == "__main__":
    main()
