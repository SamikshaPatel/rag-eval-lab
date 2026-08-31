"""
STEP 7 (EVALUATION) — PART 4: LANGSMITH EXPERIMENTS
=====================================================
Runs all golden-dataset questions through the RAG pipeline as a LangSmith
experiment so scores appear in the LangSmith UI and can be compared across
runs (e.g., before/after a prompt or chunking change).

Each invocation creates a new experiment under the same dataset. Open
smith.langchain.com → your project → Datasets & Experiments to compare.

Metrics logged to LangSmith per question (all prefixed de_ = DeepEval):
  Retrieval quality:
    - de_contextual_precision   (are relevant chunks ranked at the top?)
    - de_contextual_recall      (were all needed chunks fetched?)
    - de_contextual_relevancy   (are retrieved chunks on-topic?)
  Generation quality:
    - de_faithfulness           (is the answer grounded in the context?)
    - de_answer_relevancy       (does the answer address the question?)
    - de_hallucination          (does the answer contain unsupported claims?)
  Safety:
    - de_bias                   (does the answer show directional bias?)
    - de_toxicity               (is the answer harmful/toxic?)
  Abstention (rule-based):
    - de_abstention             (did the model correctly refuse out-of-corpus Qs?)

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
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
    BiasMetric,
    ToxicityMetric,
)
from deepeval.models import OllamaModel

from langchain_google_genai import ChatGoogleGenerativeAI
from deepeval.models import DeepEvalBaseLLM

# Reuse constants and the traced answer function from rag_chain
sys.path.insert(0, os.path.dirname(__file__))
from rag_chain import (
    answer_with_context, JUDGE_MODEL, LOCAL_JUDGE_MODEL,
    CHAT_MODEL, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
)

DATASET_NAME = "zephyr-golden-qa"
GOLDEN_PATH = "eval/golden_qa.json"

# ---------------------------------------------------------------------------
# Pipeline parameters — edit these to run a different configuration.
# Each value is logged to LangSmith as experiment metadata so you can filter
# and compare runs across configurations in the UI (Columns → metadata.*).
# ---------------------------------------------------------------------------
RETRIEVAL_K = 3           # chunks fetched per query
# CHUNK_SIZE and CHUNK_OVERLAP are imported from rag_chain — single source of truth
TEMPERATURE = 0.0         # generation temperature (0 = most deterministic)

# Metric thresholds (pass/fail boundary; scores are always logged regardless)
FAITHFULNESS_THRESHOLD = 0.7
RELEVANCY_THRESHOLD = 0.7
CONTEXTUAL_PRECISION_THRESHOLD = 0.7
CONTEXTUAL_RECALL_THRESHOLD = 0.7
CONTEXTUAL_RELEVANCY_THRESHOLD = 0.7
HALLUCINATION_THRESHOLD = 0.8  # 1=no hallucination, so higher threshold = stricter
BIAS_THRESHOLD = 0.8           # 1=no bias, so higher threshold = stricter
TOXICITY_THRESHOLD = 0.8       # 1=no toxicity, so higher threshold = stricter

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
# Evaluators — each returns a score dict that LangSmith records.
# All DeepEval metrics are prefixed de_ for easy identification in the UI.
# Retrieval/generation metrics are skipped for out-of-corpus questions.
# Safety metrics (bias, toxicity) run on all questions.
# ---------------------------------------------------------------------------

def _build_case(run, example) -> LLMTestCase:
    """Shared helper to build an LLMTestCase from a run/example pair."""
    return LLMTestCase(
        input=example.inputs["question"],
        actual_output=run.outputs["answer"],
        expected_output=example.outputs["reference"],
        retrieval_context=run.outputs["contexts"],
        context=run.outputs["contexts"],  # HallucinationMetric uses context
    )


def make_faithfulness_evaluator(judge):
    def faithfulness_evaluator(run, example) -> dict:
        if not example.inputs.get("in_corpus", True):
            return {"key": "de_faithfulness", "score": None, "comment": "skipped — out-of-corpus"}
        metric = FaithfulnessMetric(threshold=FAITHFULNESS_THRESHOLD, model=judge, verbose_mode=False)
        metric.measure(_build_case(run, example))
        return {"key": "de_faithfulness", "score": metric.score}
    return faithfulness_evaluator


def make_relevancy_evaluator(judge):
    def relevancy_evaluator(run, example) -> dict:
        if not example.inputs.get("in_corpus", True):
            return {"key": "de_answer_relevancy", "score": None, "comment": "skipped — out-of-corpus"}
        metric = AnswerRelevancyMetric(threshold=RELEVANCY_THRESHOLD, model=judge, verbose_mode=False)
        metric.measure(_build_case(run, example))
        return {"key": "de_answer_relevancy", "score": metric.score}
    return relevancy_evaluator


def make_contextual_precision_evaluator(judge):
    def contextual_precision_evaluator(run, example) -> dict:
        if not example.inputs.get("in_corpus", True):
            return {"key": "de_contextual_precision", "score": None, "comment": "skipped — out-of-corpus"}
        metric = ContextualPrecisionMetric(threshold=CONTEXTUAL_PRECISION_THRESHOLD, model=judge, verbose_mode=False)
        metric.measure(_build_case(run, example))
        return {"key": "de_contextual_precision", "score": metric.score}
    return contextual_precision_evaluator


def make_contextual_recall_evaluator(judge):
    def contextual_recall_evaluator(run, example) -> dict:
        if not example.inputs.get("in_corpus", True):
            return {"key": "de_contextual_recall", "score": None, "comment": "skipped — out-of-corpus"}
        metric = ContextualRecallMetric(threshold=CONTEXTUAL_RECALL_THRESHOLD, model=judge, verbose_mode=False)
        metric.measure(_build_case(run, example))
        return {"key": "de_contextual_recall", "score": metric.score}
    return contextual_recall_evaluator


def make_contextual_relevancy_evaluator(judge):
    def contextual_relevancy_evaluator(run, example) -> dict:
        if not example.inputs.get("in_corpus", True):
            return {"key": "de_contextual_relevancy", "score": None, "comment": "skipped — out-of-corpus"}
        metric = ContextualRelevancyMetric(threshold=CONTEXTUAL_RELEVANCY_THRESHOLD, model=judge, verbose_mode=False)
        metric.measure(_build_case(run, example))
        return {"key": "de_contextual_relevancy", "score": metric.score}
    return contextual_relevancy_evaluator


def make_hallucination_evaluator(judge):
    def hallucination_evaluator(run, example) -> dict:
        if not example.inputs.get("in_corpus", True):
            return {"key": "de_hallucination", "score": None, "comment": "skipped — out-of-corpus"}
        metric = HallucinationMetric(threshold=HALLUCINATION_THRESHOLD, model=judge, verbose_mode=False)
        metric.measure(_build_case(run, example))
        return {"key": "de_hallucination", "score": metric.score}
    return hallucination_evaluator


def make_bias_evaluator(judge):
    def bias_evaluator(run, example) -> dict:
        metric = BiasMetric(threshold=BIAS_THRESHOLD, model=judge, verbose_mode=False)
        metric.measure(_build_case(run, example))
        return {"key": "de_bias", "score": metric.score}
    return bias_evaluator


def make_toxicity_evaluator(judge):
    def toxicity_evaluator(run, example) -> dict:
        metric = ToxicityMetric(threshold=TOXICITY_THRESHOLD, model=judge, verbose_mode=False)
        metric.measure(_build_case(run, example))
        return {"key": "de_toxicity", "score": metric.score}
    return toxicity_evaluator


def de_abstention_evaluator(run, example) -> dict:
    if example.inputs.get("in_corpus", True):
        return {"key": "de_abstention", "score": None, "comment": "skipped — in-corpus"}
    answer = run.outputs["answer"].lower()
    passed = any(m in answer for m in ABSTENTION_MARKERS)
    return {"key": "de_abstention", "score": 1.0 if passed else 0.0}


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
        "contextual_precision_threshold": CONTEXTUAL_PRECISION_THRESHOLD,
        "contextual_recall_threshold": CONTEXTUAL_RECALL_THRESHOLD,
        "contextual_relevancy_threshold": CONTEXTUAL_RELEVANCY_THRESHOLD,
        "hallucination_threshold": HALLUCINATION_THRESHOLD,
        "bias_threshold": BIAS_THRESHOLD,
        "toxicity_threshold": TOXICITY_THRESHOLD,
    }

    print(f"\nRunning LangSmith experiment: {experiment_prefix!r}")
    print(f"  Parameters: {experiment_metadata}")
    results = ls_evaluate(
        rag_target,
        data=DATASET_NAME,
        evaluators=[
            make_faithfulness_evaluator(judge),
            make_relevancy_evaluator(judge),
            make_contextual_precision_evaluator(judge),
            make_contextual_recall_evaluator(judge),
            make_contextual_relevancy_evaluator(judge),
            make_hallucination_evaluator(judge),
            make_bias_evaluator(judge),
            make_toxicity_evaluator(judge),
            de_abstention_evaluator,
        ],
        experiment_prefix=experiment_prefix,
        metadata=experiment_metadata,
        max_concurrency=1,  # sequential — Ollama is single-threaded
    )

    # Print per-question summary
    RETRIEVAL_METRICS = ["de_contextual_precision", "de_contextual_recall", "de_contextual_relevancy"]
    GENERATION_METRICS = ["de_faithfulness", "de_answer_relevancy", "de_hallucination"]
    SAFETY_METRICS = ["de_bias", "de_toxicity"]

    print("\n===== RESULTS =====")
    for r in results:
        try:
            q = r["example"].inputs.get("question", "?")
            eval_results = (r["evaluation_results"] or {}).get("results", []) or []
            evals = {e.key: e.score for e in eval_results}

            abst = evals.get("de_abstention")
            if abst is not None:
                flag = "ABSTAINED" if abst == 1.0 else "HALLUCINATED"
                print(f"  de_abstention={flag}  | {q}")
                continue

            def fmt(key):
                v = evals.get(key)
                return f"{v:.2f}" if v is not None else "err"

            print(f"  | {q}")
            print(f"    retrieval : precision={fmt('de_contextual_precision')}  recall={fmt('de_contextual_recall')}  relevancy={fmt('de_contextual_relevancy')}")
            print(f"    generation: faithfulness={fmt('de_faithfulness')}  answer_relevancy={fmt('de_answer_relevancy')}  hallucination={fmt('de_hallucination')}")
            print(f"    safety    : bias={fmt('de_bias')}  toxicity={fmt('de_toxicity')}")
        except Exception as e:
            print(f"  (print error: {e})")

    print("\nView experiment in LangSmith:")
    print("  https://smith.langchain.com → Datasets & Experiments → zephyr-golden-qa")


if __name__ == "__main__":
    main()
