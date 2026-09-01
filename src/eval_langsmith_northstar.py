"""
NORTHSTAR — LANGSMITH EVALUATION
=================================
Creates the 'northstar-golden-qa' dataset in LangSmith (44 questions:
36 in-corpus, 8 out-of-domain) and runs a named experiment logging
10 DeepEval metrics per question.

The Northstar corpus lives in its own Chroma DB (chroma_db_northstar/)
so it is fully isolated from the Zephyr corpus.

Run:
    python src/eval_langsmith_northstar.py
    USE_LOCAL_JUDGE=1 python src/eval_langsmith_northstar.py   # Ollama judge
"""

import json
import logging
import os
import sys
import threading
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
    # Structured error logger — writes to eval_errors.log separate from stdout.
    # Each entry includes a timestamp, the metric key, the question, and the
    # exception so failures are auditable without digging through mixed stdout.
# ---------------------------------------------------------------------------
_error_log = logging.getLogger("eval_errors_northstar")
_error_log.setLevel(logging.ERROR)
_error_log.propagate = False # don't echo to root logger / stdout
_fh = logging.FileHandler("eval_errors_northstar.log")
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
_error_log.addHandler(_fh)


# Fail-fast flag: set to the first fatal exception; all subsequent evaluators
 # check this and raise immediately instead of calling the judge again.
_fatal_error: Exception | None = None
_fatal_lock = threading.Lock()

def _check_fatal():
    """Raise if a fatal model error was already recorded — enables early exit."""
    if _fatal_error is not None:
        raise _fatal_error


def _record_fatal(exc: Exception, metric_key: str, question: str) -> None:
    """Set the fail-fast flag (once) and write a structured error log entry."""
    global _fatal_error
    with _fatal_lock:
        if _fatal_error is None:
            _fatal_error = exc
    _error_log.error(
        "metric=%s question=%r error=%s: %s",
        metric_key, question, type(exc).__name__, exc,
    )


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

from langchain_google_genai.chat_models import GoogleRateLimitError
from google.genai.errors import ClientError

# Model errors that should abort the entire experiment immediately.
_FATAL_EXCEPTIONS = (GoogleRateLimitError, ClientError)


def _safe_evaluate(key: str, question: str, fn):
    """
   Run fn() — a callable that calls the judge and returns a score dict.
   On rate-limit or API errors: log to eval_errors.log, set the fail-fast
   flag, and re-raise so LangSmith surfaces the failure instead of silently
   continuing with broken scores.
   """
    _check_fatal()
    try:
        return fn()
    except _FATAL_EXCEPTIONS as exc:
        _record_fatal(exc, key, question)
        raise


sys.path.insert(0, os.path.dirname(__file__))
from config import (
    JUDGE_MODEL, LOCAL_JUDGE_MODEL,
    CHAT_MODEL, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
    RETRIEVAL_K, TEMPERATURE,
    FAITHFULNESS_THRESHOLD, RELEVANCY_THRESHOLD,
    CONTEXTUAL_PRECISION_THRESHOLD, CONTEXTUAL_RECALL_THRESHOLD,
    CONTEXTUAL_RELEVANCY_THRESHOLD, HALLUCINATION_THRESHOLD,
    BIAS_THRESHOLD, TOXICITY_THRESHOLD,
    CORRECTNESS_THRESHOLD, COMPLETENESS_THRESHOLD,
    PROMPT_JUDGE_CORRECTNESS, PROMPT_JUDGE_COMPLETENESS,
    NORTHSTAR_DB_PATH, NORTHSTAR_GOLDEN_PATH, NORTHSTAR_DATASET_NAME,
    _load_prompt,
)
from judgeUtil import make_judge, abstained as _abstained

# ---------------------------------------------------------------------------
# Northstar retriever — points at the isolated Northstar Chroma DB
# ---------------------------------------------------------------------------
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

_embeddings = OllamaEmbeddings(model=EMBED_MODEL)
_vectorstore = Chroma(persist_directory=NORTHSTAR_DB_PATH, embedding_function=_embeddings)


def _get_retriever(k: int = RETRIEVAL_K):
    return _vectorstore.as_retriever(search_kwargs={"k": k})


_RAG_PROMPT = ChatPromptTemplate.from_template(_load_prompt("rag_grounding_v1.txt"))
_llm = ChatOllama(model=CHAT_MODEL, temperature=TEMPERATURE)


@traceable(run_type="chain", name="northstar-rag-answer")
def _answer_northstar(question: str, k: int = RETRIEVAL_K) -> str:
    retriever = _get_retriever(k=k)
    chain = (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
         "question": RunnablePassthrough()}
        | _RAG_PROMPT
        | _llm
        | StrOutputParser()
    )
    return chain.invoke(question)


def answer_with_context(question: str, k: int = RETRIEVAL_K):
    retriever = _get_retriever(k=k)
    contexts = retriever.invoke(question)
    answer = _answer_northstar(question, k=k)
    return answer, [c.page_content for c in contexts]


# ---------------------------------------------------------------------------
# LangSmith dataset helpers
# ---------------------------------------------------------------------------
def load_golden() -> list:
    with open(NORTHSTAR_GOLDEN_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def ensure_dataset(client: Client, golden: list) -> str:
    existing = [d.name for d in client.list_datasets()]
    if NORTHSTAR_DATASET_NAME in existing:
        print(f"[dataset] '{NORTHSTAR_DATASET_NAME}' already exists — reusing.")
        return NORTHSTAR_DATASET_NAME

    print(f"[dataset] Creating '{NORTHSTAR_DATASET_NAME}' with {len(golden)} examples...")
    dataset = client.create_dataset(
        NORTHSTAR_DATASET_NAME,
        description=(
            "44-question golden set for Northstar Digital Bank RAG eval "
            "(36 in-corpus, 8 out-of-domain). Categories: factual, procedural, "
            "comparison, multi_hop, exception, ood."
        ),
    )
    client.create_examples(
        inputs=[{
            "question":    q["question"],
            "in_corpus":   q["in_corpus"],
            "id":          q["id"],
            "category":    q["category"],
            "difficulty":  q["difficulty"],
            "criticality": q["criticality"],
        } for q in golden],
        outputs=[{
            "reference":       q["expected_answer"],
            "source_section":  q.get("source_section"),
        } for q in golden],
        dataset_id=dataset.id,
    )
    print(f"[dataset] Created {len(golden)} examples (id={dataset.id}).")
    return NORTHSTAR_DATASET_NAME


# ---------------------------------------------------------------------------
# Target function -  what LangSmith runs for each example
# ---------------------------------------------------------------------------
def rag_target(inputs: dict) -> dict:
    question = inputs["question"]
    answer, contexts = answer_with_context(question, k=RETRIEVAL_K)
    return {"answer": answer, "contexts": contexts}


# ---------------------------------------------------------------------------
# Shared test-case builder
# ---------------------------------------------------------------------------
def _build_case(run, example) -> LLMTestCase:
    return LLMTestCase(
        input=example.inputs["question"],
        actual_output=run.outputs["answer"],
        expected_output=example.outputs["reference"],
        retrieval_context=run.outputs["contexts"],
        context=run.outputs["contexts"],
    )


# ---------------------------------------------------------------------------
# Evaluator factories (identical pattern to eval_langsmith.py)
# ---------------------------------------------------------------------------
def make_faithfulness_evaluator(judge):
    def _eval(run, example):
        if not example.inputs.get("in_corpus", True):
            return {"key": "de_faithfulness", "score": None, "comment": "skipped — ood"}
        q = example.inputs.get("question", "?")
        def _run():
            m = FaithfulnessMetric(threshold=FAITHFULNESS_THRESHOLD, model=judge, verbose_mode=False)
            m.measure(_build_case(run, example))
            return {"key": "de_faithfulness", "score": m.score}
        return _safe_evaluate("de_faithfulness", q, _run)
    return _eval


def make_relevancy_evaluator(judge):
    def _eval(run, example):
        if not example.inputs.get("in_corpus", True):
            return {"key": "de_answer_relevancy", "score": None, "comment": "skipped — ood"}
        q = example.inputs.get("question", "?")
        def _run():
            m = AnswerRelevancyMetric(threshold=RELEVANCY_THRESHOLD, model=judge, verbose_mode=False)
            m.measure(_build_case(run, example))
            return {"key": "de_answer_relevancy", "score": m.score}
        return _safe_evaluate("de_answer_relevancy", q, _run)
    return _eval


def make_contextual_precision_evaluator(judge):
    def _eval(run, example):
        if not example.inputs.get("in_corpus", True):
            return {"key": "de_contextual_precision", "score": None, "comment": "skipped — ood"}
        q = example.inputs.get("question", "?")
        def _run():
            m = ContextualPrecisionMetric(threshold=CONTEXTUAL_PRECISION_THRESHOLD, model=judge, verbose_mode=False)
            m.measure(_build_case(run, example))
            return {"key": "de_contextual_precision", "score": m.score}
        return _safe_evaluate("de_contextual_precision", q, _run)
    return _eval


def make_contextual_recall_evaluator(judge):
    def _eval(run, example):
        if not example.inputs.get("in_corpus", True):
            return {"key": "de_contextual_recall", "score": None, "comment": "skipped — ood"}
        q = example.inputs.get("question", "?")
        def _run():
            m = ContextualRecallMetric(threshold=CONTEXTUAL_RECALL_THRESHOLD, model=judge, verbose_mode=False)
            m.measure(_build_case(run, example))
            return {"key": "de_contextual_recall", "score": m.score}
        return _safe_evaluate("de_contextual_recall", q, _run)
    return _eval


def make_contextual_relevancy_evaluator(judge):
    def _eval(run, example):
        if not example.inputs.get("in_corpus", True):
            return {"key": "de_contextual_relevancy", "score": None, "comment": "skipped — ood"}
        q = example.inputs.get("question", "?")
        def _run():
            m = ContextualRelevancyMetric(threshold=CONTEXTUAL_RELEVANCY_THRESHOLD, model=judge, verbose_mode=False)
            m.measure(_build_case(run, example))
            return {"key": "de_contextual_relevancy", "score": m.score}
        return _safe_evaluate("de_contextual_relevancy", q, _run)
    return _eval


def make_hallucination_evaluator(judge):
    def _eval(run, example):
        if not example.inputs.get("in_corpus", True):
            return {"key": "de_hallucination", "score": None, "comment": "skipped — ood"}
        q = example.inputs.get("question", "?")
        def _run():
            m = HallucinationMetric(threshold=HALLUCINATION_THRESHOLD, model=judge, verbose_mode=False)
            m.measure(_build_case(run, example))
            return {"key": "de_hallucination", "score": m.score}
        return _safe_evaluate("de_hallucination", q, _run)
    return _eval


def make_bias_evaluator(judge):
    def _eval(run, example):
        q = example.inputs.get("question", "?")
        def _run():
            m = BiasMetric(threshold=BIAS_THRESHOLD, model=judge, verbose_mode=False)
            m.measure(_build_case(run, example))
            return {"key": "de_bias", "score": m.score}
        return _safe_evaluate("de_bias", q, _run)
    return _eval


def make_toxicity_evaluator(judge):
    def _eval(run, example):
        q = example.inputs.get("question", "?")
        def _run():
            m = ToxicityMetric(threshold=TOXICITY_THRESHOLD, model=judge, verbose_mode=False)
            m.measure(_build_case(run, example))
            return {"key": "de_toxicity", "score": m.score}
        return _safe_evaluate("de_toxicity", q, _run)
    return _eval


def _judge_with_json_fallback(judge, prompt: str) -> float:
    import re as _re
    result = judge.generate(prompt)
    raw = (result[0] if isinstance(result, tuple) else result).strip()
    json_match = _re.search(r'\{.*?"score"\s*:\s*([0-9.]+).*?\}', raw, _re.DOTALL)
    if json_match:
        try:
            return max(0.0, min(1.0, float(json_match.group(1))))
        except ValueError:
            pass
    num_match = _re.search(r'\b([01](?:\.\d+)?)\b', raw)
    if num_match:
        return max(0.0, min(1.0, float(num_match.group(1))))
    return 0.0


_CORRECTNESS_PROMPT  = _load_prompt(PROMPT_JUDGE_CORRECTNESS)
_COMPLETENESS_PROMPT = _load_prompt(PROMPT_JUDGE_COMPLETENESS)


def make_correctness_evaluator(judge):
    def _eval(run, example):
        if not example.inputs.get("in_corpus", True):
            return {"key": "de_answer_correctness", "score": None, "comment": "skipped — ood"}
        q = example.inputs["question"]
        def _run():
            prompt = _CORRECTNESS_PROMPT.format(
                question=q,
                reference=example.outputs["reference"],
                actual=run.outputs["answer"],
            )
            score = _judge_with_json_fallback(judge, prompt)
            return {"key": "de_answer_correctness", "score": score,
                    "comment": f"pass={score >= CORRECTNESS_THRESHOLD}"}
        return _safe_evaluate("de_answer_correctness", q, _run)
    return _eval


def make_completeness_evaluator(judge):
    def _eval(run, example):
        if not example.inputs.get("in_corpus", True):
            return {"key": "de_answer_completeness", "score": None, "comment": "skipped — ood"}
        q = example.inputs["question"]
        def _run():
            prompt = _COMPLETENESS_PROMPT.format(
                question=q,
                reference=example.outputs["reference"],
                actual=run.outputs["answer"],
            )
            score = _judge_with_json_fallback(judge, prompt)
            return {"key": "de_answer_completeness", "score": score,
                    "comment": f"pass={score >= COMPLETENESS_THRESHOLD}"}
        return _safe_evaluate("de_answer_completeness", q, _run)
    return _eval


def de_abstention_evaluator(run, example) -> dict:
    if example.inputs.get("in_corpus", True):
        return {"key": "de_abstention", "score": None, "comment": "skipped — in-corpus"}
    answer = run.outputs["answer"].lower()
    passed = _abstained(answer)
    return {"key": "de_abstention", "score": 1.0 if passed else 0.0}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    golden = load_golden()
    client = Client()
    ensure_dataset(client, golden)

    judge = make_judge()
    judge_label = "ollama" if os.getenv("USE_LOCAL_JUDGE") == "1" else "gemini"
    chat_label  = CHAT_MODEL.replace(":", "-").replace(".", "_")

    experiment_prefix = (
        f"northstar"
        f"-k{RETRIEVAL_K}"
        f"-chunk{CHUNK_SIZE}o{CHUNK_OVERLAP}"
        f"-{chat_label}"
        f"-judge-{judge_label}"
    )

    experiment_metadata = {
        "corpus":           "northstar",
        "retrieval_k":      RETRIEVAL_K,
        "chunk_size":       CHUNK_SIZE,
        "chunk_overlap":    CHUNK_OVERLAP,
        "embed_model":      EMBED_MODEL,
        "chat_model":       CHAT_MODEL,
        "temperature":      TEMPERATURE,
        "judge_model":      LOCAL_JUDGE_MODEL if os.getenv("USE_LOCAL_JUDGE") == "1" else JUDGE_MODEL,
        "n_total":          len(golden),
        "n_in_corpus":      sum(1 for q in golden if q["in_corpus"]),
        "n_ood":            sum(1 for q in golden if not q["in_corpus"]),
    }

    print(f"\nRunning LangSmith experiment: {experiment_prefix!r}")
    print(f"  {experiment_metadata['n_in_corpus']} in-corpus questions, "
          f"{experiment_metadata['n_ood']} out-of-domain questions")

    results = ls_evaluate(
        rag_target,
        data=NORTHSTAR_DATASET_NAME,
        evaluators=[
            make_faithfulness_evaluator(judge),
            make_relevancy_evaluator(judge),
            make_contextual_precision_evaluator(judge),
            make_contextual_recall_evaluator(judge),
            make_contextual_relevancy_evaluator(judge),
            make_hallucination_evaluator(judge),
            make_bias_evaluator(judge),
            make_toxicity_evaluator(judge),
            make_correctness_evaluator(judge),
            make_completeness_evaluator(judge),
            de_abstention_evaluator,
        ],
        experiment_prefix=experiment_prefix,
        metadata=experiment_metadata,
        max_concurrency=1,
    )

    # Per-question summary
    print("\n===== RESULTS =====")
    for r in results:
        try:
            q    = r["example"].inputs.get("question", "?")
            cat  = r["example"].inputs.get("category", "")
            diff = r["example"].inputs.get("difficulty", "")
            evals = {e.key: e.score for e in (r["evaluation_results"] or {}).get("results", []) or []}

            abst = evals.get("de_abstention")
            if abst is not None:
                flag = "ABSTAINED" if abst == 1.0 else "HALLUCINATED"
                print(f"  [ood/{diff}] de_abstention={flag}  | {q}")
                continue

            def fmt(key):
                v = evals.get(key)
                return f"{v:.2f}" if v is not None else "err"

            print(f"  [{cat}/{diff}] {q}")
            print(f"    retrieval : precision={fmt('de_contextual_precision')}  recall={fmt('de_contextual_recall')}  relevancy={fmt('de_contextual_relevancy')}")
            print(f"    generation: faithfulness={fmt('de_faithfulness')}  answer_relevancy={fmt('de_answer_relevancy')}  hallucination={fmt('de_hallucination')}")
            print(f"    reference : correctness={fmt('de_answer_correctness')}  completeness={fmt('de_answer_completeness')}")
            print(f"    safety    : bias={fmt('de_bias')}  toxicity={fmt('de_toxicity')}")
        except Exception as e:
            print(f"  (print error: {e})")

    print("\nView experiment in LangSmith:")
    print(f"  https://smith.langchain.com → Datasets & Experiments → {NORTHSTAR_DATASET_NAME}")


if __name__ == "__main__":
    main()
