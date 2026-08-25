"""
STEP 7 (EVALUATION) — PART 3: DEEPEVAL
=======================================
DeepEval is a pytest-style LLM eval framework. Each question becomes an
LLMTestCase; metrics score it independently. Compared with RAGAS it makes
fewer sub-calls per metric, supports result caching, and surfaces per-case
pass/fail reasons — easier to debug.

Metrics used (same signals as eval_ragas.py, fewer API calls):
  - FaithfulnessMetric       — answer grounded in retrieved context?
  - AnswerRelevancyMetric    — answer actually addresses the question?
  - ContextualRecallMetric   — context covers the reference answer?
  - ContextualPrecisionMetric— retrieved chunks are on-topic?

Out-of-corpus questions skip the four metrics and use abstention detection
(same rule-based check as eval_custom.py).

Run:  python src/eval_deepeval.py
"""

import json
import os
import sys
from dotenv import load_dotenv
load_dotenv()

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.models import DeepEvalBaseLLM, OllamaModel
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from langchain_google_genai import ChatGoogleGenerativeAI

from rag_chain import answer_with_context, JUDGE_MODEL, LOCAL_JUDGE_MODEL


# ---------------------------------------------------------------------------
# DeepEval requires models to subclass DeepEvalBaseLLM.
# For Gemini we wrap ChatGoogleGenerativeAI; for Ollama we use the built-in
# OllamaModel. Set USE_LOCAL_JUDGE=1 in .env to use Ollama (free, no quota).
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
# Abstention check (mirrors eval_custom.py)
# ---------------------------------------------------------------------------
ABSTENTION_MARKERS = [
    "don't know", "do not know", "not in", "no information",
    "cannot", "can't", "not contain", "not available",
]

def abstained(answer: str) -> bool:
    low = answer.lower()
    return any(m in low for m in ABSTENTION_MARKERS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    with open("eval/golden_qa.json") as f:
        golden = json.load(f)

    judge = make_judge()

    metrics = [
        FaithfulnessMetric(threshold=0.7, model=judge, verbose_mode=False),
        AnswerRelevancyMetric(threshold=0.7, model=judge, verbose_mode=False),
        ContextualRecallMetric(threshold=0.7, model=judge, verbose_mode=False),
        ContextualPrecisionMetric(threshold=0.7, model=judge, verbose_mode=False),
    ]

    in_corpus_cases = []
    out_of_corpus_results = []

    print("Running RAG pipeline over golden dataset...")
    for item in golden:
        answer, contexts = answer_with_context(item["question"])

        if item["in_corpus"]:
            case = LLMTestCase(
                input=item["question"],
                actual_output=answer,
                expected_output=item["reference"],
                retrieval_context=contexts,
            )
            in_corpus_cases.append((item, case))
        else:
            passed = abstained(answer)
            out_of_corpus_results.append((item, answer, passed))
            flag = "OK" if passed else "HALLUCINATED"
            print(f"[out] abstention={'Y' if passed else 'N'} ({flag}) :: {item['question']}")
            if not passed:
                print(f"       -> model said: {answer[:120]}")

    # Run DeepEval on in-corpus cases
    print(f"\nScoring {len(in_corpus_cases)} in-corpus cases with DeepEval...\n")
    test_cases = [case for _, case in in_corpus_cases]
    results = evaluate(
        test_cases,
        metrics,
        async_config=AsyncConfig(run_async=False),
        display_config=DisplayConfig(print_results=False, inspect_after_run=False),
    )

    # Per-question summary
    print("\n===== PER-QUESTION RESULTS =====")
    for (item, _), test_result in zip(in_corpus_cases, results.test_results):
        scores = {r.name: (r.score, r.success) for r in test_result.metrics_data}
        passed = all(s for _, s in scores.values())
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {item['question']}")
        for name, (score, ok) in scores.items():
            short = name.replace("Metric", "").strip()
            flag = "✓" if ok else "✗"
            print(f"       {flag} {short:<28} {score:.2f}")
            if not ok:
                reason = next((r.reason for r in test_result.metrics_data if r.name == name), "")
                if reason:
                    print(f"         reason: {reason[:120]}")

    # Aggregate summary — group by the name DeepEval reports, not the class name
    print("\n===== SUMMARY =====")
    from collections import defaultdict
    metric_rows = defaultdict(list)
    for _, tr in zip(in_corpus_cases, results.test_results):
        for r in tr.metrics_data:
            metric_rows[r.name].append((r.score, r.success, r.threshold))
    for name, rows in metric_rows.items():
        avg = sum(s for s, _, _ in rows) / len(rows)
        n_pass = sum(1 for _, ok, _ in rows if ok)
        threshold = rows[0][2]
        print(f"{name:<32} avg: {avg:.2f}  pass: {n_pass}/{len(rows)}  (threshold: {threshold})")

    n_out = len(out_of_corpus_results)
    n_abstained = sum(1 for _, _, p in out_of_corpus_results if p)
    print(f"\nAbstention (no hallucination): {n_abstained}/{n_out} = "
          f"{n_abstained/n_out:.0%}" if n_out else "No out-of-corpus cases.")


if __name__ == "__main__":
    main()
