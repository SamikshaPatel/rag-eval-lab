"""
Score a single golden question with all DeepEval metrics (no LangSmith upload).
Used to get formal metric scores for questions that need post-patch validation.

Run:
    USE_LOCAL_JUDGE=1 python3 src/score_single_question.py NSB-034
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()

from config import (
    NORTHSTAR_GOLDEN_PATH,
    FAITHFULNESS_THRESHOLD, RELEVANCY_THRESHOLD,
    CONTEXTUAL_PRECISION_THRESHOLD, CONTEXTUAL_RECALL_THRESHOLD,
    CONTEXTUAL_RELEVANCY_THRESHOLD, HALLUCINATION_THRESHOLD,
    BIAS_THRESHOLD, TOXICITY_THRESHOLD,
    CORRECTNESS_THRESHOLD, COMPLETENESS_THRESHOLD,
    PROMPT_JUDGE_CORRECTNESS, PROMPT_JUDGE_COMPLETENESS,
    _load_prompt,
)
from judgeUtil import make_judge

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric, AnswerRelevancyMetric,
    ContextualPrecisionMetric, ContextualRecallMetric,
    ContextualRelevancyMetric, HallucinationMetric,
    BiasMetric, ToxicityMetric,
)

# Import pipeline from eval script (includes prompt v2 + multi-query fix)
from eval_langsmith_northstar import answer_with_context

import re

TARGET_ID = sys.argv[1] if len(sys.argv) > 1 else "NSB-034"


def judge_with_json_fallback(judge, prompt: str) -> float:
    result = judge.generate(prompt)
    raw = (result[0] if isinstance(result, tuple) else result).strip()
    m = re.search(r'\{.*?"score"\s*:\s*([0-9.]+).*?\}', raw, re.DOTALL)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            pass
    m = re.search(r'\b([01](?:\.\d+)?)\b', raw)
    return max(0.0, min(1.0, float(m.group(1)))) if m else 0.0


def main():
    with open(NORTHSTAR_GOLDEN_PATH) as f:
        questions = [json.loads(l) for l in f if l.strip()]
    target = next((q for q in questions if q["id"] == TARGET_ID), None)
    if not target:
        print(f"Question {TARGET_ID} not found.")
        sys.exit(1)

    judge = make_judge()
    correctness_prompt  = _load_prompt(PROMPT_JUDGE_CORRECTNESS)
    completeness_prompt = _load_prompt(PROMPT_JUDGE_COMPLETENESS)

    q        = target["question"]
    expected = target["expected_answer"]

    print(f"\nRunning pipeline for {TARGET_ID}...")
    answer, contexts = answer_with_context(q)
    print(f"  Answer   : {answer.strip()}")
    print(f"  Expected : {expected}")
    print(f"  Contexts : {len(contexts)} retrieved\n")

    case = LLMTestCase(
        input=q,
        actual_output=answer,
        expected_output=expected,
        retrieval_context=contexts,
        context=contexts,
    )

    metrics = [
        ("de_contextual_precision",  ContextualPrecisionMetric(threshold=CONTEXTUAL_PRECISION_THRESHOLD,  model=judge, verbose_mode=False)),
        ("de_contextual_recall",     ContextualRecallMetric(threshold=CONTEXTUAL_RECALL_THRESHOLD,         model=judge, verbose_mode=False)),
        ("de_contextual_relevancy",  ContextualRelevancyMetric(threshold=CONTEXTUAL_RELEVANCY_THRESHOLD,   model=judge, verbose_mode=False)),
        ("de_faithfulness",          FaithfulnessMetric(threshold=FAITHFULNESS_THRESHOLD,                  model=judge, verbose_mode=False)),
        ("de_answer_relevancy",      AnswerRelevancyMetric(threshold=RELEVANCY_THRESHOLD,                  model=judge, verbose_mode=False)),
        ("de_hallucination",         HallucinationMetric(threshold=HALLUCINATION_THRESHOLD,                model=judge, verbose_mode=False)),
        ("de_bias",                  BiasMetric(threshold=BIAS_THRESHOLD,                                  model=judge, verbose_mode=False)),
        ("de_toxicity",              ToxicityMetric(threshold=TOXICITY_THRESHOLD,                          model=judge, verbose_mode=False)),
    ]

    scores = {}
    for key, metric in metrics:
        try:
            metric.measure(case)
            scores[key] = round(metric.score, 2)
            status = "PASS" if metric.score >= metric.threshold else "FAIL"
            print(f"  {key:<30} {metric.score:.2f}  [{status}]")
        except Exception as e:
            scores[key] = None
            print(f"  {key:<30} ERROR: {e}")

    # Custom correctness + completeness
    for key, prompt_tpl in [("de_answer_correctness", correctness_prompt),
                             ("de_answer_completeness", completeness_prompt)]:
        try:
            prompt = prompt_tpl.format(question=q, reference=expected, actual=answer)
            score  = judge_with_json_fallback(judge, prompt)
            scores[key] = round(score, 2)
            print(f"  {key:<30} {score:.2f}")
        except Exception as e:
            scores[key] = None
            print(f"  {key:<30} ERROR: {e}")

    print(f"\n{'='*55}")
    print(f"  Summary for {TARGET_ID}:")
    print(f"    precision={scores.get('de_contextual_precision')}  recall={scores.get('de_contextual_recall')}  relevancy={scores.get('de_contextual_relevancy')}")
    print(f"    faithfulness={scores.get('de_faithfulness')}  answer_relevancy={scores.get('de_answer_relevancy')}  hallucination={scores.get('de_hallucination')}")
    print(f"    correctness={scores.get('de_answer_correctness')}  completeness={scores.get('de_answer_completeness')}")
    print(f"    bias={scores.get('de_bias')}  toxicity={scores.get('de_toxicity')}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
