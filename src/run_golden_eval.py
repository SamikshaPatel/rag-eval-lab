"""
Run all golden-dataset questions as a LangSmith experiment and let LangSmith's
configured online evaluator (Answer Relevancy) score the runs automatically.

No local judge is needed — LangSmith applies the evaluator defined in the UI.

Run:
    python src/run_golden_eval.py

Results appear in:
    smith.langchain.com → Datasets & Experiments → zephyr-golden-qa
"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()

from langsmith import Client
from langsmith.evaluation import evaluate as ls_evaluate

sys.path.insert(0, os.path.dirname(__file__))
from rag_chain import answer_with_context
from eval_langsmith import ensure_dataset, DATASET_NAME, GOLDEN_PATH

import json


def rag_target(inputs: dict) -> dict:
    question = inputs["question"]
    answer, contexts = answer_with_context(question)
    return {"answer": answer, "contexts": contexts}


def main():
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    client = Client()
    ensure_dataset(client, golden)

    print("\nRunning golden dataset — LangSmith online evaluator will score answer relevancy...\n")
    results = ls_evaluate(
        rag_target,
        data=DATASET_NAME,
        experiment_prefix="golden-answer-relevancy",
        max_concurrency=1,
    )

    print("\n===== ANSWERS =====")
    for r in results:
        try:
            q = r.example.inputs.get("question", "?")
            ans = r.run.outputs.get("answer", "?")
            print(f"Q: {q}")
            print(f"A: {ans}\n")
        except Exception:
            pass

    print("Scores available in LangSmith:")
    print("  smith.langchain.com → Datasets & Experiments → zephyr-golden-qa")


if __name__ == "__main__":
    main()
