"""
STEP 7 (EVALUATION) — PART 1: BUILD YOUR OWN EVAL HARNESS
========================================================
Before importing an eval library, build one. If you understand what these
metrics compute, you will never be fooled by a green dashboard.

This harness runs the golden dataset through the RAG chain and measures four
things, each teaching a different testing concept:

  1. RETRIEVAL HIT RATE   — did the retriever even fetch the right chunk?
                            (If retrieval misses, generation cannot recover.)
  2. KEYWORD CORRECTNESS   — reference-based check: does the answer contain the
                            expected fact? Cheap, deterministic, brittle.
  3. ABSTENTION            — for out-of-corpus questions, did the system
                            correctly say "I don't know" instead of inventing?
                            THIS is your hallucination test.
  4. LLM-AS-JUDGE          — reference-free check: a second model grades
                            faithfulness. Powerful but has pitfalls (below).

BIG TESTING LESSON: outputs vary between runs. A single pass tells you almost
nothing. Real AI testing measures PASS RATES over repeated runs, not one result.
Set REPEATS below to 3+ and watch the numbers wobble. That wobble is the whole
reason AI testing is a different discipline from traditional QA.

Run:  python src/eval_custom.py
"""

import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from rag_chain import answer_with_context, JUDGE_MODEL, LOCAL_JUDGE_MODEL

REPEATS = 1   # bump to 3 to see run-to-run variance (recommended once it works)


def load_golden():
    with open("eval/golden_qa.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# METRIC 1: retrieval hit rate
# For in-corpus questions, check that at least one required keyword appears in
# the retrieved context. This isolates retrieval from generation — a fault-
# localisation habit straight from traditional QA.
# ---------------------------------------------------------------------------
def retrieval_hit(item, contexts) -> bool:
    joined = " ".join(contexts).lower()
    return any(kw.lower() in joined for kw in item["must_contain"])


# ---------------------------------------------------------------------------
# METRIC 2: keyword correctness (reference-based)
# What you learn: exact/keyword matching is deterministic and fast but punishes
# correct answers that use different words. This weakness is exactly WHY the
# field moved to LLM-as-judge (metric 4).
# ---------------------------------------------------------------------------
def keyword_correct(item, answer) -> bool:
    ans = answer.lower()
    return any(kw.lower() in ans for kw in item["must_contain"])


# ---------------------------------------------------------------------------
# METRIC 3: abstention (hallucination test)
# For questions NOT in the corpus, "correct" means refusing to answer.
# ---------------------------------------------------------------------------
def abstained(answer) -> bool:
    markers = ["don't know", "do not know", "not in", "no information",
               "cannot", "can't", "not contain", "not available"]
    return any(m in answer.lower() for m in markers)


# ---------------------------------------------------------------------------
# METRIC 4: LLM-as-judge (reference-free faithfulness)
# A second LLM decides whether the answer is supported by the context.
# PITFALLS you must know as a tester:
#   - the judge can be wrong, confidently
#   - judges drift with model/prompt changes
#   - a judge can favour verbose answers
# Mitigation: temperature 0, a tight rubric, and forcing a one-word verdict.
# Never treat the judge as ground truth — validate it against your golden set.
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """You are grading whether an ANSWER is fully supported by the CONTEXT.
Reply with exactly one word: PASS if every claim in the answer is supported by
the context, or FAIL if any claim is not supported.

CONTEXT:
{context}

ANSWER:
{answer}

Verdict (PASS or FAIL):"""

# USE_LOCAL_JUDGE=1 in .env uses Ollama (free, no quota) instead of Gemini.
if os.getenv("USE_LOCAL_JUDGE") == "1":
    judge = ChatOllama(model=LOCAL_JUDGE_MODEL, temperature=0)
    print(f"[judge] Using local Ollama model: {LOCAL_JUDGE_MODEL}")
else:
    judge = ChatGoogleGenerativeAI(model=JUDGE_MODEL, temperature=0)
    print(f"[judge] Using Gemini: {JUDGE_MODEL}")


def llm_judge(answer, contexts) -> bool:
    content = judge.invoke(
        JUDGE_PROMPT.format(context="\n\n".join(contexts), answer=answer)
    ).content
    # Gemini SDK may return content as a list of parts; extract text field
    if isinstance(content, list):
        content = " ".join(p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in content)
    return content.strip().upper().startswith("PASS")


# ---------------------------------------------------------------------------
# Run the suite
# ---------------------------------------------------------------------------
def main():
    golden = load_golden()
    totals = {"retrieval": 0, "keyword": 0, "abstention": 0, "judge": 0,
              "n_in": 0, "n_out": 0}

    for run in range(REPEATS):
        print(f"\n===== RUN {run + 1} of {REPEATS} =====")
        for item in golden:
            answer, contexts = answer_with_context(item["question"])

            if item["in_corpus"]:
                totals["n_in"] += 1
                r = retrieval_hit(item, contexts)
                k = keyword_correct(item, answer)
                j = llm_judge(answer, contexts)
                totals["retrieval"] += r
                totals["keyword"] += k
                totals["judge"] += j
                print(f"[in ] retrieval={'Y' if r else 'N'} "
                      f"keyword={'Y' if k else 'N'} judge={'Y' if j else 'N'} "
                      f":: {item['question']}")
            else:
                totals["n_out"] += 1
                a = abstained(answer)
                totals["abstention"] += a
                flag = "OK" if a else "HALLUCINATED"
                print(f"[out] abstention={'Y' if a else 'N'} ({flag}) "
                      f":: {item['question']}")
                if not a:
                    print(f"       -> model said: {answer[:120]}")

    # ---- summary: report RATES, not single verdicts ----
    print("\n===== SUMMARY (pass rates across all runs) =====")
    ni, no = totals["n_in"], totals["n_out"]
    print(f"Retrieval hit rate : {totals['retrieval']}/{ni} = {totals['retrieval']/ni:.0%}")
    print(f"Keyword correctness: {totals['keyword']}/{ni} = {totals['keyword']/ni:.0%}")
    print(f"LLM-judge faithful : {totals['judge']}/{ni} = {totals['judge']/ni:.0%}")
    print(f"Abstention (no hallucination): {totals['abstention']}/{no} = {totals['abstention']/no:.0%}")
    print("\nTip: run again with REPEATS=3. If these rates move, you have just "
          "seen why single-run testing does not work for AI systems.")


if __name__ == "__main__":
    main()
