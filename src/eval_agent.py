"""
AGENT EVALUATION
================
Evaluates the LangGraph ReAct agent on a golden dataset of 12 questions
across four categories:

  retrieval_only  — should call search_handbook once, not calculator
  calculator_only — should call calculator only, not search_handbook
  multi_step      — should call search_handbook then calculator (order matters)
  abstention      — out-of-corpus question; agent must refuse to invent an answer

Four metrics, each measuring a different failure mode:

  Tool hit rate     — did the agent call every expected tool?
                      < 100% means the agent bypassed a tool (e.g. answered
                      a math question from training knowledge instead of
                      calling calculator)

  No phantom calls  — did the agent avoid calling unexpected tools?
                      Phantom calls waste latency and signal confused routing

  Sequence accuracy — for multi-step questions only: did the agent retrieve
                      before calculating? Wrong order = wrong intermediate
                      value fed to the next tool

  Answer accuracy   — does the final answer contain the expected fact or value?
                      A correct answer via wrong tool path is still a routing
                      failure — the trace is the only way to tell

  Abstention rate   — for out-of-corpus questions: did the agent refuse to
                      invent an answer?

Run:
    python3 src/eval_agent.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import ToolMessage
from agent import agent
from judgeUtil import abstained

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "..", "eval", "golden_qa_agent.json")

CATEGORIES = ["retrieval_only", "calculator_only", "multi_step", "abstention"]
CATEGORY_LABELS = {
    "retrieval_only":  "RETRIEVAL ONLY — must call search_handbook, not calculator",
    "calculator_only": "CALCULATOR ONLY — must call calculator, not search_handbook",
    "multi_step":      "MULTI-STEP — must call search_handbook then calculator (order matters)",
    "abstention":      "ABSTENTION — out-of-corpus question; must refuse, not invent",
}


def run_agent_with_trace(question: str) -> tuple[str, list[str]]:
    """
    Invoke the agent and return (final_answer, tools_called_in_order).
    Tool names are extracted from ToolMessage objects in the LangGraph
    message list — this is the only reliable way to know which tools fired
    vs. which tools the model merely considered.
    """
    result = agent.invoke({"messages": [("user", question)]})
    messages = result["messages"]
    tools_called = [msg.name for msg in messages if isinstance(msg, ToolMessage)]
    final_answer = messages[-1].content
    return final_answer, tools_called


def keyword_match(answer: str, must_contain: list[str]) -> bool:
    """
    True if at least one must_contain string appears in the answer.
    OR semantics — entries like ["1,000", "1000"] handle format variants.
    For entries that require all facts (e.g. ["Pro", "Enterprise"]) each
    string is listed because both must appear; the caller should pass them
    as separate items and the check returns True only if all appear.
    """
    a = answer.lower()
    return all(kw.lower() in a for kw in must_contain)


def evaluate_question(entry: dict) -> dict:
    question       = entry["question"]
    expected_tools = entry["expected_tools"]
    category       = entry["category"]

    try:
        actual_answer, actual_tools = run_agent_with_trace(question)
    except Exception as e:
        return {"id": entry["id"], "question": question, "category": category, "error": str(e)}

    expected_set = set(expected_tools)
    actual_set   = set(actual_tools)

    # --- Metric 1: tool hit rate (recall over expected tools) ---
    tool_hit_rate = (
        len(expected_set & actual_set) / len(expected_set)
        if expected_set else 1.0
    )

    # --- Metric 2: no phantom calls ---
    phantom_tools = sorted(actual_set - expected_set)
    no_phantom    = len(phantom_tools) == 0

    # --- Metric 3: sequence correct (multi_step only) ---
    # Deduplicate while preserving order (handles repeated tool calls)
    actual_ordered = list(dict.fromkeys(actual_tools))
    sequence_correct = (actual_ordered == expected_tools) if category == "multi_step" else None

    # --- Metric 4: answer correctness ---
    if category == "abstention":
        answer_correct = abstained(actual_answer)
    else:
        must_contain   = entry.get("must_contain", [])
        answer_correct = keyword_match(actual_answer, must_contain) if must_contain else None

    return {
        "id":               entry["id"],
        "question":         question,
        "category":         category,
        "expected_tools":   expected_tools,
        "actual_tools":     actual_tools,
        "actual_answer":    actual_answer,
        "tool_hit_rate":    tool_hit_rate,
        "no_phantom":       no_phantom,
        "phantom_tools":    phantom_tools,
        "sequence_correct": sequence_correct,
        "answer_correct":   answer_correct,
    }


def _icon(value) -> str:
    if value is True:  return "✓"
    if value is False: return "✗"
    return "—"


def main():
    with open(GOLDEN_PATH) as f:
        questions = json.load(f)

    print(f"\n{'='*68}")
    print(f"  AGENT EVALUATION")
    print(f"  {len(questions)} questions | model: llama3.1:8b")
    print(f"  tools: search_handbook (RAG retriever), calculator (arithmetic)")
    print(f"{'='*68}\n")

    results = []

    for category in CATEGORIES:
        cat_qs = [q for q in questions if q["category"] == category]
        if not cat_qs:
            continue

        print(f"--- {CATEGORY_LABELS[category]} ---\n")

        for entry in cat_qs:
            r = evaluate_question(entry)
            results.append(r)

            if "error" in r:
                print(f"  [{r['id']}] ERROR: {r['error']}\n")
                continue

            seq_part = (
                f"seq={_icon(r['sequence_correct'])}  "
                if r["sequence_correct"] is not None else ""
            )

            overall_pass = (
                r["tool_hit_rate"] == 1.0
                and r["no_phantom"]
                and r["answer_correct"] is not False
                and r["sequence_correct"] is not False
            )
            status = "PASS" if overall_pass else "FAIL"

            print(f"  [{status}] {r['id']} — {r['question']}")
            print(f"         expected tools : {r['expected_tools']}")
            print(f"         actual tools   : {r['actual_tools']}")
            print(f"         tool_hit={_icon(r['tool_hit_rate'] == 1.0)}  "
                  f"phantom={_icon(r['no_phantom'])}  "
                  f"{seq_part}"
                  f"answer={_icon(r['answer_correct'])}")
            if r["phantom_tools"]:
                print(f"         ⚠ phantom calls : {r['phantom_tools']}")
            print(f"         answer : {r['actual_answer'][:120].strip()}")
            print()

    # -----------------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------------
    valid      = [r for r in results if "error" not in r]
    in_corpus  = [r for r in valid if r["category"] != "abstention"]
    multi      = [r for r in valid if r["category"] == "multi_step"]
    abst       = [r for r in valid if r["category"] == "abstention"]

    tool_hits_total  = sum(r["tool_hit_rate"] for r in valid)
    no_phantom_count = sum(1 for r in valid if r["no_phantom"])
    seq_correct      = sum(1 for r in multi if r["sequence_correct"])
    ans_correct      = sum(1 for r in in_corpus if r["answer_correct"])
    abst_correct     = sum(1 for r in abst if r["answer_correct"])

    n = len(valid)

    print(f"\n{'='*68}")
    print(f"  SUMMARY  ({n} questions evaluated)")
    print(f"{'='*68}")
    print(f"  Tool hit rate     : {tool_hits_total:.0f}/{n} = {tool_hits_total/n*100:.0f}%"
          f"  (expected tools were called)")
    print(f"  No phantom calls  : {no_phantom_count}/{n} = {no_phantom_count/n*100:.0f}%"
          f"  (no unexpected tools called)")
    if multi:
        print(f"  Sequence accuracy : {seq_correct}/{len(multi)} = {seq_correct/len(multi)*100:.0f}%"
              f"  (correct tool order, multi-step only)")
    if in_corpus:
        print(f"  Answer accuracy   : {ans_correct}/{len(in_corpus)} = {ans_correct/len(in_corpus)*100:.0f}%"
              f"  (in-corpus questions, keyword match)")
    if abst:
        print(f"  Abstention rate   : {abst_correct}/{len(abst)} = {abst_correct/len(abst)*100:.0f}%"
              f"  (out-of-corpus refusal)")
    print(f"{'='*68}")

    print("""
  What each failure pattern means:
  ─────────────────────────────────────────────────────────────────
  Tool hit rate < 100%    Agent skipped a required tool — likely used
                          parametric (training) knowledge instead of
                          grounding the answer in the corpus or computing
                          it explicitly. Unreliable in production.

  Phantom calls > 0       Agent called a tool it did not need — wasted
                          latency, and the extra tool output may have
                          confused the final answer.

  Sequence error          Agent called tools in the wrong order — e.g.
  (multi-step only)       calculated before retrieving, so the expression
                          contained the wrong value.

  Answer wrong despite    Routing was correct but retrieval or generation
  correct tool routing    failed. Investigate the retrieved chunks and the
                          prompt grounding instruction.

  Abstention failure      Agent hallucinated an answer for an out-of-corpus
                          question. The grounding prompt needs investigation.
  ─────────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
