"""
NORTHSTAR — TARGETED FIX VALIDATION (mini eval)
================================================
Runs only NSB-023, NSB-024, and NSB-033 against the current pipeline
configuration and prints actual vs expected answers for quick visual
verification. Does NOT upload anything to LangSmith.

Use this before running the full 44-question eval to confirm that
targeted fixes (prompt v2 exception clause, role query expansion) work.

Run:
    python3 src/mini_eval_northstar.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()

from config import (
    NORTHSTAR_GOLDEN_PATH, NORTHSTAR_DB_PATH,
    EMBED_MODEL, CHAT_MODEL, TEMPERATURE, RETRIEVAL_K,
    NORTHSTAR_PROMPT_RAG_GROUNDING, _load_prompt,
)
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

TARGET_IDS = {"NSB-023", "NSB-024", "NSB-033", "NSB-034"}

# Multi-query retrieval for role disambiguation (mirrors eval_langsmith_northstar.py)
_ROLE_TERMS = {"viewer", "user", "administrator", "admin"}
_ROLE_COMPARE_TERMS = {"difference", "compare", "versus", "vs", "differ"}


def _get_contexts(question: str, retriever, k: int) -> list:
    q_lower = question.lower()
    mentioned_roles = [r for r in _ROLE_TERMS if r in q_lower]
    is_role_comparison = len(mentioned_roles) >= 2 and any(t in q_lower for t in _ROLE_COMPARE_TERMS)
    if is_role_comparison:
        seen, combined = set(), []
        for role in mentioned_roles:
            for doc in retriever.invoke(f"{role} role definition access"):
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    combined.append(doc)
        return combined if combined else retriever.invoke(question)
    return retriever.invoke(question)


# Heuristic pass checks for each targeted question
def _heuristic_pass(qid: str, answer: str) -> bool:
    a = answer.lower()
    return {
        "NSB-023": "dual approval" in a and ("always" in a or "regardless" in a or "every" in a or "no" in a),
        "NSB-024": "10,000" in answer or "10000" in answer,
        "NSB-033": "read-only" in a and "viewer" in a and "user" in a,
        "NSB-034": "five minutes" in a or "5 minutes" in a,
    }.get(qid, False)


def main():
    # Load golden dataset and filter to target questions
    with open(NORTHSTAR_GOLDEN_PATH) as f:
        all_questions = [json.loads(line) for line in f if line.strip()]
    targets = [q for q in all_questions if q["id"] in TARGET_IDS]

    # Setup pipeline
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(persist_directory=NORTHSTAR_DB_PATH, embedding_function=embeddings)
    retriever   = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    llm         = ChatOllama(model=CHAT_MODEL, temperature=TEMPERATURE)
    prompt_tmpl = ChatPromptTemplate.from_template(_load_prompt(NORTHSTAR_PROMPT_RAG_GROUNDING))
    chain       = prompt_tmpl | llm | StrOutputParser()

    print(f"\n{'='*70}")
    print(f"MINI EVAL — Targeted fix validation")
    print(f"  prompt : {NORTHSTAR_PROMPT_RAG_GROUNDING}")
    print(f"  model  : {CHAT_MODEL}  k={RETRIEVAL_K}")
    print(f"{'='*70}\n")

    all_pass = True
    for q in sorted(targets, key=lambda x: x["id"]):
        qid      = q["id"]
        question = q["question"]
        expected = q["expected_answer"]

        contexts    = _get_contexts(question, retriever, k=RETRIEVAL_K)
        context_str = "\n\n---\n\n".join(c.page_content for c in contexts)
        answer      = chain.invoke({"context": context_str, "question": question})

        passed   = _heuristic_pass(qid, answer)
        all_pass = all_pass and passed
        status   = "PASS" if passed else "FAIL  <-- STILL BROKEN"

        print(f"[{qid}] [{q['category']}/{q['difficulty']}] {question}")
        print(f"  Expected : {expected}")
        print(f"  Actual   : {answer.strip()}")
        q_lower = question.lower()
        mentioned = [r for r in _ROLE_TERMS if r in q_lower]
        if len(mentioned) >= 2 and any(t in q_lower for t in _ROLE_COMPARE_TERMS):
            print(f"  (multi-query retrieval: separate query per role)")
        print(f"\n  Retrieved contexts ({len(contexts)}):")
        for i, c in enumerate(contexts, 1):
            snippet = c.page_content[:200].replace("\n", " ")
            print(f"    [{i}] {snippet!r}")
        print(f"\n  Status: {status}")
        print(f"  {'-'*60}\n")

    print(f"{'='*70}")
    print(f"  Overall: {'ALL PASS — safe to run full eval' if all_pass else 'FAILURES REMAIN — review before full eval'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
