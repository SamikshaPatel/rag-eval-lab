"""
STEP 6 (AGENTS & TOOL USE) — a LangGraph agent
==============================================
A RAG chain always does the same thing: retrieve, then answer. An AGENT
*decides* what to do. Here it chooses between two tools:

  - search_handbook : the RAG retriever from step 5, wrapped as a tool
  - calculator      : evaluates arithmetic

Concepts you learn here:
  1. TOOLS            — functions the model is allowed to call
  2. AGENTIC ROUTING  — the model decides WHICH tool, and in what order
  3. MULTI-STEP       — some questions need retrieve THEN calculate
  4. NEW FAILURE MODES— agents fail in ways chains can't (wrong tool, wrong
                        arguments, infinite loops). This is why agents need
                        even more testing than plain RAG.

Why this matters for a tester: the moment the system makes decisions, your test
surface explodes. You are no longer testing one answer — you are testing a
*decision tree* whose branches change with the model's mood. That is the core
challenge of agent QA.

Try a multi-step question:
  python src/agent.py "I need 90 extra days of retention. What will the add-on cost per month?"
The agent should retrieve the add-on price (15 dollars per 30 days) and then
compute 90 / 30 * 15 = 45 using the calculator.
"""

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from config import CHAT_MODEL
from rag_chain import get_retriever


# ---------------------------------------------------------------------------
# 1. TOOLS — each is a plain Python function with a docstring. The docstring
# is NOT a comment; the model reads it to decide when to use the tool. A vague
# docstring is a real bug: the agent will pick the wrong tool. Writing good
# tool descriptions is a testable skill.
# ---------------------------------------------------------------------------
@tool
def search_handbook(query: str) -> str:
    """Look up facts about Zephyr Analytics: plans, pricing, features, limits,
    support, and data policies. Use this for any question about how Zephyr works."""
    retriever = get_retriever(k=3)
    docs = retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '90 / 30 * 15'.
    Use this whenever the user's question requires a numeric calculation."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: only basic arithmetic is allowed."
    try:
        return str(eval(expression))   # safe: input is whitelisted above
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# 2 + 3. AGENTIC ROUTING via a prebuilt ReAct agent.
# create_react_agent builds the classic loop: the model THINKS, optionally
# CALLS a tool, reads the result, and repeats until it can answer. LangGraph
# runs that loop as a state machine.
#
# Note: tool-calling reliability depends on the open model. llama3.1:8b is
# decent; smaller models often pick wrong tools — which is itself a great thing
# to observe and measure.
# ---------------------------------------------------------------------------
llm = ChatOllama(model=CHAT_MODEL, temperature=0)
agent = create_react_agent(llm, tools=[search_handbook, calculator])


def ask_agent(question: str) -> str:
    result = agent.invoke({"messages": [("user", question)]})
    return result["messages"][-1].content


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else \
        "I need 90 extra days of retention. What will the add-on cost per month?"
    print(f"\nQ: {q}\n")
    print(f"A: {ask_agent(q)}")
    # LangSmith tip: open the trace for this run to SEE which tools fired and in
    # what order. For agents, the trace is not a nicety — it is the only way to
    # tell a lucky right answer from a correct process.
