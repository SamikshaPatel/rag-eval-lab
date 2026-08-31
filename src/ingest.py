"""
STEP 5 (RAG) — PART 1: INGESTION
=================================
This script turns a document into something a computer can search by *meaning*.

Concepts you learn here, in the order they appear:
  1. LOADING      — getting raw text into the pipeline
  2. CHUNKING     — splitting text into retrievable pieces
  3. EMBEDDINGS   — turning each chunk into a vector (a list of numbers)
  4. VECTOR STORE — saving those vectors so we can search them later

Run this ONCE before anything else:  python src/ingest.py
It writes a folder called ./chroma_db that the other scripts read from.
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from rag_chain import DB_PATH, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

DATA_PATH = "data/zephyr_handbook.md"

# ---------------------------------------------------------------------------
# 1. LOADING
# What you learn: RAG starts with plain text. The loader is just plumbing.
# ---------------------------------------------------------------------------
docs = TextLoader(DATA_PATH).load()
print(f"Loaded {len(docs)} document(s), {len(docs[0].page_content)} characters.")

# ---------------------------------------------------------------------------
# 2. CHUNKING
# What you learn: models retrieve *chunks*, not whole files. Chunk size is a
# real tuning knob and a real source of bugs. Too big -> you retrieve
# irrelevant text and dilute the answer. Too small -> a fact gets split across
# two chunks and the retriever never sees it whole. This single parameter is
# one of the most common causes of "why did my RAG give a bad answer?"
#
# TESTING INSIGHT: later, come back and change chunk_size to 100, re-run the
# eval, and watch retrieval quality drop. That is a controlled experiment on a
# non-deterministic system — the core skill you are building.
# ---------------------------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks.")

# ---------------------------------------------------------------------------
# 3. EMBEDDINGS + 4. VECTOR STORE
# What you learn: an embedding maps text to a point in space where "similar
# meaning" = "close together". The vector store (Chroma, free + local) holds
# those points and answers the question "which chunks are nearest to THIS
# query?" That nearest-neighbour search IS retrieval.
#
# Note: the embedding model and the chat model are SEPARATE models. A common
# beginner bug is forgetting to `ollama pull nomic-embed-text`.
# ---------------------------------------------------------------------------
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_PATH,   # saved to disk automatically in langchain-chroma
)
print(f"Embedded and stored {len(chunks)} chunks in {DB_PATH}.")
print("Ingestion complete. You can now run src/rag_chain.py")
