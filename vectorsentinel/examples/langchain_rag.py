"""LangChain RAG integration example.

Requires:
    pip install vectorsentinel[langchain]
    pip install langchain-openai  # or another LLM provider

This example shows how to add VectorSentinel to an existing LangChain pipeline
in 3 lines of code.

Note: This is a template — set OPENAI_API_KEY to run with real embeddings.
"""

from __future__ import annotations

import os

import numpy as np

from vectorsentinel import Sentinel
from vectorsentinel.integrations.langchain import AbstainError, SentinelRetriever

# ---------------------------------------------------------------------------
# Option A: Template with real OpenAI embeddings (requires API key)
# ---------------------------------------------------------------------------

def run_with_openai():
    """Full working example with LangChain + OpenAI."""
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain.chains import RetrievalQA
    except ImportError:
        print("Install: pip install langchain-community langchain-openai chromadb")
        return

    # 1. Build your existing RAG pipeline
    embedding_model = OpenAIEmbeddings()
    texts = [
        "Python is a high-level programming language.",
        "Lists in Python are mutable sequences.",
        "Dictionaries map keys to values in Python.",
        "Python supports object-oriented programming.",
        "The asyncio library enables async programming.",
    ]
    vectorstore = Chroma.from_texts(texts, embedding_model)
    base_retriever = vectorstore.as_retriever()

    # 2. Build a Sentinel from the same corpus (3 lines!)
    corpus_emb = np.array([embedding_model.embed_query(t) for t in texts], dtype=np.float32)
    sentinel = Sentinel.from_embeddings(corpus_emb, labels=["python"] * len(texts))

    # 3. Wrap the retriever
    guarded_retriever = SentinelRetriever(
        retriever=base_retriever,
        sentinel=sentinel,
        embed_fn=embedding_model.embed_query,
        threshold=0.45,
    )

    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=guarded_retriever)

    # In-domain query -> answers
    response = chain.invoke("What is a Python list?")
    print(f"In-domain: {response['result']}")

    # OOD query -> abstains (empty retrieval -> LLM says "I don't know")
    response = chain.invoke("What did ancient Martians eat for breakfast?")
    print(f"OOD: {response['result']}")


# ---------------------------------------------------------------------------
# Option B: Self-contained demo with synthetic embeddings (no API key)
# ---------------------------------------------------------------------------

def run_synthetic():
    """Demonstrates the SentinelRetriever contract with fake embeddings."""
    print("Running synthetic LangChain integration demo...")
    print("(Replace with real embeddings + LangChain chain for production use)")
    print()

    DIM = 128
    rng = np.random.RandomState(0)

    # Build a corpus of 100 vectors around 3 topic clusters
    corpus_parts, labels = [], []
    for c in range(3):
        center = rng.randn(DIM).astype(np.float32)
        center /= np.linalg.norm(center)
        noise = rng.randn(34, DIM).astype(np.float32) * (0.2 / DIM ** 0.5)
        vecs = center + noise
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        corpus_parts.append(vecs)
        labels.extend([f"topic_{c}"] * 34)
    corpus = np.vstack(corpus_parts)

    sentinel = Sentinel.from_embeddings(corpus, labels=labels, threshold=0.45, k=7)

    class FakeRetriever:
        """Simulates a LangChain retriever."""
        def get_relevant_documents(self, query: str):
            return [{"page_content": f"Document relevant to: {query}"}]

    def fake_embed(text: str) -> list[float]:
        """Simulate embedding by returning the first corpus vector (in-domain)."""
        return corpus[0].tolist()

    def fake_embed_ood(text: str) -> list[float]:
        """Simulate an OOD embedding."""
        v = rng.randn(DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        return v.tolist()

    retriever = SentinelRetriever(
        retriever=FakeRetriever(),
        sentinel=sentinel,
        embed_fn=fake_embed,
        threshold=0.45,
    )

    # In-domain
    docs = retriever.get_relevant_documents("Python list operations")
    print(f"In-domain query -> docs retrieved: {len(docs)} (should be > 0)")

    # OOD (use a different embed_fn)
    ood_retriever = SentinelRetriever(
        retriever=FakeRetriever(),
        sentinel=sentinel,
        embed_fn=fake_embed_ood,
        threshold=0.45,
    )
    docs = ood_retriever.get_relevant_documents("Ancient Martian civilizations")
    print(f"OOD query       -> docs retrieved: {len(docs)} (should be 0 — abstained)")

    print()
    print("In a real chain:")
    print("  guarded_retriever = SentinelRetriever(base_retriever, sentinel, embed_fn)")
    print("  chain = RetrievalQA.from_chain_type(llm=llm, retriever=guarded_retriever)")


if __name__ == "__main__":
    if os.getenv("OPENAI_API_KEY"):
        run_with_openai()
    else:
        run_synthetic()
