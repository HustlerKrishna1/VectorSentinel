"""LangChain integration for VectorSentinel.

Requires: pip install vectorsentinel[langchain]

Provides a guarded retriever that intercepts retrieval results and applies the
confidence gate before returning documents. When the gate abstains, retrieval
returns an empty list — preventing the LLM from being called with bad context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from vectorsentinel.sentinel import Sentinel

if TYPE_CHECKING:
    pass


def _require_langchain():
    try:
        import langchain_core  # noqa: F401
    except ImportError:
        raise ImportError(
            "langchain-core is required for LangChain integration. "
            "Install with: pip install vectorsentinel[langchain]"
        )


class SentinelRetriever:
    """LangChain-compatible retriever that adds confidence gating.

    Wraps any LangChain BaseRetriever. When the sentinel gate determines
    the query is out-of-distribution, it returns an empty list so the LLM
    receives no (potentially misleading) context.

    Example::

        from langchain_community.vectorstores import Chroma
        from langchain_openai import OpenAIEmbeddings
        from vectorsentinel import Sentinel
        from vectorsentinel.integrations.langchain import SentinelRetriever

        vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
        base_retriever = vectorstore.as_retriever()

        sentinel = Sentinel.load("my_sentinel.npz")
        retriever = SentinelRetriever(base_retriever, sentinel, embed_fn=OpenAIEmbeddings().embed_query)

        docs = retriever.get_relevant_documents("What is quantum computing?")
        if not docs:
            print("I don't have reliable information about this topic.")
    """

    def __init__(
        self,
        retriever: Any,
        sentinel: Sentinel,
        embed_fn: Any,
        threshold: float | None = None,
        on_abstain: str = "empty",
    ):
        """
        Parameters
        ----------
        retriever : any LangChain retriever with .get_relevant_documents()
        sentinel : built Sentinel instance
        embed_fn : callable(str) -> list[float], embeds a query string
        threshold : override sentinel threshold for this retriever
        on_abstain : "empty" (return []) or "raise" (raise AbstainError)
        """
        _require_langchain()
        self._retriever = retriever
        self._sentinel = sentinel
        self._embed_fn = embed_fn
        self._threshold = threshold
        self._on_abstain = on_abstain

    def get_relevant_documents(self, query: str) -> list:
        """Gate the query, then retrieve docs if confident."""
        embedding = np.array(self._embed_fn(query), dtype=np.float32)
        result = self._sentinel.gate(embedding, threshold=self._threshold)

        if not result.should_answer:
            if self._on_abstain == "raise":
                raise AbstainError(result)
            return []

        return self._retriever.get_relevant_documents(query)

    # Async support
    async def aget_relevant_documents(self, query: str) -> list:
        return self.get_relevant_documents(query)

    @property
    def last_gate_result(self):
        return getattr(self, "_last_result", None)


class AbstainError(Exception):
    """Raised when the sentinel gate decides to abstain."""

    def __init__(self, gate_result):
        self.gate_result = gate_result
        super().__init__(
            f"Sentinel abstained: confidence={gate_result.confidence:.3f}, "
            f"reason={gate_result.reason}"
        )


class SentinelRunnable:
    """LangChain Runnable that gates queries in a chain.

    Can be inserted into any LangChain LCEL chain. Passes through the input
    dict if confident; raises AbstainError or returns a fallback dict otherwise.

    Example::

        from langchain_core.runnables import RunnablePassthrough
        from vectorsentinel.integrations.langchain import SentinelRunnable

        guarded_chain = (
            SentinelRunnable(sentinel, embed_fn, input_key="question")
            | retriever
            | prompt
            | llm
        )
    """

    def __init__(
        self,
        sentinel: Sentinel,
        embed_fn: Any,
        input_key: str = "question",
        threshold: float | None = None,
        fallback_response: str = "I don't have reliable information about this topic.",
    ):
        _require_langchain()
        self._sentinel = sentinel
        self._embed_fn = embed_fn
        self._input_key = input_key
        self._threshold = threshold
        self._fallback = fallback_response

    def invoke(self, input_dict: dict, config=None) -> dict:
        query = input_dict[self._input_key]
        embedding = np.array(self._embed_fn(query), dtype=np.float32)
        result = self._sentinel.gate(embedding, threshold=self._threshold)

        if not result.should_answer:
            raise AbstainError(result)

        return {**input_dict, "_gate_result": result.to_dict()}

    def __or__(self, other):
        """Support | chaining syntax."""
        from langchain_core.runnables import RunnableLambda, RunnableSequence

        return RunnableSequence(RunnableLambda(self.invoke), other)
