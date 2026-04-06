"""LlamaIndex integration for VectorSentinel.

Requires: pip install vectorsentinel[llamaindex]

Wraps a LlamaIndex query engine with a confidence gate. When the query is
out-of-distribution, returns a safe "I don't know" response instead of
hallucinating.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vectorsentinel.sentinel import Sentinel


def _require_llamaindex():
    try:
        import llama_index  # noqa: F401
    except ImportError:
        try:
            import llama_index.core  # noqa: F401
        except ImportError:
            raise ImportError(
                "llama-index-core is required for LlamaIndex integration. "
                "Install with: pip install vectorsentinel[llamaindex]"
            )


class SentinelQueryEngine:
    """LlamaIndex query engine wrapper with confidence gating.

    Example::

        from llama_index.core import VectorStoreIndex
        from vectorsentinel import Sentinel
        from vectorsentinel.integrations.llamaindex import SentinelQueryEngine

        index = VectorStoreIndex.from_documents(documents)
        base_engine = index.as_query_engine()

        sentinel = Sentinel.load("my_sentinel.npz")
        engine = SentinelQueryEngine(
            base_engine,
            sentinel,
            embed_fn=lambda q: embed_model.get_text_embedding(q),
        )

        response = engine.query("What is the capital of Mars?")
        print(response.response)
        # "I don't have reliable information about this topic."
    """

    def __init__(
        self,
        query_engine: Any,
        sentinel: Sentinel,
        embed_fn: Any,
        threshold: float | None = None,
        fallback_response: str = "I don't have reliable information about this topic.",
    ):
        _require_llamaindex()
        self._engine = query_engine
        self._sentinel = sentinel
        self._embed_fn = embed_fn
        self._threshold = threshold
        self._fallback = fallback_response

    def query(self, query_str: str) -> Any:
        """Gate query, then delegate to underlying engine if confident."""
        embedding = np.array(self._embed_fn(query_str), dtype=np.float32)
        gate_result = self._sentinel.gate(embedding, threshold=self._threshold)

        if not gate_result.should_answer:
            return _FallbackResponse(
                response=self._fallback,
                gate_result=gate_result,
            )

        response = self._engine.query(query_str)
        response.metadata = getattr(response, "metadata", {}) or {}
        response.metadata["gate_result"] = gate_result.to_dict()
        return response

    async def aquery(self, query_str: str) -> Any:
        return self.query(query_str)


class _FallbackResponse:
    """Mimics a LlamaIndex Response object for the abstain case."""

    def __init__(self, response: str, gate_result: Any):
        self.response = response
        self.source_nodes = []
        self.metadata = {"gate_result": gate_result.to_dict(), "abstained": True}

    def __str__(self) -> str:
        return self.response
