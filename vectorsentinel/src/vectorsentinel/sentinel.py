"""Main Sentinel class — the public-facing API."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vectorsentinel.core.gate import GateResult, gate_query
from vectorsentinel.core.index import DensityIndex


class Sentinel:
    """Confidence gate for vector search pipelines.

    Sits between your vector store and your LLM. Returns a structured verdict
    telling you whether the system should answer or abstain, with a confidence
    score and reasoning.

    Quick start::

        from vectorsentinel import Sentinel
        import numpy as np

        sentinel = Sentinel(dim=768)
        sentinel.add(corpus_embeddings, ids=doc_ids, labels=doc_labels)

        result = sentinel.gate(query_embedding)
        if result.should_answer:
            # call your LLM
        else:
            return "I don't have reliable information about this topic."
    """

    def __init__(
        self,
        dim: int,
        threshold: float = 0.5,
        k: int = 5,
    ):
        """
        Parameters
        ----------
        dim : embedding dimension
        threshold : confidence below this → abstain (0.0–1.0)
        k : number of nearest neighbors used for scoring
        """
        self.dim = dim
        self.threshold = threshold
        self.k = k
        self._index = DensityIndex(dim=dim)

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def add(
        self,
        embeddings: NDArray[np.float32],
        ids: list[str] | None = None,
        labels: list[str] | None = None,
    ) -> "Sentinel":
        """Add vectors to the sentinel index.

        Parameters
        ----------
        embeddings : array of shape (n, dim) or (dim,)
        ids : optional string IDs per vector
        labels : optional label per vector (improves agreement scoring)

        Returns self for chaining.
        """
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings, dtype=np.float32)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        self._index.add(embeddings, ids=ids, labels=labels)
        return self

    def delete(self, ids: list[str]) -> int:
        """Remove vectors by ID. Returns number removed."""
        return self._index.delete(ids)

    def refine(self) -> int:
        """Re-evaluate cluster assignments. Returns reassignment count."""
        return self._index.refine()

    # ------------------------------------------------------------------
    # Core gate
    # ------------------------------------------------------------------

    def gate(
        self,
        query: NDArray[np.float32] | list[float],
        threshold: float | None = None,
        k: int | None = None,
    ) -> GateResult:
        """Score a query and decide: answer or abstain.

        Parameters
        ----------
        query : embedding vector of shape (dim,)
        threshold : override instance threshold for this call
        k : override instance k for this call

        Returns
        -------
        GateResult with .should_answer, .confidence, .reason, .neighbors, etc.
        """
        q = np.asarray(query, dtype=np.float32)
        if q.ndim != 1:
            raise ValueError(f"query must be 1-D, got shape {q.shape}")

        return gate_query(
            query=q,
            index=self._index,
            threshold=threshold if threshold is not None else self.threshold,
            k=k if k is not None else self.k,
        )

    def gate_batch(
        self,
        queries: NDArray[np.float32],
        threshold: float | None = None,
        k: int | None = None,
    ) -> list[GateResult]:
        """Gate multiple queries at once."""
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        return [self.gate(queries[i], threshold=threshold, k=k) for i in range(len(queries))]

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def auto_calibrate(
        self,
        holdout_queries: NDArray[np.float32],
        in_domain_mask: list[bool],
        target_abstain_rate: float = 0.05,
        search_steps: int = 20,
    ) -> float:
        """Find a threshold that achieves approximately the desired abstain rate on OOD queries.

        Parameters
        ----------
        holdout_queries : array of shape (n, dim) — mix of in-domain and OOD
        in_domain_mask : True = in-domain, False = out-of-domain per query
        target_abstain_rate : desired fraction of OOD queries to abstain on
        search_steps : number of threshold candidates to try

        Returns
        -------
        Optimal threshold (also sets self.threshold).
        """
        queries = np.asarray(holdout_queries, dtype=np.float32)
        ood_queries = queries[~np.array(in_domain_mask)]

        if len(ood_queries) == 0:
            return self.threshold

        best_threshold = self.threshold
        best_diff = float("inf")

        for step in range(search_steps + 1):
            t = step / search_steps
            results = self.gate_batch(ood_queries, threshold=t)
            abstain_rate = sum(1 for r in results if not r.should_answer) / len(results)
            diff = abs(abstain_rate - target_abstain_rate)
            if diff < best_diff:
                best_diff = diff
                best_threshold = t

        self.threshold = best_threshold
        return best_threshold

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def cluster_report(self) -> list[dict]:
        """Return cluster health summary."""
        return self._index.cluster_report()

    def benchmark(
        self, queries: NDArray[np.float32], threshold: float | None = None
    ) -> dict[str, Any]:
        """Time the gate on a batch of queries and return latency stats."""
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        latencies = []
        for i in range(len(queries)):
            t0 = time.perf_counter()
            self.gate(queries[i], threshold=threshold)
            latencies.append((time.perf_counter() - t0) * 1000)

        latencies_arr = np.array(latencies)
        return {
            "n_queries": len(queries),
            "mean_ms": round(float(latencies_arr.mean()), 3),
            "p50_ms": round(float(np.percentile(latencies_arr, 50)), 3),
            "p95_ms": round(float(np.percentile(latencies_arr, 95)), 3),
            "p99_ms": round(float(np.percentile(latencies_arr, 99)), 3),
            "min_ms": round(float(latencies_arr.min()), 3),
            "max_ms": round(float(latencies_arr.max()), 3),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save index to disk."""
        self._index.save(path)

    @classmethod
    def load(cls, path: str, threshold: float = 0.5, k: int = 5) -> "Sentinel":
        """Load index from disk."""
        index = DensityIndex.load(path)
        sentinel = cls(dim=index.dim, threshold=threshold, k=k)
        sentinel._index = index
        return sentinel

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_embeddings(
        cls,
        embeddings: NDArray[np.float32],
        ids: list[str] | None = None,
        labels: list[str] | None = None,
        threshold: float = 0.5,
        k: int = 5,
    ) -> "Sentinel":
        """Build a Sentinel directly from a numpy embedding matrix."""
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        dim = embeddings.shape[1]
        sentinel = cls(dim=dim, threshold=threshold, k=k)
        sentinel.add(embeddings, ids=ids, labels=labels)
        return sentinel

    def __repr__(self) -> str:
        return (
            f"Sentinel(dim={self.dim}, size={self._index.size}, "
            f"threshold={self.threshold}, k={self.k})"
        )

    def __len__(self) -> int:
        return self._index.size
