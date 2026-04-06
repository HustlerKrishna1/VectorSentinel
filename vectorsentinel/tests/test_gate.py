"""Tests for the confidence gate."""

import numpy as np
import pytest

from vectorsentinel.core.gate import GateVerdict, gate_query
from vectorsentinel.core.index import DensityIndex
from vectorsentinel.sentinel import Sentinel


def make_embeddings(n: int, dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    emb = rng.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms


def make_ood_embedding(dim: int, corpus: np.ndarray, seed: int = 999) -> np.ndarray:
    """Generate an embedding that is far from the corpus."""
    rng = np.random.RandomState(seed)
    # Try random vectors until we find one with low max similarity to corpus
    for _ in range(100):
        v = rng.randn(dim).astype(np.float32)
        v /= np.linalg.norm(v)
        max_sim = float((corpus @ v).max())
        if max_sim < 0.3:
            return v
    # Fallback: negate the corpus mean
    mean = corpus.mean(axis=0)
    v = -mean
    v /= np.linalg.norm(v)
    return v


# ---------------------------------------------------------------------------
# gate_query
# ---------------------------------------------------------------------------

class TestGateQuery:
    def setup_method(self):
        self.dim = 32
        self.corpus = make_embeddings(50, self.dim, seed=1)
        self.index = DensityIndex(dim=self.dim)
        labels = [f"class_{i % 5}" for i in range(50)]
        self.index.add(self.corpus, labels=labels)

    def test_in_domain_gets_high_confidence(self):
        # Query == a corpus vector → should be very confident
        query = self.corpus[0]
        result = gate_query(query, self.index, threshold=0.0)
        assert result.confidence > 0.0
        assert result.verdict == GateVerdict.ANSWER

    def test_ood_gets_low_confidence(self):
        ood = make_ood_embedding(self.dim, self.corpus)
        result = gate_query(ood, self.index, threshold=1.0)  # impossible threshold
        assert result.verdict == GateVerdict.ABSTAIN

    def test_empty_index_abstains(self):
        empty_index = DensityIndex(dim=self.dim)
        query = self.corpus[0]
        result = gate_query(query, empty_index, threshold=0.5)
        assert result.verdict == GateVerdict.ABSTAIN
        assert result.reason == "empty_index"

    def test_result_has_neighbors(self):
        query = self.corpus[10]
        result = gate_query(query, self.index, threshold=0.0, k=5)
        assert len(result.nearest_neighbors) == 5

    def test_confidence_in_range(self):
        for i in range(10):
            q = self.corpus[i]
            result = gate_query(q, self.index, threshold=0.0)
            assert 0.0 <= result.confidence <= 1.0

    def test_deterministic(self):
        query = self.corpus[3]
        r1 = gate_query(query, self.index, threshold=0.5)
        r2 = gate_query(query, self.index, threshold=0.5)
        assert r1.confidence == r2.confidence
        assert r1.verdict == r2.verdict

    def test_to_dict_shape(self):
        query = self.corpus[0]
        result = gate_query(query, self.index, threshold=0.5)
        d = result.to_dict()
        assert "should_answer" in d
        assert "confidence" in d
        assert "reason" in d
        assert "neighbors" in d
        assert isinstance(d["should_answer"], bool)


# ---------------------------------------------------------------------------
# Sentinel (high-level)
# ---------------------------------------------------------------------------

class TestSentinel:
    def setup_method(self):
        self.dim = 64
        rng = np.random.RandomState(42)
        self.corpus = rng.randn(100, self.dim).astype(np.float32)
        norms = np.linalg.norm(self.corpus, axis=1, keepdims=True)
        self.corpus /= norms
        self.labels = [f"cat_{i % 4}" for i in range(100)]

    def test_from_embeddings(self):
        sentinel = Sentinel.from_embeddings(self.corpus, labels=self.labels)
        assert len(sentinel) == 100
        assert sentinel.dim == self.dim

    def test_gate_returns_result(self):
        sentinel = Sentinel.from_embeddings(self.corpus)
        result = sentinel.gate(self.corpus[0])
        assert result is not None
        assert hasattr(result, "should_answer")
        assert hasattr(result, "confidence")

    def test_gate_batch(self):
        sentinel = Sentinel.from_embeddings(self.corpus)
        results = sentinel.gate_batch(self.corpus[:10])
        assert len(results) == 10

    def test_save_load(self, tmp_path):
        sentinel = Sentinel.from_embeddings(self.corpus, labels=self.labels, threshold=0.7)
        path = str(tmp_path / "sentinel")
        sentinel.save(path)
        loaded = Sentinel.load(path + ".npz", threshold=0.7)
        assert len(loaded) == len(sentinel)
        assert loaded.dim == sentinel.dim

    def test_benchmark_returns_stats(self):
        sentinel = Sentinel.from_embeddings(self.corpus)
        stats = sentinel.benchmark(self.corpus[:20])
        assert "mean_ms" in stats
        assert "p95_ms" in stats
        assert stats["n_queries"] == 20

    def test_cluster_report(self):
        sentinel = Sentinel.from_embeddings(self.corpus, labels=self.labels)
        report = sentinel.cluster_report()
        assert isinstance(report, list)

    def test_repr(self):
        sentinel = Sentinel(dim=128, threshold=0.6, k=3)
        r = repr(sentinel)
        assert "128" in r
        assert "0.6" in r

    def test_add_incremental(self):
        sentinel = Sentinel(dim=self.dim)
        sentinel.add(self.corpus[:50])
        assert len(sentinel) == 50
        sentinel.add(self.corpus[50:])
        assert len(sentinel) == 100

    def test_delete(self):
        ids = [f"v{i}" for i in range(100)]
        sentinel = Sentinel.from_embeddings(self.corpus, ids=ids)
        removed = sentinel.delete(["v0", "v1", "v2"])
        assert removed == 3
        assert len(sentinel) == 97
