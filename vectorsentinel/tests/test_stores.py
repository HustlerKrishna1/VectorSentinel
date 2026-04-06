"""Tests for store adapters."""

import numpy as np
import pytest

from vectorsentinel.stores.numpy_store import NumpyStore


def make_embeddings(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    emb = rng.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms


class TestNumpyStore:
    def test_build_sentinel(self):
        emb = make_embeddings(30, 32)
        store = NumpyStore(emb)
        sentinel = store.build_sentinel(threshold=0.5)
        assert len(sentinel) == 30
        assert sentinel.dim == 32

    def test_with_ids_and_labels(self):
        emb = make_embeddings(10, 16)
        ids = [f"doc_{i}" for i in range(10)]
        labels = [f"class_{i % 2}" for i in range(10)]
        store = NumpyStore(emb, ids=ids, labels=labels)
        sentinel = store.build_sentinel()
        assert len(sentinel) == 10

    def test_get_all_embeddings(self):
        emb = make_embeddings(5, 8)
        store = NumpyStore(emb)
        out_emb, out_ids, out_labels = store.get_all_embeddings()
        assert out_emb.shape == (5, 8)
        assert len(out_ids) == 5
        assert len(out_labels) == 5

    def test_1d_input_reshaped(self):
        emb = make_embeddings(1, 16)[0]  # 1-D
        store = NumpyStore(emb)
        out_emb, _, _ = store.get_all_embeddings()
        assert out_emb.shape == (1, 16)

    def test_gate_after_build(self):
        emb = make_embeddings(50, 64)
        store = NumpyStore(emb, labels=[f"c{i%3}" for i in range(50)])
        sentinel = store.build_sentinel(threshold=0.0)
        result = sentinel.gate(emb[0])
        assert result.should_answer  # threshold=0, always answers
