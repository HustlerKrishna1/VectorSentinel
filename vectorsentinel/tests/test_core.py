"""Tests for core clustering and index functionality."""

import numpy as np
import pytest

from vectorsentinel.core.cluster import DensityClusterer
from vectorsentinel.core.index import DensityIndex


def make_embeddings(n: int, dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    emb = rng.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms


def make_clustered_embeddings(n_per_cluster: int, n_clusters: int, dim: int, seed: int = 0):
    """Make embeddings with clear cluster structure."""
    rng = np.random.RandomState(seed)
    all_emb = []
    all_labels = []
    for c in range(n_clusters):
        center = rng.randn(dim).astype(np.float32)
        center /= np.linalg.norm(center)
        noise = rng.randn(n_per_cluster, dim).astype(np.float32) * 0.1
        cluster_emb = center + noise
        norms = np.linalg.norm(cluster_emb, axis=1, keepdims=True)
        cluster_emb /= norms
        all_emb.append(cluster_emb)
        all_labels.extend([f"class_{c}"] * n_per_cluster)
    return np.vstack(all_emb), all_labels


# ---------------------------------------------------------------------------
# DensityClusterer
# ---------------------------------------------------------------------------

class TestDensityClusterer:
    def test_single_vector(self):
        emb = make_embeddings(1, 32)
        clusterer = DensityClusterer()
        clusters = clusterer.fit(emb)
        assert len(clusters) == 1
        assert clusters[0].size == 1

    def test_all_vectors_assigned(self):
        n = 50
        emb = make_embeddings(n, 64)
        clusterer = DensityClusterer(absorption_threshold=0.0)
        clusters = clusterer.fit(emb)
        total = sum(c.size for c in clusters)
        assert total == n

    def test_well_separated_clusters(self):
        emb, labels = make_clustered_embeddings(20, 3, dim=32)
        clusterer = DensityClusterer(absorption_threshold=0.7)
        clusters = clusterer.fit(emb, labels)
        # Should find at least 1 cluster
        assert len(clusters) >= 1
        # All vectors assigned
        total = sum(c.size for c in clusters)
        assert total == 60

    def test_deterministic(self):
        emb = make_embeddings(100, 64, seed=7)
        clusterer = DensityClusterer()
        clusters1 = clusterer.fit(emb)
        clusters2 = clusterer.fit(emb)
        assert len(clusters1) == len(clusters2)
        for c1, c2 in zip(clusters1, clusters2):
            assert c1.size == c2.size
            np.testing.assert_array_equal(c1.member_indices, c2.member_indices)

    def test_purity_with_labels(self):
        emb, labels = make_clustered_embeddings(10, 2, dim=32)
        clusterer = DensityClusterer(absorption_threshold=0.5)
        clusters = clusterer.fit(emb, labels)
        for c in clusters:
            assert 0.0 <= c.purity <= 1.0


# ---------------------------------------------------------------------------
# DensityIndex
# ---------------------------------------------------------------------------

class TestDensityIndex:
    def test_add_and_search(self):
        index = DensityIndex(dim=32)
        emb = make_embeddings(20, 32)
        index.add(emb)
        assert index.size == 20

        results = index.search(emb[0], k=3)
        assert len(results) == 3
        # Top result should be the query itself (highest similarity)
        assert results[0].similarity > 0.99

    def test_search_returns_sorted(self):
        index = DensityIndex(dim=32)
        emb = make_embeddings(50, 32)
        index.add(emb)
        query = emb[5]
        results = index.search(query, k=5)
        sims = [r.similarity for r in results]
        assert sims == sorted(sims, reverse=True)

    def test_delete(self):
        index = DensityIndex(dim=32)
        emb = make_embeddings(10, 32)
        ids = [f"vec_{i}" for i in range(10)]
        index.add(emb, ids=ids)
        removed = index.delete(["vec_0", "vec_1"])
        assert removed == 2
        assert index.size == 8

    def test_save_load(self, tmp_path):
        index = DensityIndex(dim=32)
        emb = make_embeddings(15, 32)
        ids = [f"id_{i}" for i in range(15)]
        labels = [f"label_{i % 3}" for i in range(15)]
        index.add(emb, ids=ids, labels=labels)

        save_path = str(tmp_path / "test_index")
        index.save(save_path)

        loaded = DensityIndex.load(save_path + ".npz")
        assert loaded.size == 15
        assert loaded._ids == ids
        assert loaded._labels == labels

    def test_mean_density_positive(self):
        index = DensityIndex(dim=64)
        emb = make_embeddings(30, 64)
        index.add(emb)
        assert index.mean_density > 0

    def test_cluster_report(self):
        index = DensityIndex(dim=32)
        emb, labels = make_clustered_embeddings(10, 3, dim=32)
        index.add(emb, labels=labels)
        report = index.cluster_report()
        assert isinstance(report, list)
        assert len(report) >= 1
        for entry in report:
            assert "cluster_id" in entry
            assert "size" in entry
            assert "purity" in entry

    def test_nearest_cluster(self):
        index = DensityIndex(dim=32)
        emb = make_embeddings(30, 32)
        index.add(emb)
        query = emb[0]
        cid = index.nearest_cluster(query)
        assert cid is not None

    def test_dimension_mismatch_raises(self):
        index = DensityIndex(dim=32)
        bad_emb = make_embeddings(5, 64)
        with pytest.raises(ValueError):
            index.add(bad_emb)

    def test_labels_preserved(self):
        index = DensityIndex(dim=16)
        emb = make_embeddings(6, 16)
        labels = ["a", "b", "c", "a", "b", "c"]
        index.add(emb, labels=labels)
        assert index._labels == labels
