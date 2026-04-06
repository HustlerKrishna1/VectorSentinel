"""Density-aware vector index.

The core data structure: stores embeddings with metadata, supports exact kNN
search, and maintains density statistics for confidence scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vectorsentinel.core.cluster import Cluster, DensityClusterer


@dataclass
class NeighborResult:
    """A single nearest-neighbor result."""

    id: str
    index: int
    similarity: float
    label: str | None = None


class DensityIndex:
    """A deterministic, density-aware vector index.

    Stores L2-normalized embeddings and provides:
    - Exact k-nearest-neighbor search via cosine similarity
    - Per-vector density statistics
    - Cluster structure for OOD detection
    """

    def __init__(self, dim: int):
        self.dim = dim
        self._embeddings: list[NDArray[np.float32]] = []
        self._ids: list[str] = []
        self._labels: list[str | None] = []
        self._matrix: NDArray[np.float32] | None = None  # (n, dim) cached
        self._dirty = True  # matrix needs rebuild

        # Density stats
        self._densities: NDArray[np.float64] | None = None
        self._clusters: list[Cluster] = []
        self._cluster_centroids: NDArray[np.float32] | None = None
        self._mean_density: float = 0.0
        self._density_computed = False

    @property
    def size(self) -> int:
        return len(self._embeddings)

    @property
    def mean_density(self) -> float:
        if not self._density_computed:
            self._compute_density()
        return self._mean_density

    @property
    def clusters(self) -> list[Cluster]:
        if not self._density_computed:
            self._compute_density()
        return self._clusters

    def add(
        self,
        embeddings: NDArray[np.float32],
        ids: list[str] | None = None,
        labels: list[str] | None = None,
    ) -> None:
        """Add vectors to the index.

        Parameters
        ----------
        embeddings : array of shape (n, dim) or (dim,)
        ids : optional string IDs (auto-generated if not provided)
        labels : optional labels for classification / agreement scoring
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dim {embeddings.shape[1]} != index dim {self.dim}"
            )

        n = embeddings.shape[0]

        # L2-normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = (embeddings / norms).astype(np.float32)

        if ids is None:
            base = self.size
            ids = [str(base + i) for i in range(n)]
        if labels is None:
            labels = [None] * n

        if len(ids) != n or len(labels) != n:
            raise ValueError("ids and labels must match embedding count")

        for i in range(n):
            self._embeddings.append(normalized[i])
            self._ids.append(ids[i])
            self._labels.append(labels[i])

        self._dirty = True
        self._density_computed = False

    def _build_matrix(self) -> None:
        if not self._dirty or not self._embeddings:
            return
        self._matrix = np.stack(self._embeddings, axis=0)
        self._dirty = False

    def _compute_density(self, density_k: int = 10) -> None:
        """Compute per-vector density and cluster structure."""
        if self.size == 0:
            self._mean_density = 0.0
            self._density_computed = True
            return

        self._build_matrix()
        assert self._matrix is not None
        n = self._matrix.shape[0]
        k = min(density_k, n - 1)

        if k == 0:
            self._densities = np.ones(1, dtype=np.float64)
            self._mean_density = 1.0
        else:
            # Compute densities in chunks
            chunk_size = 2048
            densities = np.zeros(n, dtype=np.float64)
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                sims = self._matrix[start:end] @ self._matrix.T
                for i in range(start, end):
                    sims[i - start, i] = -1.0
                if k < n - 1:
                    topk_idx = np.argpartition(sims, -k, axis=1)[:, -k:]
                    topk_sims = np.take_along_axis(sims, topk_idx, axis=1)
                else:
                    topk_sims = sims
                densities[start:end] = topk_sims.mean(axis=1)

            self._densities = densities
            self._mean_density = float(densities.mean())

        # Build clusters
        labels_list = [l if l is not None else f"_unlabeled_{i}" for i, l in enumerate(self._labels)]
        clusterer = DensityClusterer()
        self._clusters = clusterer.fit(self._matrix, labels_list)

        if self._clusters:
            self._cluster_centroids = np.stack(
                [c.centroid for c in self._clusters], axis=0
            )
        else:
            self._cluster_centroids = None

        self._density_computed = True

    def search(self, query: NDArray[np.float32], k: int = 5) -> list[NeighborResult]:
        """Exact k-nearest-neighbor search by cosine similarity.

        Parameters
        ----------
        query : L2-normalized vector of shape (dim,)
        k : number of neighbors

        Returns
        -------
        list of NeighborResult, sorted by descending similarity
        """
        if self.size == 0:
            return []

        self._build_matrix()
        assert self._matrix is not None

        # Normalize query
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        sims = self._matrix @ query  # (n,)
        k = min(k, self.size)

        if k < self.size:
            top_indices = np.argpartition(sims, -k)[-k:]
        else:
            top_indices = np.arange(self.size)

        top_indices = top_indices[np.argsort(-sims[top_indices])]

        results = []
        for idx in top_indices:
            results.append(
                NeighborResult(
                    id=self._ids[idx],
                    index=int(idx),
                    similarity=float(sims[idx]),
                    label=self._labels[idx],
                )
            )
        return results

    def nearest_cluster(self, query: NDArray[np.float32]) -> int | None:
        """Find the nearest cluster to a query vector."""
        if not self._density_computed:
            self._compute_density()

        if self._cluster_centroids is None or len(self._clusters) == 0:
            return None

        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        sims = self._cluster_centroids @ query
        return int(np.argmax(sims))

    def delete(self, ids: list[str]) -> int:
        """Remove vectors by ID. Returns count of removed vectors."""
        id_set = set(ids)
        removed = 0
        new_embeddings = []
        new_ids = []
        new_labels = []

        for i, vid in enumerate(self._ids):
            if vid in id_set:
                removed += 1
            else:
                new_embeddings.append(self._embeddings[i])
                new_ids.append(self._ids[i])
                new_labels.append(self._labels[i])

        self._embeddings = new_embeddings
        self._ids = new_ids
        self._labels = new_labels
        self._dirty = True
        self._density_computed = False

        return removed

    def refine(self) -> int:
        """Re-evaluate cluster assignments and fix suboptimal placements.

        Returns the number of vectors that were reassigned.
        """
        if not self._density_computed:
            self._compute_density()

        if not self._clusters or self._cluster_centroids is None:
            return 0

        self._build_matrix()
        assert self._matrix is not None

        reassigned = 0
        # For each vector, check if it's closer to a different cluster centroid
        sims = self._matrix @ self._cluster_centroids.T  # (n, n_clusters)
        best_clusters = np.argmax(sims, axis=1)

        # Build current assignment map
        current_assignment = np.full(self.size, -1, dtype=np.int64)
        for cluster in self._clusters:
            for m in cluster.member_indices:
                if m < self.size:
                    current_assignment[m] = cluster.cluster_id

        # Count reassignments
        for i in range(self.size):
            if current_assignment[i] != best_clusters[i]:
                reassigned += 1

        if reassigned > 0:
            # Rebuild clusters with optimal assignments
            self._density_computed = False
            self._compute_density()

        return reassigned

    def cluster_report(self) -> list[dict]:
        """Return a summary of all clusters."""
        if not self._density_computed:
            self._compute_density()

        return [
            {
                "cluster_id": c.cluster_id,
                "size": c.size,
                "density": round(c.density, 4),
                "radius": round(c.radius, 4),
                "purity": round(c.purity, 4),
                "top_label": max(c.label_counts, key=c.label_counts.get)
                if c.label_counts
                else None,
            }
            for c in self._clusters
        ]

    def save(self, path: str) -> None:
        """Persist the index to disk."""
        self._build_matrix()
        data = {
            "dim": self.dim,
            "embeddings": self._matrix,
            "ids": np.array(self._ids, dtype=object),
            "labels": np.array(self._labels, dtype=object),
        }
        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, path: str) -> DensityIndex:
        """Load an index from disk."""
        if not path.endswith(".npz"):
            path += ".npz"
        data = np.load(path, allow_pickle=True)
        dim = int(data["dim"])
        index = cls(dim=dim)
        embeddings = data["embeddings"]
        ids = data["ids"].tolist()
        labels = data["labels"].tolist()
        index.add(embeddings, ids=ids, labels=labels)
        return index
