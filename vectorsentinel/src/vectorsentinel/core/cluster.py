"""Density-based clustering for vector embeddings.

Implements a deterministic, density-aware partitioning algorithm inspired by Re-kNN.
No random seeds, no probabilistic approximation — same input always produces
the same clusters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class Cluster:
    """A single density cluster with its statistics."""

    cluster_id: int
    centroid: NDArray[np.float32]
    member_indices: list[int] = field(default_factory=list)
    radius: float = 0.0
    density: float = 0.0
    label_counts: dict[str, int] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.member_indices)

    @property
    def purity(self) -> float:
        """Fraction of members belonging to the dominant label."""
        if not self.label_counts:
            return 1.0
        max_count = max(self.label_counts.values())
        total = sum(self.label_counts.values())
        return max_count / total if total > 0 else 1.0


class DensityClusterer:
    """Deterministic density-based clusterer.

    Uses a greedy centroid-absorption approach:
    1. Sort vectors by local density (descending).
    2. The densest unassigned vector becomes a new centroid.
    3. Absorb all unassigned neighbors within `absorption_radius` similarity.
    4. Repeat until all vectors are assigned.

    This is deterministic because sorting by density (with index-based tiebreaking)
    produces a fixed ordering, and absorption is a greedy deterministic scan.
    """

    def __init__(
        self,
        absorption_threshold: float = 0.70,
        min_cluster_size: int = 1,
        density_k: int = 10,
    ):
        self.absorption_threshold = absorption_threshold
        self.min_cluster_size = min_cluster_size
        self.density_k = density_k

    def fit(
        self,
        embeddings: NDArray[np.float32],
        labels: list[str] | None = None,
    ) -> list[Cluster]:
        """Cluster embeddings deterministically.

        Parameters
        ----------
        embeddings : array of shape (n, dim), L2-normalized
        labels : optional label per vector (for purity tracking)

        Returns
        -------
        list of Cluster objects
        """
        n = embeddings.shape[0]
        if n == 0:
            return []

        # Compute local density for each vector: average cosine similarity to k nearest neighbors
        k = min(self.density_k, n - 1)
        if k == 0:
            # Single vector
            cluster = Cluster(
                cluster_id=0,
                centroid=embeddings[0].copy(),
                member_indices=[0],
                radius=0.0,
                density=1.0,
            )
            if labels:
                cluster.label_counts = {labels[0]: 1}
            return [cluster]

        # Batch cosine similarity via matmul (embeddings are L2-normalized)
        # Process in chunks to avoid OOM on large datasets
        chunk_size = 2048
        densities = np.zeros(n, dtype=np.float64)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            sims = embeddings[start:end] @ embeddings.T  # (chunk, n)
            # Zero out self-similarity
            for i in range(start, end):
                sims[i - start, i] = -1.0
            # Top-k similarities
            if k < n - 1:
                topk_indices = np.argpartition(sims, -k, axis=1)[:, -k:]
                topk_sims = np.take_along_axis(sims, topk_indices, axis=1)
            else:
                topk_sims = sims
            densities[start:end] = topk_sims.mean(axis=1)

        # Sort by density descending, with index as tiebreaker for determinism
        order = np.lexsort((np.arange(n), -densities))

        assigned = np.full(n, -1, dtype=np.int64)
        clusters: list[Cluster] = []
        cluster_id = 0

        for idx in order:
            if assigned[idx] >= 0:
                continue

            # This vector becomes a new centroid
            centroid = embeddings[idx].copy()

            # Find all unassigned vectors within absorption threshold
            sims_to_centroid = embeddings @ centroid  # (n,)
            candidates = np.where(
                (assigned < 0) & (sims_to_centroid >= self.absorption_threshold)
            )[0]

            # Sort candidates by similarity (descending) for deterministic absorption
            candidate_sims = sims_to_centroid[candidates]
            absorption_order = np.argsort(-candidate_sims)
            members = candidates[absorption_order].tolist()

            if len(members) < self.min_cluster_size:
                members = [idx]

            # Assign members
            for m in members:
                assigned[m] = cluster_id

            # Compute cluster stats
            member_embeddings = embeddings[members]
            actual_centroid = member_embeddings.mean(axis=0)
            norm = np.linalg.norm(actual_centroid)
            if norm > 0:
                actual_centroid /= norm

            sims_to_actual = member_embeddings @ actual_centroid
            radius = float(1.0 - sims_to_actual.min()) if len(members) > 1 else 0.0

            cluster = Cluster(
                cluster_id=cluster_id,
                centroid=actual_centroid.astype(np.float32),
                member_indices=members,
                radius=radius,
                density=float(densities[idx]),
            )

            if labels:
                lc: dict[str, int] = {}
                for m in members:
                    lbl = labels[m]
                    lc[lbl] = lc.get(lbl, 0) + 1
                cluster.label_counts = lc

            clusters.append(cluster)
            cluster_id += 1

        return clusters
