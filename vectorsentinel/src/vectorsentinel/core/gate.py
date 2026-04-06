"""Confidence gating — the core decision engine.

Given a query embedding and an index, computes a confidence score and returns
a structured verdict: answer or abstain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from vectorsentinel.core.index import DensityIndex, NeighborResult


class GateVerdict(str, Enum):
    """The gate's binary decision."""

    ANSWER = "answer"
    ABSTAIN = "abstain"


@dataclass
class GateResult:
    """Structured output from the confidence gate."""

    verdict: GateVerdict
    confidence: float
    reason: str
    nearest_neighbors: list[NeighborResult] = field(default_factory=list)
    nearest_cluster_id: int | None = None
    query_density: float = 0.0
    neighbor_agreement: float = 0.0

    @property
    def should_answer(self) -> bool:
        return self.verdict == GateVerdict.ANSWER

    def to_dict(self) -> dict:
        return {
            "should_answer": self.should_answer,
            "verdict": self.verdict.value,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "nearest_cluster_id": self.nearest_cluster_id,
            "query_density": round(self.query_density, 4),
            "neighbor_agreement": round(self.neighbor_agreement, 4),
            "neighbors": [
                {"id": n.id, "similarity": round(n.similarity, 4), "label": n.label}
                for n in self.nearest_neighbors
            ],
        }


def gate_query(
    query: NDArray[np.float32],
    index: DensityIndex,
    threshold: float = 0.5,
    k: int = 5,
    density_weight: float = 0.4,
    agreement_weight: float = 0.3,
    proximity_weight: float = 0.3,
) -> GateResult:
    """Score a query and decide whether the system should answer or abstain.

    Confidence is a weighted combination of three signals:
    1. **Proximity**: cosine similarity to the nearest neighbor
    2. **Density**: how dense the local neighborhood is relative to corpus average
    3. **Agreement**: fraction of k neighbors sharing the most common label

    Parameters
    ----------
    query : L2-normalized embedding vector
    index : built DensityIndex
    threshold : confidence below this triggers abstain
    k : number of neighbors to consider
    density_weight, agreement_weight, proximity_weight : signal weights (must sum to 1)
    """
    if index.size == 0:
        return GateResult(
            verdict=GateVerdict.ABSTAIN,
            confidence=0.0,
            reason="empty_index",
        )

    # Normalize query
    norm = np.linalg.norm(query)
    if norm > 0:
        query = query / norm

    # Find k nearest neighbors
    neighbors = index.search(query, k=k)

    if not neighbors:
        return GateResult(
            verdict=GateVerdict.ABSTAIN,
            confidence=0.0,
            reason="no_neighbors_found",
        )

    # Signal 1: Proximity — similarity to nearest neighbor
    proximity = max(0.0, neighbors[0].similarity)

    # Signal 2: Density — average similarity to k neighbors vs corpus baseline
    neighbor_sims = np.array([n.similarity for n in neighbors], dtype=np.float64)
    query_density = float(neighbor_sims.mean())
    corpus_density = index.mean_density
    # Ratio clamped to [0, 1]
    if corpus_density > 0:
        relative_density = min(1.0, query_density / corpus_density)
    else:
        relative_density = query_density

    # Signal 3: Agreement — label consensus among neighbors
    if any(n.label is not None for n in neighbors):
        label_counts: dict[str | None, int] = {}
        for n in neighbors:
            label_counts[n.label] = label_counts.get(n.label, 0) + 1
        max_count = max(label_counts.values())
        agreement = max_count / len(neighbors)
    else:
        agreement = 1.0  # No labels → no disagreement signal

    # Weighted confidence
    total_weight = density_weight + agreement_weight + proximity_weight
    confidence = (
        proximity_weight * proximity
        + density_weight * relative_density
        + agreement_weight * agreement
    ) / total_weight

    confidence = float(np.clip(confidence, 0.0, 1.0))

    # Find nearest cluster
    nearest_cluster_id = index.nearest_cluster(query)

    # Verdict
    if confidence >= threshold:
        verdict = GateVerdict.ANSWER
        reason = "confident"
    else:
        verdict = GateVerdict.ABSTAIN
        reasons = []
        if proximity < threshold:
            reasons.append("low_proximity")
        if relative_density < 0.5:
            reasons.append("sparse_neighborhood")
        if agreement < 0.5:
            reasons.append("label_disagreement")
        reason = "query_out_of_distribution" if not reasons else "+".join(reasons)

    return GateResult(
        verdict=verdict,
        confidence=confidence,
        reason=reason,
        nearest_neighbors=neighbors,
        nearest_cluster_id=nearest_cluster_id,
        query_density=query_density,
        neighbor_agreement=agreement,
    )
