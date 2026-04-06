"""Abstract base class for vector store adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from vectorsentinel.sentinel import Sentinel


class VectorStore(ABC):
    """Base adapter that wraps an external vector store and builds a Sentinel."""

    @abstractmethod
    def get_all_embeddings(self) -> tuple[NDArray[np.float32], list[str], list[str | None]]:
        """Return (embeddings, ids, labels) for all stored vectors."""
        ...

    def build_sentinel(
        self,
        threshold: float = 0.5,
        k: int = 5,
    ) -> Sentinel:
        """Fetch all embeddings from the store and build a Sentinel index."""
        embeddings, ids, labels = self.get_all_embeddings()
        sentinel = Sentinel.from_embeddings(
            embeddings=embeddings,
            ids=ids,
            labels=labels,
            threshold=threshold,
            k=k,
        )
        return sentinel
