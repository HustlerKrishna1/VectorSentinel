"""Pure-numpy in-memory vector store — zero extra dependencies."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from vectorsentinel.sentinel import Sentinel
from vectorsentinel.stores.base import VectorStore


class NumpyStore(VectorStore):
    """Simple in-memory store backed by numpy arrays.

    Useful for testing, small datasets, or when you want to build a Sentinel
    from an existing embedding matrix without any external vector DB.

    Example::

        store = NumpyStore(embeddings, ids=doc_ids, labels=categories)
        sentinel = store.build_sentinel(threshold=0.6)
    """

    def __init__(
        self,
        embeddings: NDArray[np.float32],
        ids: list[str] | None = None,
        labels: list[str] | None = None,
    ):
        self._embeddings = np.asarray(embeddings, dtype=np.float32)
        if self._embeddings.ndim == 1:
            self._embeddings = self._embeddings.reshape(1, -1)
        n = len(self._embeddings)
        self._ids = ids or [str(i) for i in range(n)]
        self._labels = labels or [None] * n

    def get_all_embeddings(
        self,
    ) -> tuple[NDArray[np.float32], list[str], list[str | None]]:
        return self._embeddings, self._ids, self._labels

    @classmethod
    def from_sentinel(cls, sentinel: Sentinel) -> "NumpyStore":
        """Reconstruct a NumpyStore from an existing Sentinel index."""
        index = sentinel._index
        index._build_matrix()
        return cls(
            embeddings=index._matrix,
            ids=index._ids,
            labels=index._labels,
        )
