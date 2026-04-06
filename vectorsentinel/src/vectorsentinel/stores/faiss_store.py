"""FAISS vector store adapter.

Requires: pip install vectorsentinel[faiss]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from vectorsentinel.stores.base import VectorStore

if TYPE_CHECKING:
    pass


class FAISSStore(VectorStore):
    """Adapter for a FAISS flat index.

    Example::

        import faiss, numpy as np
        from vectorsentinel.stores.faiss_store import FAISSStore

        index = faiss.IndexFlatIP(768)
        index.add(embeddings)
        store = FAISSStore(index, ids=doc_ids, labels=labels)
        sentinel = store.build_sentinel(threshold=0.6)
    """

    def __init__(
        self,
        faiss_index,
        ids: list[str] | None = None,
        labels: list[str] | None = None,
    ):
        try:
            import faiss  # noqa: F401
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for FAISSStore. "
                "Install with: pip install vectorsentinel[faiss]"
            )
        self._faiss_index = faiss_index
        n = faiss_index.ntotal
        self._ids = ids or [str(i) for i in range(n)]
        self._labels = labels or [None] * n

    def get_all_embeddings(
        self,
    ) -> tuple[NDArray[np.float32], list[str], list[str | None]]:
        import faiss

        n = self._faiss_index.ntotal
        dim = self._faiss_index.d
        # Reconstruct all vectors from index
        embeddings = np.zeros((n, dim), dtype=np.float32)
        self._faiss_index.reconstruct_n(0, n, embeddings)
        return embeddings, self._ids, self._labels
