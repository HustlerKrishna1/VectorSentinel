"""ChromaDB vector store adapter.

Requires: pip install vectorsentinel[chroma]
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from vectorsentinel.stores.base import VectorStore


class ChromaStore(VectorStore):
    """Adapter for a ChromaDB collection.

    Fetches all embeddings and builds a Sentinel from them.

    Example::

        import chromadb
        from vectorsentinel.stores.chroma_store import ChromaStore

        client = chromadb.Client()
        collection = client.get_collection("my_docs")

        store = ChromaStore(collection, label_metadata_key="category")
        sentinel = store.build_sentinel(threshold=0.6)
    """

    def __init__(self, collection, label_metadata_key: str | None = "label"):
        try:
            import chromadb  # noqa: F401
        except ImportError:
            raise ImportError(
                "chromadb is required for ChromaStore. "
                "Install with: pip install vectorsentinel[chroma]"
            )
        self._collection = collection
        self._label_key = label_metadata_key

    def get_all_embeddings(
        self,
    ) -> tuple[NDArray[np.float32], list[str], list[str | None]]:
        result = self._collection.get(include=["embeddings", "metadatas"])
        embeddings = np.array(result["embeddings"], dtype=np.float32)
        ids = result["ids"]

        labels: list[str | None] = []
        if self._label_key and result.get("metadatas"):
            for meta in result["metadatas"]:
                labels.append(meta.get(self._label_key) if meta else None)
        else:
            labels = [None] * len(ids)

        return embeddings, ids, labels
