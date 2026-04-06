"""VectorSentinel — Stop RAG hallucinations before the LLM is called.

A drop-in Python library that adds honest "I don't know" intelligence to any
vector search pipeline. Inspired by the Re-kNN philosophy that vector search
should be deterministic and honest about uncertainty.

Quick start:

    from vectorsentinel import Sentinel
    import numpy as np

    sentinel = Sentinel.from_embeddings(corpus_embeddings, labels=labels)

    result = sentinel.gate(query_embedding)
    print(result.should_answer)   # True / False
    print(result.confidence)      # 0.0 – 1.0
    print(result.reason)          # "confident" | "query_out_of_distribution" | ...

Attribution:
    Core philosophy inspired by Re-kNN by Toshikatsu Okada (Fuuta System Service LLC).
    https://github.com/ToshikatsuOkadaFSS/Re-kNN
"""

from vectorsentinel.core.gate import GateResult, GateVerdict
from vectorsentinel.core.index import DensityIndex, NeighborResult
from vectorsentinel.sentinel import Sentinel

__version__ = "0.1.0"
__all__ = [
    "Sentinel",
    "DensityIndex",
    "GateResult",
    "GateVerdict",
    "NeighborResult",
]
