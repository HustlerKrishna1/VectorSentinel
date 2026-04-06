# VectorSentinel

**Stop RAG hallucinations before the LLM is called.**

[![PyPI version](https://img.shields.io/pypi/v/vectorsentinel.svg)](https://pypi.org/project/vectorsentinel/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/vectorsentinel/vectorsentinel/ci.yml?label=tests)](https://github.com/vectorsentinel/vectorsentinel/actions)

A drop-in Python library that adds **honest "I don't know" intelligence** to any vector search pipeline. VectorSentinel sits between your vector store and your LLM — scoring retrieval confidence and blocking out-of-distribution queries before they ever reach the model.

```
User Query → [Embedding] → [Vector Store] → [VectorSentinel ✋] → LLM  or  "I don't know"
                                                      ↑
                              Density-based confidence scoring
                              Deterministic — same query, same verdict, always
                              Sub-millisecond on CPU, no GPU needed
```

---

## The Problem

Your RAG pipeline retrieves the top-k documents and hands them to the LLM — even when the query is completely outside your knowledge base. The LLM then fabricates a confident-sounding answer. This is a hallucination, and it happens silently.

**Standard RAG has no "I don't know" signal.** VectorSentinel adds one.

---

## Quickstart

```bash
pip install vectorsentinel
```

```python
from vectorsentinel import Sentinel
import numpy as np

# Build an index from your corpus embeddings
sentinel = Sentinel.from_embeddings(corpus_embeddings, labels=labels)

# Gate any query before calling your LLM
result = sentinel.gate(query_embedding)

if result.should_answer:
    # Proceed to retrieval + LLM
    docs = retriever.get_relevant_documents(query)
    answer = llm.invoke(query, context=docs)
else:
    answer = "I don't have reliable information about this topic."

# Full result object
print(result.confidence)           # 0.0 – 1.0
print(result.reason)               # "confident" | "query_out_of_distribution" | ...
print(result.nearest_neighbors)    # k nearest corpus vectors with similarity scores
```

---

## Why VectorSentinel?

| | Standard RAG | RAG + VectorSentinel |
|---|---|---|
| In-domain queries | ✅ Answers | ✅ Answers |
| Out-of-domain queries | ❌ Halluccinates | ✅ Abstains |
| Confidence score | ❌ None | ✅ 0.0 – 1.0 |
| Deterministic | ❌ Probabilistic | ✅ Same query = same verdict |
| GPU required | — | ❌ CPU only |
| Extra latency | — | < 1 ms (1K vectors), < 5 ms (10K vectors) |
| Dependencies | — | `numpy` only (core) |

### Benchmark Results

Synthetic benchmark — 1,000 corpus vectors, 256 dimensions, threshold=0.45:

| Metric | Value |
|---|---|
| In-domain answer rate | **96.5%** |
| OOD abstain rate | **88.0%** |
| Hallucination risk | **12.0%** → reduced from ~100% |
| Mean query latency | **< 2 ms** |
| p99 latency | **< 8 ms** |

Threshold is fully tunable — higher threshold = fewer hallucinations, more abstains.

---

## Integrations

### LangChain

```python
from vectorsentinel import Sentinel
from vectorsentinel.integrations.langchain import SentinelRetriever

sentinel = Sentinel.from_embeddings(corpus_emb, labels=labels)

# Wrap any existing LangChain retriever — 1 line
guarded = SentinelRetriever(base_retriever, sentinel, embed_fn=embed_model.embed_query)

# Use exactly like a normal retriever
chain = RetrievalQA.from_chain_type(llm=llm, retriever=guarded)
```

### LlamaIndex

```python
from vectorsentinel import Sentinel
from vectorsentinel.integrations.llamaindex import SentinelQueryEngine

sentinel = Sentinel.from_embeddings(corpus_emb)

engine = SentinelQueryEngine(
    query_engine=base_engine,
    sentinel=sentinel,
    embed_fn=lambda q: embed_model.get_text_embedding(q),
)

response = engine.query("What did ancient Martians eat?")
# → "I don't have reliable information about this topic."
```

### FAISS

```python
import faiss
from vectorsentinel.stores.faiss_store import FAISSStore

faiss_index = faiss.IndexFlatIP(768)
faiss_index.add(embeddings)

store = FAISSStore(faiss_index, ids=doc_ids, labels=labels)
sentinel = store.build_sentinel(threshold=0.5)
```

### ChromaDB

```python
import chromadb
from vectorsentinel.stores.chroma_store import ChromaStore

collection = chromadb.Client().get_collection("my_docs")
store = ChromaStore(collection, label_metadata_key="category")
sentinel = store.build_sentinel(threshold=0.5)
```

### REST API (any language)

```bash
# Start server
sentinel serve --index my_index.npz --threshold 0.5 --port 8000

# Gate a query from Node.js, Go, Ruby, etc.
curl -X POST http://localhost:8000/gate \
  -H "Content-Type: application/json" \
  -d '{"embedding": [0.1, 0.2, ...], "threshold": 0.5}'

# Response
{
  "should_answer": false,
  "confidence": 0.21,
  "reason": "query_out_of_distribution",
  "neighbors": [...]
}
```

---

## Core Concepts

### Confidence Scoring

VectorSentinel computes confidence as a weighted combination of three signals:

1. **Proximity** — cosine similarity to the nearest neighbor
2. **Density** — how dense the local neighborhood is relative to the corpus baseline
3. **Agreement** — label consensus among k nearest neighbors

```python
confidence = (
    0.3 * proximity_score +
    0.4 * relative_density_score +
    0.3 * neighbor_agreement_score
)
```

All three are deterministic — no random numbers, no sampling.

### Determinism

> "Vector search should not depend on dice." — Re-kNN philosophy

Given the same index and the same query, VectorSentinel always returns the same verdict. This enables:
- Reproducible audit logs
- A/B testing of thresholds without variance
- Compliance in regulated industries

### Threshold Tuning

```python
# Tune threshold on a held-out evaluation set
optimal_threshold = sentinel.auto_calibrate(
    holdout_queries=eval_embeddings,
    in_domain_mask=[True, True, False, True, False, ...],
    target_abstain_rate=0.05,  # abstain on 5% of OOD queries minimum
)
```

Or sweep manually:

| Threshold | In-domain answer rate | OOD abstain rate |
|---|---|---|
| 0.3 | 99% | 65% |
| 0.5 | 96% | 85% |
| 0.7 | 88% | 95% |
| 0.9 | 70% | 99% |

---

## Full API

```python
from vectorsentinel import Sentinel

# Build
sentinel = Sentinel(dim=768, threshold=0.5, k=5)
sentinel.add(embeddings, ids=doc_ids, labels=categories)

# Gate
result = sentinel.gate(query_embedding)
results = sentinel.gate_batch(query_matrix)           # many at once

# Manage
sentinel.delete(["doc_1", "doc_2"])                   # remove by ID
sentinel.refine()                                     # re-optimize clusters
sentinel.cluster_report()                             # cluster health

# Calibrate
sentinel.auto_calibrate(holdout_queries, in_domain_mask)

# Persist
sentinel.save("my_index")                            # → my_index.npz
sentinel = Sentinel.load("my_index.npz")

# Benchmark
stats = sentinel.benchmark(queries)                  # latency percentiles
```

### GateResult

```python
result = sentinel.gate(query)

result.should_answer        # bool
result.verdict              # GateVerdict.ANSWER | GateVerdict.ABSTAIN
result.confidence           # float, 0.0–1.0
result.reason               # str: "confident" | "query_out_of_distribution" | ...
result.nearest_neighbors    # list[NeighborResult] with .id, .similarity, .label
result.nearest_cluster_id   # int
result.query_density        # float
result.neighbor_agreement   # float
result.to_dict()            # JSON-serializable dict
```

---

## Installation

```bash
# Core (numpy only)
pip install vectorsentinel

# With integrations
pip install vectorsentinel[faiss]      # FAISS support
pip install vectorsentinel[chroma]     # ChromaDB support
pip install vectorsentinel[langchain]  # LangChain support
pip install vectorsentinel[llamaindex] # LlamaIndex support
pip install vectorsentinel[server]     # REST API server
pip install vectorsentinel[all]        # Everything
```

---

## Examples

```bash
# Quickstart demo (no dependencies beyond numpy)
python examples/quickstart.py

# Hallucination stopper demo
python examples/demo_hallucination.py

# LangChain integration
python examples/langchain_rag.py

# Full benchmark suite
python benchmarks/synthetic_benchmark.py
```

---

## CLI

```bash
# Start REST server
sentinel serve --index my_index.npz --port 8000

# Print index info
sentinel info --index my_index.npz

# Latency benchmark
sentinel benchmark --index my_index.npz --queries queries.npy
```

---

## Design Philosophy

VectorSentinel is inspired by the [Re-kNN](https://github.com/ToshikatsuOkadaFSS/Re-kNN) system by Toshikatsu Okada (Fuuta System Service LLC), which introduced the idea that vector search should be **deterministic** and **honest about uncertainty**. We implement these principles as an open-source, cross-platform Python library.

**Key principles:**

- **Honest uncertainty** — "I don't know" is a valid, first-class answer
- **Determinism** — no random seeds, no sampling, reproducible results
- **Density-awareness** — confidence reflects the natural structure of your data
- **Zero friction** — `pip install` and 3 lines of code; no GPU, no cloud dependency
- **Composable** — works alongside any embedding model, vector store, or LLM

---

## When to Use VectorSentinel

✅ **Good fit:**
- RAG pipelines serving diverse, unpredictable user queries
- Customer-facing chatbots where hallucinations are costly
- Domain-specific assistants (legal, medical, finance)
- Compliance environments requiring auditable, reproducible decisions
- Edge deployment without GPU resources

❌ **Not designed for:**
- Replacing semantic search (use FAISS/Chroma for that)
- Tasks where "I don't know" is never acceptable
- Real-time streaming with > 100K corpus vectors and < 1ms budget

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
git clone https://github.com/vectorsentinel/vectorsentinel
cd vectorsentinel
pip install -e ".[dev]"
pytest
```

---

## License

Apache 2.0. See [LICENSE](LICENSE).

---

*VectorSentinel is inspired by [Re-kNN](https://github.com/ToshikatsuOkadaFSS/Re-kNN) by Toshikatsu Okada. We encourage you to check out the original work.*
