"""The Hallucination Stopper Demo.

Simulates a RAG pipeline on two topics:
  - KNOWN topic: Python programming (in-domain)
  - UNKNOWN topic: Ancient Martian civilizations (out-of-domain)

Without VectorSentinel: the LLM would hallucinate on the unknown topic.
With VectorSentinel: the gate catches the OOD query and returns "I don't know".

This demo uses synthetic embeddings (no LLM required) to illustrate the concept.
Replace `fake_embed()` with your actual embedding model for real use.

Run:
    python examples/demo_hallucination.py
"""

import numpy as np

from vectorsentinel import Sentinel

print("=" * 65)
print(" VectorSentinel -- The Hallucination Stopper Demo")
print("=" * 65)
print()

# -- Corpus: Python programming knowledge base -----------------------------
# In a real system these would be sentence-transformer or OpenAI embeddings.
# Here we simulate them as tight clusters in 768-D space.

DIM = 768
rng = np.random.RandomState(0)

print("[Building knowledge base: Python programming docs]")

TOPICS = {
    "python_basics":      ("Variables, types, and control flow in Python", 80),
    "python_functions":   ("Defining and calling functions, lambda, closures", 80),
    "python_oop":         ("Classes, inheritance, dunder methods",           80),
    "python_stdlib":      ("os, sys, pathlib, json, datetime modules",       80),
    "python_async":       ("asyncio, await, event loops, coroutines",        80),
}

def make_cluster(center_seed: int, n: int, noise: float = 0.08):
    rng_c = np.random.RandomState(center_seed)
    center = rng_c.randn(DIM).astype(np.float32)
    center /= np.linalg.norm(center)
    noise_vecs = rng.randn(n, DIM).astype(np.float32) * (noise / DIM ** 0.5)
    vecs = center + noise_vecs
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / norms).astype(np.float32), center

all_emb, all_labels, all_ids = [], [], []
cluster_centers = {}
for seed, (topic_name, topic_info) in enumerate(TOPICS.items()):
    description, n = topic_info
    vecs, center = make_cluster(seed, n)
    all_emb.append(vecs)
    all_labels.extend([topic_name] * n)
    all_ids.extend([f"{topic_name}_{j}" for j in range(n)])
    cluster_centers[topic_name] = center
    print(f"  + {topic_name}: {n} documents")

corpus = np.vstack(all_emb)
print(f"  Total: {len(corpus)} documents\n")

# -- Build Sentinel --------------------------------------------------------
sentinel = Sentinel.from_embeddings(
    corpus,
    ids=all_ids,
    labels=all_labels,
    threshold=0.45,
    k=7,
)

# -- Simulate queries ------------------------------------------------------

def fake_embed_in_domain(topic: str) -> np.ndarray:
    """Simulate embedding a query about a known topic."""
    center = cluster_centers[topic]
    noise = rng.randn(DIM).astype(np.float32) * 0.12
    v = center + noise
    v /= np.linalg.norm(v)
    return v

def fake_embed_ood() -> np.ndarray:
    """Simulate embedding a query about an unknown topic (Mars civilizations)."""
    # A random direction far from all known clusters
    for _ in range(200):
        v = rng.randn(DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        if float((corpus @ v).max()) < 0.15:
            return v
    v = rng.randn(DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v

def simulate_llm(query: str, docs_retrieved: int) -> str:
    """Fake LLM that always generates an answer (hallucination risk)."""
    if docs_retrieved == 0:
        return "(no context provided -- LLM fabricates from priors)"
    return f"(LLM answers using {docs_retrieved} retrieved documents)"

# -- Run comparison --------------------------------------------------------
queries = [
    ("What is a Python closure?",                  "python_functions", True),
    ("How does asyncio work in Python?",            "python_async",     True),
    ("What is the Martian ancient pyramid theory?", None,               False),
    ("Explain alien archaeology on Mars",           None,               False),
    ("How do Python classes use __init__?",         "python_oop",       True),
]

print("-" * 65)
print(f"{'QUERY':<42} {'WITHOUT':>10} {'WITH':>10}")
print(f"{'':42} {'SENTINEL':>10} {'SENTINEL':>10}")  # noqa
print("-" * 65)

n_hallucinations_prevented = 0
for query_text, topic, is_in_domain in queries:
    # Embed query
    if is_in_domain:
        q_emb = fake_embed_in_domain(topic)
    else:
        q_emb = fake_embed_ood()

    # Gate
    gate_result = sentinel.gate(q_emb)

    # Simulate what happens WITHOUT sentinel: always retrieve 5 docs
    without = "ANSWERS"

    # WITH sentinel:
    if gate_result.should_answer:
        with_sentinel = f"ANSWERS ({gate_result.confidence:.2f})"
    else:
        with_sentinel = f"ABSTAINS ({gate_result.confidence:.2f})"
        if not is_in_domain:
            n_hallucinations_prevented += 1

    trunc_query = (query_text[:39] + "...") if len(query_text) > 42 else query_text
    print(f"{trunc_query:<42} {without:>10} {with_sentinel:>20}")

print("-" * 65)
print(f"\nHallucinations prevented: {n_hallucinations_prevented}/{sum(1 for _,_,ind in queries if not ind)}")

# -- Threshold sensitivity -------------------------------------------------
print(f"\n{'-'*65}")
print("Threshold sensitivity (in-domain query: 'What is a Python closure?'):")
print("-" * 65)
q = fake_embed_in_domain("python_functions")
for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
    r = sentinel.gate(q, threshold=t)
    icon = "ANSWERS " if r.should_answer else "ABSTAINS"
    print(f"  threshold={t:.1f}  -> {icon}  confidence={r.confidence:.4f}")

print(f"\n{'-'*65}")
print("Threshold sensitivity (OOD query: 'Martian pyramid theory'):")
print("-" * 65)
q_ood = fake_embed_ood()
for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
    r = sentinel.gate(q_ood, threshold=t)
    icon = "ANSWERS " if r.should_answer else "ABSTAINS"
    print(f"  threshold={t:.1f}  -> {icon}  confidence={r.confidence:.4f}")

print()
print("Demo complete. In production, replace fake_embed() with your")
print("actual embedding model (OpenAI, HuggingFace, Cohere, etc.).")
