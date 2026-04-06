"""VectorSentinel Quickstart — runs with zero external dependencies.

Demonstrates the full lifecycle:
1. Build a corpus of embeddings (simulated here with random vectors)
2. Build a Sentinel index
3. Gate in-domain queries (should answer)
4. Gate out-of-domain queries (should abstain)
5. Show cluster report and latency benchmark

Run:
    python examples/quickstart.py
"""

import time

import numpy as np

from vectorsentinel import Sentinel

# -- 1. Simulate a corpus --------------------------------------------------
print("=" * 60)
print("VectorSentinel Quickstart")
print("=" * 60)

DIM = 128
N_CORPUS = 500
N_CLUSTERS = 5
rng = np.random.RandomState(42)

print(f"\n[1] Building corpus: {N_CORPUS} vectors, dim={DIM}, {N_CLUSTERS} topics")
corpus_embeddings = []
corpus_labels = []
corpus_ids = []

for cluster_idx in range(N_CLUSTERS):
    # Create a tight cluster around a random center.
    # Scale noise by 1/sqrt(DIM) so intra-cluster cosine sim stays high (~0.95).
    center = rng.randn(DIM).astype(np.float32)
    center /= np.linalg.norm(center)
    n_per_cluster = N_CORPUS // N_CLUSTERS
    noise = rng.randn(n_per_cluster, DIM).astype(np.float32) * (0.2 / DIM ** 0.5)
    cluster_emb = center + noise
    norms = np.linalg.norm(cluster_emb, axis=1, keepdims=True)
    cluster_emb /= norms
    corpus_embeddings.append(cluster_emb)
    corpus_labels.extend([f"topic_{cluster_idx}"] * n_per_cluster)
    corpus_ids.extend([f"doc_{cluster_idx}_{j}" for j in range(n_per_cluster)])

corpus = np.vstack(corpus_embeddings)

# -- 2. Build Sentinel -----------------------------------------------------
print(f"\n[2] Building Sentinel index (threshold=0.5, k=5) ...")
t0 = time.perf_counter()
sentinel = Sentinel.from_embeddings(
    corpus,
    ids=corpus_ids,
    labels=corpus_labels,
    threshold=0.45,
    k=7,
)
build_time = (time.perf_counter() - t0) * 1000
print(f"    Built in {build_time:.1f} ms")
print(f"    {sentinel}")

# -- 3. Cluster report -----------------------------------------------------
print(f"\n[3] Cluster report:")
report = sentinel.cluster_report()
print(f"    Total clusters: {len(report)}")
avg_purity = sum(c["purity"] for c in report) / len(report) if report else 0
print(f"    Avg purity: {avg_purity:.3f}")
print(f"    Top 3 clusters by size:")
for c in sorted(report, key=lambda x: -x["size"])[:3]:
    print(
        f"      cluster={c['cluster_id']:3d}  size={c['size']:4d}  "
        f"purity={c['purity']:.3f}  label={c['top_label']}"
    )

# -- 4. In-domain queries --------------------------------------------------
print(f"\n[4] Gating IN-DOMAIN queries (should answer):")
in_domain_queries = corpus[rng.choice(N_CORPUS, 5, replace=False)]
for i, q in enumerate(in_domain_queries):
    result = sentinel.gate(q)
    icon = "OK" if result.should_answer else "NO"
    print(
        f"    [{icon}] query_{i}  confidence={result.confidence:.3f}  "
        f"verdict={result.verdict.value}  reason={result.reason}"
    )

# -- 5. Out-of-domain queries ----------------------------------------------
print(f"\n[5] Gating OUT-OF-DOMAIN queries (should abstain):")
# Craft OOD queries: random directions far from corpus
ood_count = 0
for attempt in range(20):
    v = rng.randn(DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    max_sim = float((corpus @ v).max())
    if max_sim < 0.2 and ood_count < 5:
        result = sentinel.gate(v, threshold=0.4)
        icon = "OK" if not result.should_answer else "NO"
        print(
            f"    [{icon}] ood_{ood_count}    confidence={result.confidence:.3f}  "
            f"verdict={result.verdict.value}  max_corpus_sim={max_sim:.3f}"
        )
        ood_count += 1

if ood_count == 0:
    # Lower threshold to demonstrate abstain
    v = rng.randn(DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    result = sentinel.gate(v, threshold=0.99)
    print(f"    [OK] forced_ood  confidence={result.confidence:.3f}  verdict={result.verdict.value}")

# -- 6. Latency benchmark --------------------------------------------------
print(f"\n[6] Latency benchmark (100 queries):")
bench_queries = corpus[rng.choice(N_CORPUS, 100, replace=True)]
stats = sentinel.benchmark(bench_queries)
print(f"    mean={stats['mean_ms']:.3f} ms")
print(f"    p50={stats['p50_ms']:.3f} ms")
print(f"    p95={stats['p95_ms']:.3f} ms")
print(f"    p99={stats['p99_ms']:.3f} ms")

# -- 7. Save / load --------------------------------------------------------
import tempfile, os

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "my_sentinel")
    sentinel.save(path)
    loaded = Sentinel.load(path + ".npz", threshold=0.5)
    result = loaded.gate(corpus[0])
    print(f"\n[7] Save/load verified: {len(loaded)} vectors, gate works: {result.should_answer}")

print("\n" + "=" * 60)
print("Quickstart complete!")
print("=" * 60)
