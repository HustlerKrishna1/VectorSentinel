"""Synthetic hallucination benchmark.

Measures:
  - Accuracy (correct abstain on OOD, correct answer on in-domain)
  - Hallucination rate (OOD queries that slip through as "answer")
  - Abstain rate on in-domain (false rejections)
  - Latency at various corpus sizes

Run:
    python benchmarks/synthetic_benchmark.py

Results on a typical laptop:
    Corpus: 1000 vectors  |  threshold: 0.45
    In-domain accuracy (answer rate):  ~95%+
    OOD abstain rate:                  ~85%+
    Mean latency:                      < 5ms
"""

import time

import numpy as np

from vectorsentinel import Sentinel


def make_corpus(n_per_class: int, n_classes: int, dim: int, noise: float = 0.1, seed: int = 0):
    rng = np.random.RandomState(seed)
    emb_parts, labels, centers = [], [], []
    for c in range(n_classes):
        center = rng.randn(dim).astype(np.float32)
        center /= np.linalg.norm(center)
        noise_vecs = rng.randn(n_per_class, dim).astype(np.float32) * (noise / dim ** 0.5)
        vecs = center + noise_vecs
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        emb_parts.append(vecs)
        labels.extend([f"class_{c}"] * n_per_class)
        centers.append(center)
    return np.vstack(emb_parts), labels, np.stack(centers)


def make_in_domain_queries(centers: np.ndarray, n: int, noise: float = 0.20, seed: int = 1):
    rng = np.random.RandomState(seed)
    queries = []
    for _ in range(n):
        c = centers[rng.randint(len(centers))]
        dim = len(c)
        v = c + rng.randn(dim).astype(np.float32) * (noise / dim ** 0.5)
        v /= np.linalg.norm(v)
        queries.append(v)
    return np.stack(queries)


def make_ood_queries(corpus: np.ndarray, n: int, seed: int = 2):
    rng = np.random.RandomState(seed)
    dim = corpus.shape[1]
    queries = []
    for _ in range(n * 10):
        v = rng.randn(dim).astype(np.float32)
        v /= np.linalg.norm(v)
        if float((corpus @ v).max()) < 0.15:
            queries.append(v)
        if len(queries) >= n:
            break
    # Pad if needed
    while len(queries) < n:
        v = rng.randn(dim).astype(np.float32) * 0.1 - corpus.mean(axis=0)
        v /= np.linalg.norm(v)
        queries.append(v)
    return np.stack(queries[:n])


def run_benchmark(
    n_per_class: int = 200,
    n_classes: int = 5,
    dim: int = 256,
    threshold: float = 0.45,
    k: int = 7,
    n_eval: int = 200,
):
    print(f"\n{'='*60}")
    print(f"Benchmark: {n_per_class*n_classes} vectors | dim={dim} | "
          f"threshold={threshold} | k={k}")
    print(f"{'='*60}")

    # Build corpus
    corpus, labels, centers = make_corpus(n_per_class, n_classes, dim)
    print(f"Corpus size: {len(corpus)}")

    # Build sentinel
    t0 = time.perf_counter()
    sentinel = Sentinel.from_embeddings(corpus, labels=labels, threshold=threshold, k=k)
    build_ms = (time.perf_counter() - t0) * 1000
    print(f"Index built in {build_ms:.1f} ms")

    cluster_report = sentinel.cluster_report()
    if cluster_report:
        avg_purity = sum(c["purity"] for c in cluster_report) / len(cluster_report)
        print(f"Clusters: {len(cluster_report)}, avg purity: {avg_purity:.3f}")

    # Evaluate in-domain queries
    in_domain = make_in_domain_queries(centers, n_eval)
    ood = make_ood_queries(corpus, n_eval)

    # In-domain: should answer
    t0 = time.perf_counter()
    in_results = sentinel.gate_batch(in_domain, threshold=threshold)
    in_latency_ms = (time.perf_counter() - t0) * 1000
    in_answer_rate = sum(1 for r in in_results if r.should_answer) / len(in_results)
    in_false_reject = 1 - in_answer_rate

    # OOD: should abstain
    t0 = time.perf_counter()
    ood_results = sentinel.gate_batch(ood, threshold=threshold)
    ood_latency_ms = (time.perf_counter() - t0) * 1000
    ood_abstain_rate = sum(1 for r in ood_results if not r.should_answer) / len(ood_results)
    hallucination_rate = 1 - ood_abstain_rate

    print(f"\nResults ({n_eval} queries each):")
    print(f"  In-domain  -> answer rate:    {in_answer_rate*100:5.1f}%  "
          f"(false rejections: {in_false_reject*100:.1f}%)")
    print(f"  OOD        -> abstain rate:   {ood_abstain_rate*100:5.1f}%  "
          f"(hallucination risk: {hallucination_rate*100:.1f}%)")

    stats = sentinel.benchmark(in_domain[:50])
    print(f"\nLatency (50 in-domain queries):")
    print(f"  mean={stats['mean_ms']:.3f} ms  p50={stats['p50_ms']:.3f} ms  "
          f"p95={stats['p95_ms']:.3f} ms  p99={stats['p99_ms']:.3f} ms")

    return {
        "in_answer_rate": in_answer_rate,
        "ood_abstain_rate": ood_abstain_rate,
        "hallucination_rate": hallucination_rate,
        "build_ms": build_ms,
        "mean_latency_ms": stats["mean_ms"],
    }


def threshold_sweep():
    """Show how threshold controls the precision/recall tradeoff."""
    print(f"\n{'='*60}")
    print("Threshold sweep (corpus=1000, dim=128)")
    print(f"{'='*60}")
    print(f"{'Threshold':>10}  {'Answer (ID)':>12}  {'Abstain (OOD)':>14}  {'Hallucination':>14}")
    print(f"{'-'*60}")

    corpus, labels, centers = make_corpus(200, 5, 128, noise=0.1)
    sentinel = Sentinel.from_embeddings(corpus, labels=labels, k=7)

    in_domain = make_in_domain_queries(centers, 200, seed=10)
    ood = make_ood_queries(corpus, 200, seed=11)

    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        in_res = sentinel.gate_batch(in_domain, threshold=t)
        ood_res = sentinel.gate_batch(ood, threshold=t)
        in_rate = sum(1 for r in in_res if r.should_answer) / len(in_res)
        ood_rate = sum(1 for r in ood_res if not r.should_answer) / len(ood_res)
        hall_rate = 1 - ood_rate
        print(f"{t:>10.1f}  {in_rate*100:>11.1f}%  {ood_rate*100:>13.1f}%  {hall_rate*100:>13.1f}%")


def scale_benchmark():
    """Show latency at various corpus sizes."""
    print(f"\n{'='*60}")
    print("Scale benchmark (dim=128, threshold=0.45)")
    print(f"{'='*60}")
    print(f"{'Corpus':>8}  {'Build(ms)':>10}  {'Mean(ms)':>9}  {'p95(ms)':>8}")
    print(f"{'-'*45}")

    rng = np.random.RandomState(7)
    for n in [100, 500, 1000, 5000, 10000]:
        emb = rng.randn(n, 128).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)

        t0 = time.perf_counter()
        sentinel = Sentinel.from_embeddings(emb, threshold=0.45, k=7)
        build_ms = (time.perf_counter() - t0) * 1000

        queries = emb[rng.choice(n, min(50, n), replace=False)]
        stats = sentinel.benchmark(queries)
        print(f"{n:>8,}  {build_ms:>10.1f}  {stats['mean_ms']:>9.3f}  {stats['p95_ms']:>8.3f}")


if __name__ == "__main__":
    # Main benchmark
    run_benchmark(n_per_class=200, n_classes=5, dim=256, threshold=0.45)
    run_benchmark(n_per_class=100, n_classes=10, dim=768, threshold=0.50)

    # Threshold sweep
    threshold_sweep()

    # Scale benchmark
    scale_benchmark()
