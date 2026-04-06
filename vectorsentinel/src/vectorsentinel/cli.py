"""CLI entry point for VectorSentinel.

Usage:
    sentinel serve --index my_index.npz --threshold 0.5 --port 8000
    sentinel info --index my_index.npz
    sentinel benchmark --index my_index.npz --queries queries.npy
"""

from __future__ import annotations

import argparse
import sys


def cmd_serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required: pip install vectorsentinel[server]", file=sys.stderr)
        sys.exit(1)

    from vectorsentinel.sentinel import Sentinel
    from vectorsentinel.server.app import create_app

    print(f"Loading index from {args.index} ...")
    sentinel = Sentinel.load(args.index, threshold=args.threshold, k=args.k)
    print(f"  Loaded {len(sentinel)} vectors, dim={sentinel.dim}")
    print(f"  Threshold: {sentinel.threshold}, k: {sentinel.k}")

    app = create_app(sentinel)
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_info(args: argparse.Namespace) -> None:
    from vectorsentinel.sentinel import Sentinel

    sentinel = Sentinel.load(args.index)
    print(f"Index:      {args.index}")
    print(f"Vectors:    {len(sentinel)}")
    print(f"Dim:        {sentinel.dim}")
    print(f"Threshold:  {sentinel.threshold}")
    print(f"k:          {sentinel.k}")
    print(f"Density:    {sentinel._index.mean_density:.4f}")
    print(f"Clusters:   {len(sentinel._index.clusters)}")

    report = sentinel.cluster_report()
    if report:
        avg_purity = sum(c["purity"] for c in report) / len(report)
        print(f"Avg purity: {avg_purity:.4f}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    import numpy as np

    from vectorsentinel.sentinel import Sentinel

    sentinel = Sentinel.load(args.index)
    queries = np.load(args.queries).astype(np.float32)
    print(f"Benchmarking {len(queries)} queries on index of {len(sentinel)} vectors ...")
    stats = sentinel.benchmark(queries)
    for key, val in stats.items():
        print(f"  {key}: {val}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sentinel",
        description="VectorSentinel — confidence gating for RAG pipelines",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start REST API server")
    serve_parser.add_argument("--index", required=True, help="Path to saved index (.npz)")
    serve_parser.add_argument("--threshold", type=float, default=0.5)
    serve_parser.add_argument("--k", type=int, default=5)
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)

    # info
    info_parser = subparsers.add_parser("info", help="Print index statistics")
    info_parser.add_argument("--index", required=True)

    # benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Latency benchmark")
    bench_parser.add_argument("--index", required=True)
    bench_parser.add_argument("--queries", required=True, help="Path to .npy query matrix")

    args = parser.parse_args()

    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
