"""FastAPI REST server for VectorSentinel.

Enables non-Python services (Node, Go, Ruby, etc.) to use VectorSentinel
via HTTP.

Usage:
    sentinel serve --index my_index.npz --threshold 0.5 --port 8000

Or programmatically:
    from vectorsentinel.server.app import create_app
    import uvicorn

    app = create_app(sentinel)
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vectorsentinel.sentinel import Sentinel


def create_app(sentinel: Sentinel) -> Any:
    """Create a FastAPI app wrapping the given Sentinel instance."""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError:
        raise ImportError(
            "fastapi and pydantic are required for the server. "
            "Install with: pip install vectorsentinel[server]"
        )

    app = FastAPI(
        title="VectorSentinel",
        description="Confidence gating for RAG pipelines",
        version="0.1.0",
    )

    class GateRequest(BaseModel):
        embedding: list[float]
        threshold: float | None = None
        k: int | None = None

    class BatchGateRequest(BaseModel):
        embeddings: list[list[float]]
        threshold: float | None = None
        k: int | None = None

    class AddRequest(BaseModel):
        embeddings: list[list[float]]
        ids: list[str] | None = None
        labels: list[str] | None = None

    class DeleteRequest(BaseModel):
        ids: list[str]

    @app.get("/health")
    def health():
        return {"status": "ok", "index_size": len(sentinel)}

    @app.post("/gate")
    def gate(req: GateRequest):
        q = np.array(req.embedding, dtype=np.float32)
        if len(q) != sentinel.dim:
            raise HTTPException(
                status_code=400,
                detail=f"Expected embedding dim {sentinel.dim}, got {len(q)}",
            )
        result = sentinel.gate(q, threshold=req.threshold, k=req.k)
        return result.to_dict()

    @app.post("/gate/batch")
    def gate_batch(req: BatchGateRequest):
        queries = np.array(req.embeddings, dtype=np.float32)
        results = sentinel.gate_batch(queries, threshold=req.threshold, k=req.k)
        return [r.to_dict() for r in results]

    @app.post("/add")
    def add(req: AddRequest):
        embeddings = np.array(req.embeddings, dtype=np.float32)
        sentinel.add(embeddings, ids=req.ids, labels=req.labels)
        return {"added": len(req.embeddings), "index_size": len(sentinel)}

    @app.post("/delete")
    def delete(req: DeleteRequest):
        removed = sentinel.delete(req.ids)
        return {"removed": removed, "index_size": len(sentinel)}

    @app.post("/refine")
    def refine():
        reassigned = sentinel.refine()
        return {"reassigned": reassigned}

    @app.get("/clusters")
    def clusters():
        return sentinel.cluster_report()

    @app.get("/info")
    def info():
        return {
            "dim": sentinel.dim,
            "size": len(sentinel),
            "threshold": sentinel.threshold,
            "k": sentinel.k,
            "mean_density": round(sentinel._index.mean_density, 4),
            "n_clusters": len(sentinel._index.clusters),
        }

    return app
