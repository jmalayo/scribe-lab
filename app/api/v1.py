import time

import requests
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.deps import PipelineDep
from app.schemas import (
    Citation,
    QueryRequest,
    QueryResponse,
    RetrieveRequest,
    RetrieveResponse,
    Usage,
)
from shared.settings import settings

router = APIRouter()


def _citations(chunks: list[dict], score_key: str) -> list[Citation]:

    return [
        Citation(
            chunk_id=c["chunk_id"],
            doc_id=c["doc_id"],
            chunk_index=c.get("chunk_index"),
            score=c.get(score_key),
            text=c["text"],
            tables=[t["text_content"] for t in c.get("tables") or []],
        )
        for c in chunks
    ]


@router.post("/v1/query", response_model=QueryResponse)
def query(req: QueryRequest, pipeline: PipelineDep) -> QueryResponse:

    result = pipeline.run(req.question, top_k=req.top_k)

    return QueryResponse(
        answer=result.answer,
        abstained=result.abstained,
        citations=_citations(result.chunks, pipeline.score_key),
        grounding=result.grounding,
        latency_ms=result.latencies_ms["total"],
        latencies_ms=result.latencies_ms,
        usage=Usage(tokens_in=result.tokens_in, tokens_out=result.tokens_out),
        provider=result.provider,
        model=result.model,
        pipeline_version=pipeline.config.pipeline_version,
        prompt_version=pipeline.config.prompt_version,
    )


@router.post("/v1/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest, pipeline: PipelineDep) -> RetrieveResponse:

    t0 = time.perf_counter()
    result = pipeline.retrieve(req.query, top_k=req.top_k)

    return RetrieveResponse(
        chunks=_citations(result.chunks, pipeline.score_key),
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        latencies_ms=result.latencies_ms,
        pipeline_version=pipeline.config.pipeline_version,
    )


@router.get("/health")
def health() -> dict:

    return {"status": "ok"}


def _http_ok(url: str) -> bool:

    try:
        return requests.get(f"{url}/health", timeout=5).ok
    except requests.RequestException:
        return False


@router.get("/ready")
def ready(pipeline: PipelineDep) -> JSONResponse:

    try:
        collection_ok = pipeline.client.collection_exists(pipeline.config.collection)
    except Exception:
        collection_ok = False

    health = getattr(pipeline.embedder, "health", None)
    checks = {
        "qdrant_collection": collection_ok,
        "embedder": health() if health else True,
        "llm": pipeline.llm.ping(),
    }

    if pipeline.config.grounding.enabled:
        checks["detector"] = _http_ok(settings.detector_url)
        checks["nli"] = _http_ok(settings.nli_url)

    ok = all(checks.values())

    return JSONResponse(status_code=200 if ok else 503, content={"status": "ready" if ok else "not_ready", "checks": checks})
