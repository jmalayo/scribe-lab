import time
import uuid

import structlog
from fastapi import Request

from app.schemas import QueryRequest
from shared.db import insert_query
from shared.pipeline import Pipeline, PipelineResult

logger = structlog.get_logger("app.query_log")

STAGE_COLUMNS = {
    "total": "latencia_total_ms",
    "embedding": "latencia_embedding_ms",
    "retrieval": "latencia_busqueda_ms",
    "rerank": "latencia_rerank_ms",
    "llm": "latencia_llm_ms",
    "grounding": "latencia_grounding_ms",
}


def start_record(request: Request, req: QueryRequest, pipeline: Pipeline) -> dict:

    record = {
        "row": {
            "id": uuid.uuid4(),
            "request_id": request.state.request_id,
            "api_key_id": getattr(request.state, "api_key_id", None),
            "sesion_id": req.session_id,
            "turno": len(req.history) + 1 if req.session_id else None,
            "pregunta": req.question,
            "pipeline_version": pipeline.config.pipeline_version,
            "prompt_version": pipeline.config.prompt_version,
        },
        "chunks": [],
        "t0": time.perf_counter(),
    }
    request.state.query_record = record

    return record


def finish_success(record: dict, result: PipelineResult, score_key: str) -> None:

    record["row"].update(
        {
            "estado": 200,
            "pregunta_reescrita": result.rewritten_question,
            "respuesta": result.answer,
            "abstencion": result.abstained,
            "grounding_score": result.grounding.score if result.grounding else None,
            "oraciones_sin_respaldo": len(result.grounding.unsupported_sentences) if result.grounding else None,
            "tokens_entrada": result.tokens_in,
            "tokens_salida": result.tokens_out,
            "proveedor": result.provider,
            "modelo": result.model,
            **{
                column: result.latencies_ms.get(stage)
                    for stage, column in STAGE_COLUMNS.items()
            },
        }
    )
    record["chunks"] = [
        {
            "consulta_id": record["row"]["id"],
            "posicion": position,
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "score": chunk.get(score_key),
        }
            for position, chunk in enumerate(result.chunks, start=1)
    ]


def finish_failure(record: dict, status: int, exc: Exception) -> None:

    record["row"].update(
        {
            "estado": status,
            "error": type(exc).__name__,
            "latencia_total_ms": round((time.perf_counter() - record["t0"]) * 1000, 2),
        }
    )


def save(engine, record: dict) -> None:

    if engine is None:
        return

    try:
        insert_query(engine, record["row"], record["chunks"])
    except Exception as exc:
        logger.warning("query_log_failed", error=type(exc).__name__)
