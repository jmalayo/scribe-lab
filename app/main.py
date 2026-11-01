import time
import uuid

import requests
import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from qdrant_client.http.exceptions import ResponseHandlingException
from starlette.background import BackgroundTask

from app import query_log
from app.api.v1 import router
from app.deps import lifespan
from app.log import configure_logging
from app.telemetry import configure_tracing
from shared.llm import LLMUnavailableError

configure_logging()
configure_tracing()
logger = structlog.get_logger("app")

app = FastAPI(title="scribe-lab API", version="0.1.0", lifespan=lifespan)
app.include_router(router)


@app.middleware("http")
async def request_context(request: Request, call_next):

    request.state.request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request.state.request_id)
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    response.headers["x-request-id"] = request.state.request_id
    logger.info(
        "request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        latency_ms=elapsed,
        api_key=getattr(request.state, "api_key_owner", None),
    )

    return response


def failed_query_task(request: Request, status: int, exc: Exception) -> BackgroundTask | None:

    record = getattr(request.state, "query_record", None)

    if record is None:
        return None

    query_log.finish_failure(record, status, exc)

    return BackgroundTask(query_log.save, getattr(request.app.state, "query_log", None), record)


@app.exception_handler(LLMUnavailableError)
async def llm_unavailable(request: Request, exc: LLMUnavailableError) -> JSONResponse:

    logger.info("llm_unavailable")

    return JSONResponse(
        status_code=503,
        content={"detail": f"{exc} /v1/query no está disponible; /v1/retrieve sí funciona."},
        background=failed_query_task(request, 503, exc),
    )


@app.exception_handler(requests.HTTPError)
async def upstream_http_error(request: Request, exc: requests.HTTPError) -> JSONResponse:

    status = getattr(exc.response, "status_code", None)
    logger.warning("upstream_error", upstream_status=status)

    return JSONResponse(
        status_code=502,
        content={"detail": "Un servicio interno respondió con error."},
        background=failed_query_task(request, 502, exc),
    )


@app.exception_handler(ResponseHandlingException)
@app.exception_handler(requests.RequestException)
async def upstream_unavailable(request: Request, exc: Exception) -> JSONResponse:

    logger.warning("upstream_unavailable", error=type(exc).__name__)

    return JSONResponse(
        status_code=503,
        content={"detail": "Un servicio interno no está disponible."},
        background=failed_query_task(request, 503, exc),
    )


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:

    logger.exception("unhandled_error")

    return JSONResponse(
        status_code=500,
        content={"detail": "Error interno."},
        background=failed_query_task(request, 500, exc),
    )
