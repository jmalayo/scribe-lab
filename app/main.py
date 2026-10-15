import logging
import sys
import time
import uuid

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.v1 import router
from app.deps import lifespan

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("app")

app = FastAPI(title="scribe-lab API", version="0.1.0", lifespan=lifespan)
app.include_router(router)


@app.middleware("http")
async def request_context(request: Request, call_next):

    request.state.request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    response.headers["x-request-id"] = request.state.request_id
    logger.info(
        "request_id=%s method=%s path=%s status=%s latency_ms=%s",
        request.state.request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )

    return response


@app.exception_handler(requests.HTTPError)
async def upstream_http_error(request: Request, exc: requests.HTTPError) -> JSONResponse:

    status = getattr(exc.response, "status_code", None)
    logger.warning("request_id=%s upstream_status=%s", getattr(request.state, "request_id", None), status)

    return JSONResponse(status_code=502, content={"detail": "Un servicio interno respondió con error."})


@app.exception_handler(requests.RequestException)
async def upstream_unavailable(request: Request, exc: requests.RequestException) -> JSONResponse:

    logger.warning("request_id=%s upstream_unavailable=%s", getattr(request.state, "request_id", None), type(exc).__name__)

    return JSONResponse(status_code=503, content={"detail": "Un servicio interno no está disponible."})


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:

    logger.exception("request_id=%s unhandled", getattr(request.state, "request_id", None))

    return JSONResponse(status_code=500, content={"detail": "Error interno."})
