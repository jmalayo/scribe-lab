from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request

from app import ssm_keys
from app.auth import KeyStore
from shared.db import build_writer_engine, load_active_keys
from shared.ingest import build_qdrant_client, get_embedder
from shared.llm import build_llm
from shared.pipeline import Pipeline, PipelineConfig, http_grounder
from shared.settings import settings


def build_pipeline() -> Pipeline:

    config = PipelineConfig.from_yaml(settings.pipeline_config_path)
    llm = build_llm(
        config.llm.provider,
        config.llm.model,
        max_tokens=config.llm.max_tokens,
        base_url=config.llm.base_url,
    )
    grounder = (
        http_grounder(settings.detector_url, settings.nli_url, config.grounding) if config.grounding.enabled else None
    )
    rewriter = None

    if config.memory.enabled:
        rewriter = build_llm(
            config.llm.provider,
            config.memory.model,
            max_tokens=config.memory.max_tokens,
            base_url=config.llm.base_url,
            repeat_penalty=None,
        )

    return Pipeline(config, build_qdrant_client(), get_embedder(), llm, grounder, rewriter)


def build_key_store(engine) -> KeyStore:

    if settings.api_key_store == "ssm":
        return KeyStore(lambda: ssm_keys.load_active_keys(settings.api_keys_parameter))

    return KeyStore(lambda: load_active_keys(engine))


def startup(app: FastAPI) -> None:

    app.state.pipeline = build_pipeline()
    keys_in_db = settings.require_api_key and settings.api_key_store == "db"
    engine = build_writer_engine() if settings.query_log_enabled or keys_in_db else None
    app.state.writer_engine = engine
    app.state.query_log = engine if settings.query_log_enabled else None
    app.state.api_keys = build_key_store(engine) if settings.require_api_key else None


def shutdown(app: FastAPI) -> None:

    if getattr(app.state, "writer_engine", None) is not None:
        app.state.writer_engine.dispose()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:

    startup(app)

    yield

    shutdown(app)


def get_pipeline(request: Request) -> Pipeline:

    return request.app.state.pipeline


PipelineDep = Annotated[Pipeline, Depends(get_pipeline)]
