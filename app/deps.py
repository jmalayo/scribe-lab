from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request

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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:

    app.state.pipeline = build_pipeline()

    yield


def get_pipeline(request: Request) -> Pipeline:

    return request.app.state.pipeline


PipelineDep = Annotated[Pipeline, Depends(get_pipeline)]
