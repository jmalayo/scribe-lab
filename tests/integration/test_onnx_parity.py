from pathlib import Path

import numpy as np
import pytest

from shared.eval.data import load_questions
from shared.ingest import (
    RemoteEmbedder,
    chunk_documents,
    get_tables_from_docs,
    load_corpus,
)
from shared.pipeline import PipelineConfig
from shared.settings import settings

MIN_COSINE = 0.999


@pytest.fixture(scope="module")
def tei():

    embedder = RemoteEmbedder(settings.text_embedder_url, f"sentence-transformers/{settings.embedding_model}")

    if not embedder.health():
        pytest.skip(f"TEI no responde en {settings.text_embedder_url}")

    return embedder


def test_onnx_fp32_matches_tei_on_chunks_and_questions(tei):

    pytest.importorskip("onnxruntime")
    from shared.ingest import OnnxEmbedder

    if not (Path(settings.onnx_model_dir) / "onnx" / "model.onnx").exists():
        pytest.skip(f"Falta el modelo ONNX en {settings.onnx_model_dir}")

    config = PipelineConfig.from_yaml(settings.pipeline_config_path)
    docs = load_corpus(config.corpus_dir)
    tables = [t for doc in docs for t in get_tables_from_docs(doc)[0]]
    texts = [c.text for c in chunk_documents(docs, tables, config.chunk_size, config.chunk_overlap)]
    texts += [q["question"] for q in load_questions()]

    cosines = (OnnxEmbedder(settings.onnx_model_dir).encode(texts) * tei.encode(texts)).sum(axis=1)

    assert len(texts) > 500
    assert float(np.min(cosines)) >= MIN_COSINE
