import hashlib
import re
import uuid

import numpy as np
import pytest
from alembic import command
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from sqlalchemy import text

from app.auth import KeyStore
from app.deps import get_pipeline
from app.main import app
from app.manage_keys import create
from mcp_server import tools
from shared import ingest
from shared.db import build_writer_engine, load_active_keys
from shared.ingest import Chunk, sync_index
from shared.pipeline import Pipeline
from shared.settings import settings
from tests.unit.test_pipeline import FakeLLM, make_config

DIM = 64

CHUNKS = [
    Chunk("c-forest", "music-tagger/benchmark.md", "music-tagger", "Random Forest obtuvo 48.9 de balanced accuracy", 0),
    Chunk("c-hpss", "music-tagger/genesis.md", "music-tagger", "HPSS separa la parte armónica de la percusiva", 0),
    Chunk("c-fetch", "music-tagger/fetch.md", "music-tagger", "El motor descarga audio de forma asíncrona", 0),
]


class HashEmbedder:

    def get_dimension(self) -> int:
        return DIM

    def encode(self, texts, normalize_embeddings=False):

        single = isinstance(texts, str)
        vectors = np.zeros((1 if single else len(texts), DIM), dtype=np.float32)

        for row, sentence in enumerate([texts] if single else texts):
            for word in re.findall(r"\w+", sentence.lower()):
                vectors[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0

        vectors /= np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)

        return vectors[0] if single else vectors


@pytest.fixture
def qdrant(monkeypatch):

    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=5)

    try:
        client.get_collections()
    except Exception:
        pytest.skip(f"Qdrant no responde en {settings.qdrant_host}:{settings.qdrant_port}")

    alias = f"it_{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(ingest, "get_embedder", lambda: HashEmbedder())

    yield client, alias

    for collection in client.get_collections().collections:
        if collection.name.startswith(f"{alias}__"):
            client.delete_collection(collection.name)


@pytest.fixture
def indexed(qdrant):

    client, alias = qdrant
    result = sync_index(client, CHUNKS, alias=alias, config={"embedding_model": "hash-64"})
    pipeline = Pipeline(make_config(collection=alias, top_k=2), client, HashEmbedder(), FakeLLM(), None, None)

    return pipeline, result


@pytest.fixture
def service(db, indexed, monkeypatch):

    config, engine = db
    command.upgrade(config, "servicio@head")

    with engine.begin() as conn:
        key = create(conn, "integracion", 60)

    writer = build_writer_engine(engine.url.render_as_string(hide_password=False))
    pipeline, _ = indexed

    monkeypatch.setattr(app.state, "api_keys", KeyStore(lambda: load_active_keys(writer)), raising=False)
    monkeypatch.setattr(app.state, "query_log", writer, raising=False)
    app.dependency_overrides[get_pipeline] = lambda: pipeline

    yield TestClient(app), key, engine

    app.dependency_overrides.clear()
    writer.dispose()


def test_indexing_is_searchable_and_idempotent(indexed, qdrant):

    pipeline, first = indexed
    client, alias = qdrant
    second = sync_index(client, CHUNKS, alias=alias, config={"embedding_model": "hash-64"})

    assert (first["rebuilt"], first["added"]) == (True, 3)
    assert (second["rebuilt"], second["added"], second["deleted"]) == (False, 0, 0)
    assert pipeline.retrieve("¿qué balanced accuracy obtuvo Random Forest?").chunks[0]["doc_id"] == "music-tagger/benchmark.md"


def test_api_requires_the_key_and_records_the_query(service):

    client, key, engine = service

    assert client.post("/v1/query", json={"question": "Random Forest"}).status_code == 401

    resp = client.post(
        "/v1/query",
        json={"question": "¿Qué separa HPSS?", "session_id": str(uuid.uuid4())},
        headers={"X-API-Key": key},
    )

    assert resp.status_code == 200
    assert resp.json()["citations"][0]["doc_id"] == "music-tagger/genesis.md"

    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT c.estado, c.turno, k.dueno, (SELECT count(*) FROM consulta_chunks) "
                "FROM consultas c JOIN api_keys k ON k.id = c.api_key_id"
            )
        ).one()

    assert tuple(row) == (200, 1, "integracion", 2)


def test_mcp_tool_reaches_the_api_with_its_key(service, monkeypatch):

    client, key, _ = service

    def post(url, json, timeout, headers=None):
        return client.post(url.removeprefix(settings.scribe_api_url), json=json, headers=headers)

    monkeypatch.setattr(tools.httpx, "post", post)
    monkeypatch.setattr(tools.settings, "scribe_api_key", key)

    found = tools.buscar_documentos("descarga de audio asíncrona", top_k=1)

    assert [c["doc_id"] for c in found] == ["music-tagger/fetch.md"]

    monkeypatch.setattr(tools.settings, "scribe_api_key", "sk_scribe_invalida")

    with pytest.raises(RuntimeError, match="401"):
        tools.buscar_documentos("audio")
