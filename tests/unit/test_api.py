
import pytest
import requests
from fastapi.testclient import TestClient

from app.deps import get_pipeline
from app.main import app
from tests.unit.test_pipeline import FakeLLM, make_pipeline


class AliasClient:

    def __init__(self, inner, aliases):
        self.inner = inner
        self.aliases = aliases

    def query_points(self, *args, **kwargs):
        return self.inner.query_points(*args, **kwargs)

    def collection_exists(self, name):
        return name in self.aliases


class RaisingLLM(FakeLLM):

    def __init__(self, exc):
        super().__init__()
        self.exc = exc

    def complete(self, prompt):
        raise self.exc


@pytest.fixture
def client_for():

    def build(pipeline):
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        return TestClient(app, raise_server_exceptions=False)

    yield build
    app.dependency_overrides.clear()


def test_health(client_for):

    assert client_for(make_pipeline()).get("/health").json() == {"status": "ok"}


def test_query_returns_contract(client_for):

    resp = client_for(make_pipeline()).post("/v1/query", json={"question": "¿qué pasó?"})
    body = resp.json()

    assert resp.status_code == 200
    assert resp.headers["x-request-id"]
    assert body["answer"] == "respuesta"
    assert body["abstained"] is False
    assert len(body["citations"]) == 3
    assert body["citations"][0] == {
        "chunk_id": "c0",
        "doc_id": "music-tagger/doc.md",
        "chunk_index": 0,
        "score": pytest.approx(0.9),
        "text": "texto 0",
    }
    assert body["grounding"] is None
    assert body["usage"] == {"tokens_in": 11, "tokens_out": 3}
    assert body["latency_ms"] == body["latencies_ms"]["total"]
    assert (body["pipeline_version"], body["prompt_version"]) == ("test", "v1")


def test_retrieve_returns_chunks_without_calling_the_llm(client_for):

    llm = FakeLLM()
    resp = client_for(make_pipeline(llm=llm)).post("/v1/retrieve", json={"query": "q", "top_k": 2})

    assert resp.status_code == 200
    assert [c["chunk_id"] for c in resp.json()["chunks"]] == ["c0", "c1"]
    assert llm.prompts == []


def test_request_id_is_propagated(client_for):

    resp = client_for(make_pipeline()).post("/v1/query", json={"question": "q"}, headers={"x-request-id": "abc"})

    assert resp.headers["x-request-id"] == "abc"


def test_unknown_collection_is_not_found(client_for):

    resp = client_for(make_pipeline()).post("/v1/query", json={"question": "q", "collection": "otra"})

    assert resp.status_code == 404
    assert resp.json()["detail"] == "colección desconocida: otra"


def test_empty_question_is_a_validation_error(client_for):

    resp = client_for(make_pipeline()).post("/v1/query", json={"question": ""})

    assert resp.status_code == 422
    assert resp.json()["detail"][0]["loc"] == ["body", "question"]


def test_llm_unavailable_maps_to_503(client_for):

    resp = client_for(make_pipeline(llm=RaisingLLM(requests.ConnectionError()))).post("/v1/query", json={"question": "q"})

    assert resp.status_code == 503
    assert resp.json()["detail"] == "Un servicio interno no está disponible."


def test_ready_reports_each_dependency(client_for):

    pipeline = make_pipeline()
    pipeline.client = AliasClient(pipeline.client, ["corpus_demo"])

    resp = client_for(pipeline).get("/ready")

    assert resp.status_code == 200
    assert resp.json()["checks"] == {"qdrant_collection": True, "embedder": True, "llm": True}


def test_ready_fails_without_the_collection_alias(client_for):

    pipeline = make_pipeline()
    pipeline.client = AliasClient(pipeline.client, [])

    resp = client_for(pipeline).get("/ready")

    assert resp.status_code == 503
    assert resp.json()["checks"]["qdrant_collection"] is False
