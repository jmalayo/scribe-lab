
import pytest
import requests
from fastapi.testclient import TestClient

from app.deps import get_pipeline
from app.main import app
from tests.unit.test_pipeline import FakeLLM, make_chunk, make_pipeline


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
        "tables": [],
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


def test_retrieve_returns_the_tables_of_each_chunk(client_for):

    chunk = make_chunk(0, 0.9) | {"tables": [{"text_content": "| a | b |"}]}
    resp = client_for(make_pipeline(chunks=[chunk])).post("/v1/retrieve", json={"query": "q", "top_k": 1})

    assert resp.json()["chunks"][0]["tables"] == ["| a | b |"]


def test_request_id_is_propagated(client_for):

    resp = client_for(make_pipeline()).post("/v1/query", json={"question": "q"}, headers={"x-request-id": "abc"})

    assert resp.headers["x-request-id"] == "abc"


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


@pytest.fixture
def log_lines():

    import io
    import json

    from app.log import configure_logging

    buffer = io.StringIO()
    configure_logging(stream=buffer)

    yield lambda: [json.loads(line) for line in buffer.getvalue().splitlines()]

    configure_logging()


def test_each_request_logs_one_json_line(client_for, log_lines):

    resp = client_for(make_pipeline()).get("/health", headers={"x-request-id": "rid-1"})
    lines = [line for line in log_lines() if line["event"] == "request"]

    assert resp.status_code == 200
    assert lines == [
        {
            "event": "request",
            "request_id": "rid-1",
            "method": "GET",
            "path": "/health",
            "status": 200,
            "latency_ms": lines[0]["latency_ms"],
            "api_key": None,
            "level": "info",
            "logger": "app",
            "timestamp": lines[0]["timestamp"],
        }
    ]


def test_query_event_has_metrics_but_not_the_question(client_for, log_lines):

    question = "¿pregunta privada del usuario?"
    client_for(make_pipeline()).post(
        "/v1/query",
        json={"question": question},
        headers={"x-request-id": "rid-2"},
    )
    lines = log_lines()
    event = next(line for line in lines if line["event"] == "query")

    assert event["request_id"] == "rid-2"
    assert (event["tokens_in"], event["tokens_out"]) == (11, 3)
    assert event["chunk_ids"] == ["c0", "c1", "c2"]
    assert event["latencies_ms"]["total"] >= 0
    assert (event["rewritten"], event["abstained"], event["history_turns"]) == (False, False, 0)
    assert (event["pipeline_version"], event["prompt_version"]) == ("test", "v1")
    assert all(question not in json_line for json_line in map(str, lines))


def test_unhandled_error_is_logged_as_json(client_for, log_lines):

    client_for(make_pipeline(llm=RaisingLLM(RuntimeError("boom")))).post("/v1/query", json={"question": "q"})
    error = next(line for line in log_lines() if line["event"] == "unhandled_error")

    assert error["level"] == "error"
    assert "RuntimeError: boom" in error["exception"]


@pytest.fixture
def saved_queries(monkeypatch):

    from app import query_log

    saved = []
    monkeypatch.setattr(app.state, "query_log", "engine", raising=False)
    monkeypatch.setattr(query_log, "insert_query", lambda engine, row, chunks: saved.append((engine, row, chunks)))

    return saved


def test_successful_query_is_recorded_with_its_chunks(client_for, saved_queries):

    session = "6f1d3c1e-7a51-4b43-9a55-0c5a7e8f9b10"
    client_for(make_pipeline()).post(
        "/v1/query",
        json={
            "question": "¿qué pasó?",
            "session_id": session,
            "history": [{"question": "antes", "answer": "algo"}],
        },
        headers={"x-request-id": "rid-3"},
    )
    [(engine, row, chunks)] = saved_queries

    assert engine == "engine"
    assert (row["request_id"], row["estado"], row["pregunta"], row["respuesta"]) == ("rid-3", 200, "¿qué pasó?", "respuesta")
    assert (str(row["sesion_id"]), row["turno"]) == (session, 2)
    assert (row["tokens_entrada"], row["tokens_salida"], row["abstencion"]) == (11, 3, False)
    assert row["latencia_total_ms"] >= 0
    assert [(c["posicion"], c["chunk_id"], c["consulta_id"]) for c in chunks] == [
        (1, "c0", row["id"]),
        (2, "c1", row["id"]),
        (3, "c2", row["id"]),
    ]


def test_failed_query_is_recorded_with_its_status(client_for, saved_queries):

    resp = client_for(make_pipeline(llm=RaisingLLM(requests.ConnectionError()))).post("/v1/query", json={"question": "q"})
    [(_, row, chunks)] = saved_queries

    assert resp.status_code == 503
    assert (row["estado"], row["error"], row.get("respuesta"), row["turno"]) == (503, "ConnectionError", None, None)
    assert row["latencia_total_ms"] >= 0
    assert chunks == []


def test_query_still_answers_when_the_database_fails(client_for, monkeypatch, log_lines):

    from app import query_log

    def broken(engine, row, chunks):
        raise ConnectionRefusedError()

    monkeypatch.setattr(app.state, "query_log", "engine", raising=False)
    monkeypatch.setattr(query_log, "insert_query", broken)
    resp = client_for(make_pipeline()).post("/v1/query", json={"question": "q"})
    warning = next(line for line in log_lines() if line["event"] == "query_log_failed")

    assert resp.status_code == 200
    assert (warning["level"], warning["error"]) == ("warning", "ConnectionRefusedError")


def test_retrieve_is_not_recorded(client_for, saved_queries):

    client_for(make_pipeline()).post("/v1/retrieve", json={"query": "q"})

    assert saved_queries == []


def test_without_llm_query_is_503_and_retrieve_still_works(client_for, saved_queries):

    from shared.llm import NoLLM

    client = client_for(make_pipeline(llm=NoLLM()))
    query = client.post("/v1/query", json={"question": "q"})
    retrieve = client.post("/v1/retrieve", json={"query": "q"})

    assert query.status_code == 503
    assert "/v1/retrieve sí funciona" in query.json()["detail"]
    assert retrieve.status_code == 200
    assert [(row["estado"], row["error"]) for _, row, _ in saved_queries] == [(503, "LLMUnavailableError")]


def test_qdrant_unreachable_is_503(client_for, saved_queries):

    from qdrant_client.http.exceptions import ResponseHandlingException

    pipeline = make_pipeline()

    def unreachable(*args, **kwargs):
        raise ResponseHandlingException(ConnectionError("qdrant"))

    pipeline.client.query_points = unreachable
    resp = client_for(pipeline).post("/v1/query", json={"question": "q"})

    assert resp.status_code == 503
    assert [(row["estado"], row["error"]) for _, row, _ in saved_queries] == [(503, "ResponseHandlingException")]

