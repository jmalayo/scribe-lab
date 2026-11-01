import uuid

import pytest
from fastapi.testclient import TestClient

from app.auth import CACHE_SECONDS, KeyStore, hash_key
from app.deps import get_pipeline
from app.main import app
from tests.unit.test_pipeline import make_pipeline

KEY = "sk_scribe_" + "a" * 43
KEY_ID = uuid.uuid4()


class Clock:

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def keys(monkeypatch):

    rows = [(KEY_ID, "mcp-local", hash_key(KEY), 2)]
    clock = Clock()
    store = KeyStore(lambda: list(rows), clock=clock)

    monkeypatch.setattr(app.state, "api_keys", store, raising=False)
    app.dependency_overrides[get_pipeline] = lambda: make_pipeline()

    yield TestClient(app, raise_server_exceptions=False), rows, clock

    app.dependency_overrides.clear()


def retrieve(client, key=None):

    return client.post(
        "/v1/retrieve",
        json={"query": "q"},
        headers={"X-API-Key": key} if key else {},
    )


def test_missing_or_wrong_key_is_rejected(keys):

    client, _, _ = keys

    assert retrieve(client).status_code == 401
    assert retrieve(client, "sk_scribe_otra").status_code == 401
    assert client.post("/v1/query", json={"question": "q"}).status_code == 401


def test_valid_key_passes_and_health_stays_open(keys):

    client, _, _ = keys

    assert retrieve(client, KEY).status_code == 200
    assert client.get("/health").status_code == 200


def test_revoked_key_is_rejected_after_the_cache_expires(keys):

    client, rows, clock = keys

    assert retrieve(client, KEY).status_code == 200

    rows.clear()
    clock.now += CACHE_SECONDS

    assert retrieve(client, KEY).status_code == 401


def test_limit_per_minute_returns_429_until_the_next_window(keys):

    client, _, clock = keys

    assert [retrieve(client, KEY).status_code for _ in range(3)] == [200, 200, 429]
    assert retrieve(client, KEY).headers["retry-after"] == "60"

    clock.now += 60

    assert retrieve(client, KEY).status_code == 200


def test_key_store_failure_returns_503(keys, monkeypatch):

    client, _, _ = keys

    def broken():
        raise ConnectionRefusedError()

    monkeypatch.setattr(app.state, "api_keys", KeyStore(broken), raising=False)

    assert retrieve(client, KEY).status_code == 503


def test_query_records_the_key_that_made_it(keys, monkeypatch):

    from app import query_log

    client, _, _ = keys
    saved = []
    monkeypatch.setattr(app.state, "query_log", "engine", raising=False)
    monkeypatch.setattr(query_log, "insert_query", lambda engine, row, chunks: saved.append(row))

    client.post("/v1/query", json={"question": "q"}, headers={"X-API-Key": KEY})

    assert saved[0]["api_key_id"] == KEY_ID
