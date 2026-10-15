from types import SimpleNamespace

import numpy as np
import pytest
import requests

from shared import ingest
from shared.ingest import Chunk, RemoteEmbedder, content_chunk_ids


def chunk(doc_id: str, i: int, text: str, tables: list[str] | None = None) -> Chunk:

    return Chunk(
        chunk_id=f"pos-{i}",
        doc_id=doc_id,
        library="lib",
        text=text,
        chunk_index=i,
        tables=[
            {
                "text_content": t, 
                "doc_id": doc_id, 
                "type": "Table"
            } for t in tables or []
        ],
    )


def test_content_ids_survive_a_shift_in_position():

    before = content_chunk_ids([chunk("d.md", 0, "a"), chunk("d.md", 1, "b")])
    after = content_chunk_ids([chunk("d.md", 0, "nuevo"), chunk("d.md", 1, "a"), chunk("d.md", 2, "b")])

    assert [c.chunk_id for c in before] == [c.chunk_id for c in after[1:]]


def test_content_ids_change_when_only_a_table_changes():

    [before] = content_chunk_ids([chunk("d.md", 0, "a", tables=["| x |"])])
    [after] = content_chunk_ids([chunk("d.md", 0, "a", tables=["| y |"])])

    assert before.chunk_id != after.chunk_id


def test_content_ids_separate_documents_and_repeated_text():

    ids = [c.chunk_id for c in content_chunk_ids([chunk("a.md", 0, "x"), chunk("b.md", 0, "x"), chunk("a.md", 1, "x")])]

    assert len(set(ids)) == 3


class FakeResp:

    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=SimpleNamespace(status_code=self.status_code))


@pytest.fixture
def requests_queue(monkeypatch):

    calls, queue = [], []

    def fake_request(method, url, json=None, timeout=None):
        calls.append((method, url, timeout))
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(requests, "request", fake_request)

    return calls, queue


def test_remote_embedder_uses_connect_and_read_timeouts(requests_queue):

    calls, queue = requests_queue
    queue.append(FakeResp(body=[[3.0, 4.0]]))

    vector = RemoteEmbedder("http://tei", "m").encode("q", normalize_embeddings=True)

    assert np.allclose(vector, [0.6, 0.8])
    assert calls == [("post", "http://tei/embed", (5, 30))]


@pytest.mark.parametrize("failure", [requests.ConnectionError(), requests.Timeout(), FakeResp(status=503)])
def test_remote_embedder_fails_on_first_error(requests_queue, failure):

    calls, queue = requests_queue
    queue.append(failure)

    with pytest.raises((requests.ConnectionError, requests.Timeout, requests.HTTPError)):
        RemoteEmbedder("http://tei", "m").encode(["q"])

    assert len(calls) == 1


def test_remote_embedder_health_reports_failures(requests_queue):

    calls, queue = requests_queue
    queue.extend([FakeResp(), requests.ConnectionError()])
    embedder = RemoteEmbedder("http://tei", "m")

    assert embedder.health() is True
    assert embedder.health() is False
    assert calls[0] == ("get", "http://tei/health", (5, 30))


def test_remote_embedder_sends_batches_of_32(requests_queue):

    calls, queue = requests_queue
    queue.extend([FakeResp(body=[[1.0]] * 32), FakeResp(body=[[1.0]] * 8)])

    vectors = RemoteEmbedder("http://tei", "m").encode(["t"] * 40)

    assert vectors.shape == (40, 1)
    assert len(calls) == 2


def test_remote_embedder_is_lazy(requests_queue):

    calls, queue = requests_queue
    embedder = RemoteEmbedder("http://tei", "m")

    assert calls == []
    assert embedder._tokenizer is None

    queue.append(FakeResp(body=[[0.0, 1.0, 2.0]]))
    assert embedder.get_dimension() == 3


CONFIG = {"chunk_size": 284, "chunk_overlap": 71, "embedding_model": "m", "embedding_revision": "r1"}


class RecordingEmbedder:

    def __init__(self):
        self.encoded = []

    def get_dimension(self):
        return 2

    def encode(self, texts, normalize_embeddings=False):
        self.encoded.extend(texts)
        return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)


@pytest.fixture
def store(monkeypatch):

    qdrant_client = pytest.importorskip("qdrant_client")
    embedder = RecordingEmbedder()
    monkeypatch.setattr(ingest, "get_embedder", lambda: embedder)

    return qdrant_client.QdrantClient(":memory:"), embedder


def doc(*texts: str) -> list[Chunk]:

    return [chunk("d.md", i, t) for i, t in enumerate(texts)]


def indexed(client) -> dict[str, int]:

    points, _ = client.scroll("corpus_demo", limit=100, with_payload=True)

    return {p.payload["text"]: p.payload["chunk_index"] for p in points}


def sync(client, chunks, config=CONFIG):

    return ingest.sync_index(client, chunks, "corpus_demo", dict(config))


def test_first_sync_builds_the_collection_behind_the_alias(store):

    client, embedder = store

    result = sync(client, doc("a", "b"))

    assert result["rebuilt"] is True
    assert result["added"] == 2
    assert ingest.collection_behind_alias(client, "corpus_demo") == result["collection"]
    assert indexed(client) == {"a": 0, "b": 1}
    assert embedder.encoded == ["a", "b"]


def test_unchanged_corpus_is_a_no_op(store):

    client, embedder = store
    sync(client, doc("a", "b"))
    embedder.encoded.clear()

    result = sync(client, doc("a", "b"))

    assert result == {"collection": result["collection"], "rebuilt": False, "added": 0, "moved": 0, "deleted": 0}
    assert embedder.encoded == []


def test_edited_chunk_is_replaced_and_only_it_is_embedded(store):

    client, embedder = store
    sync(client, doc("a", "b", "c"))
    embedder.encoded.clear()

    result = sync(client, doc("a", "B", "c"))

    assert (result["rebuilt"], result["added"], result["moved"], result["deleted"]) == (False, 1, 0, 1)
    assert embedder.encoded == ["B"]
    assert indexed(client) == {"a": 0, "B": 1, "c": 2}


def test_inserted_chunk_shifts_positions_without_re_embedding(store):

    client, embedder = store
    sync(client, doc("a", "b"))
    embedder.encoded.clear()

    result = sync(client, doc("nuevo", "a", "b"))

    assert (result["added"], result["moved"], result["deleted"]) == (1, 2, 0)
    assert embedder.encoded == ["nuevo"]
    assert indexed(client) == {"nuevo": 0, "a": 1, "b": 2}


def test_removed_chunk_is_deleted(store):

    client, embedder = store
    sync(client, doc("a", "b", "c"))
    embedder.encoded.clear()

    result = sync(client, doc("a", "c"))

    assert (result["added"], result["moved"], result["deleted"]) == (0, 1, 1)
    assert embedder.encoded == []
    assert indexed(client) == {"a": 0, "c": 1}


def test_table_change_alone_replaces_the_chunk(store):

    client, embedder = store
    sync(client, [chunk("d.md", 0, "a", tables=["| 1 |"])])
    embedder.encoded.clear()

    result = sync(client, [chunk("d.md", 0, "a", tables=["| 2 |"])])

    assert (result["added"], result["deleted"]) == (1, 1)
    points, _ = client.scroll("corpus_demo", with_payload=True)
    assert [p.payload["tables"][0]["text_content"] for p in points] == ["| 2 |"]


@pytest.mark.parametrize("key", ["chunk_size", "chunk_overlap", "embedding_model", "embedding_revision"])
def test_config_change_rebuilds_into_a_new_collection_and_keeps_the_previous(store, key):

    client, embedder = store
    first = sync(client, doc("a", "b"))
    embedder.encoded.clear()

    second = sync(client, doc("a", "b"), {**CONFIG, key: "otro"})

    assert second["rebuilt"] is True
    assert second["previous"] == first["collection"]
    assert second["collection"] != first["collection"]
    assert embedder.encoded == ["a", "b"]
    assert ingest.collection_behind_alias(client, "corpus_demo") == second["collection"]
    assert client.collection_exists(first["collection"])


def test_rollback_to_the_previous_config_reuses_the_kept_collection(store):

    client, embedder = store
    first = sync(client, doc("a", "b"))
    sync(client, doc("a", "b"), {**CONFIG, "chunk_size": 512})
    embedder.encoded.clear()

    back = sync(client, doc("a", "b"))

    assert back["collection"] == first["collection"]
    assert (back["rebuilt"], back["added"]) == (True, 0)
    assert embedder.encoded == []


def test_rollback_also_applies_corpus_changes_made_meanwhile(store):

    client, embedder = store
    sync(client, doc("a", "b"))
    sync(client, doc("a", "b"), {**CONFIG, "chunk_size": 512})
    embedder.encoded.clear()

    back = sync(client, doc("a", "c"))

    assert (back["added"], back["deleted"]) == (1, 1)
    assert embedder.encoded == ["c"]
    assert indexed(client) == {"a": 0, "c": 1}


def test_only_the_current_and_previous_versions_are_kept(store):

    client, _ = store
    first = sync(client, doc("a"))
    second = sync(client, doc("a"), {**CONFIG, "chunk_size": 512})
    third = sync(client, doc("a"), {**CONFIG, "chunk_size": 128})

    names = {c.name for c in client.get_collections().collections}

    assert names == {second["collection"], third["collection"]}
    assert not client.collection_exists(first["collection"])


def test_interrupted_rebuild_is_resumed_with_a_diff(store, monkeypatch):

    client, embedder = store
    first = sync(client, doc("a", "b"))
    config = {**CONFIG, "chunk_size": 512}
    encode = embedder.encode

    def fail_on_b(texts, normalize_embeddings=False):
        if "b" in texts:
            raise RuntimeError("TEI caído")
        return encode(texts, normalize_embeddings)

    monkeypatch.setattr(embedder, "encode", fail_on_b)

    with pytest.raises(RuntimeError):
        ingest.sync_index(client, doc("a", "b"), "corpus_demo", dict(config), batch_size=1)

    assert ingest.collection_behind_alias(client, "corpus_demo") == first["collection"]

    monkeypatch.setattr(embedder, "encode", encode)
    embedder.encoded.clear()

    resumed = sync(client, doc("a", "b"), config)

    assert (resumed["rebuilt"], resumed["added"], resumed["deleted"]) == (True, 1, 0)
    assert embedder.encoded == ["b"]
    assert ingest.collection_behind_alias(client, "corpus_demo") == resumed["collection"]
    assert indexed(client) == {"a": 0, "b": 1}


def test_stored_config_round_trips_from_the_payload(store):

    client, _ = store

    result = sync(client, doc("a"))

    assert ingest.get_current_config(client, result["collection"]) == CONFIG
