from types import SimpleNamespace

import numpy as np
import pytest
import requests

from shared import ingest
from shared.ingest import Chunk, RemoteEmbedder, content_chunk_ids


def chunk(doc_id: str, i: int, text: str) -> Chunk:

    return Chunk(chunk_id=f"pos-{i}", doc_id=doc_id, library="lib", text=text, chunk_index=i)


def test_content_ids_survive_a_shift_in_position():

    before = content_chunk_ids([chunk("d.md", 0, "a"), chunk("d.md", 1, "b")])
    after = content_chunk_ids([chunk("d.md", 0, "nuevo"), chunk("d.md", 1, "a"), chunk("d.md", 2, "b")])

    assert [c.chunk_id for c in before] == [c.chunk_id for c in after[1:]]


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


def test_index_with_alias_swaps_atomically_and_drops_the_old_collection(monkeypatch):

    qdrant_client = pytest.importorskip("qdrant_client")

    class FakeEmbedder:
        def get_dimension(self):
            return 2

        def encode(self, texts, normalize_embeddings=False):
            return np.ones((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr(ingest, "get_embedder", lambda: FakeEmbedder())
    client = qdrant_client.QdrantClient(":memory:")
    chunks = [chunk("d.md", 0, "a"), chunk("d.md", 1, "b")]

    first = ingest.index_with_alias(client, chunks, "corpus_demo", "v1", {})
    again = ingest.index_with_alias(client, chunks, "corpus_demo", "v1", {})
    second = ingest.index_with_alias(client, chunks, "corpus_demo", "v2", {})

    assert first == {"collection": "corpus_demo__v1", "indexed": 2, "changed": True}
    assert again["changed"] is False
    assert second["collection"] == "corpus_demo__v2"
    assert ingest.collection_behind_alias(client, "corpus_demo") == "corpus_demo__v2"
    assert not client.collection_exists("corpus_demo__v1")
    assert client.count("corpus_demo").count == 2
