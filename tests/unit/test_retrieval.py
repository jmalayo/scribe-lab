import math

import pytest

from shared import ingest, retrieval
from shared.retrieval import (
    BM25Index,
    dense_search,
    fetch_all_chunks,
    reciprocal_rank_fusion,
    rerank,
)


def chunk(cid: str, text: str = "") -> dict:
    return {"chunk_id": cid, "doc_id": "doc/a.md", "text": text}

@pytest.fixture
def corpus():

    return [
        chunk("c1", "Random Forest obtuvo balanced accuracy"),
        chunk("c2", "HPSS domina el tiempo de extracción"),
        chunk("c3", "el kernel de HPSS se validó con Spearman"),
    ]

def test_bm25_score_matches_formula():

    # 2 docs: "a b" (len 2) y "c" (len 1); avgdl=1.5, df(a)=1
    index = BM25Index([chunk("x", "a b"), chunk("y", "c")])

    idf = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    norm = 1.5 * (1 - 0.75 + 0.75 * 2 / 1.5)
    expected = idf * (1 * (1.5 + 1)) / (1 + norm)

    top = index.search("a", k=1)[0]

    assert top["chunk_id"] == "x"
    assert top["score"] == pytest.approx(expected)

def test_bm25_ranks_matching_chunk_first_and_is_case_insensitive(corpus):

    results = BM25Index(corpus).search("random FOREST", k=3)

    assert results[0]["chunk_id"] == "c1"
    assert results[0]["score"] > 0

def test_bm25_rarer_term_weighs_more(corpus):

    # "hpss" aparece en 2 chunks, "spearman" en 1: c3 tiene ambos y gana a c2
    results = BM25Index(corpus).search("hpss spearman", k=2)

    assert [r["chunk_id"] for r in results] == ["c3", "c2"]

def test_bm25_shorter_chunk_wins_with_same_term_frequency():

    index = BM25Index([
        chunk("long", "hpss uno dos tres cuatro cinco"),
        chunk("short", "hpss uno"),
    ])

    # Formula de normalización penaliza y premia la relevancia (score)
    assert index.search("hpss", k=2)[0]["chunk_id"] == "short"

def test_bm25_returns_k_results_even_with_zero_score(corpus):

    # Sin umbral la búsqueda siempre devuelve k candidatos, aunque no compartan términos
    results = BM25Index(corpus).search("término inexistente", k=3)

    assert len(results) == 3
    assert all(
        r["score"] == 0.0 
            for r in results
    )

def test_bm25_keeps_payload_and_handles_empty_index(corpus):

    top = BM25Index(corpus).search("random", k=1)[0]

    assert top["doc_id"] == "doc/a.md" and top["text"] == corpus[0]["text"]
    assert BM25Index([]).search("random", k=5) == []

def test_rrf_score_matches_formula():

    fused = reciprocal_rank_fusion([[chunk("a"), chunk("b")], [chunk("a")]])

    assert fused[0]["chunk_id"] == "a"
    assert fused[0]["rrf_score"] == pytest.approx(2 / 61)
    assert fused[1]["rrf_score"] == pytest.approx(1 / 62)

def test_rrf_rewards_agreement_over_single_top_rank():

    # "b" es 2.º en ambas listas y supera a "a", 1.º posición tras fused
    dense = [chunk("a"), chunk("b")]
    sparse = [chunk("c"), chunk("b")]

    fused = reciprocal_rank_fusion([dense, sparse])

    assert fused[0]["chunk_id"] == "b"

def test_rrf_top_k_and_unique_ids():

    rankings = [[chunk("a"), chunk("b"), chunk("c")], [chunk("c"), chunk("d")]]

    assert len(reciprocal_rank_fusion(rankings)) == 4
    assert [
        c["chunk_id"] 
            for c in reciprocal_rank_fusion(rankings, top_k=2)
        ] == ["c", "a",]

def test_rrf_empty_rankings():

    assert reciprocal_rank_fusion([[], []]) == []

# mocking Qdrant, embedder y cross-encoder
class FakeVector(list):

    def tolist(self):
        return list(self)

class FakeEmbedder:

    def encode(self, text, normalize_embeddings=False):
        return FakeVector([0.1, 0.2])

class FakePoint:

    def __init__(self, payload, score=0.0):
        self.payload = payload
        self.score = score

class FakeQueryResponse:

    def __init__(self, points):
        self.points = points

class FakeQdrant:

    def __init__(self, records, page_size):
        self.records = records
        self.page_size = page_size
        self.scroll_offsets = []
        self.query_args = None

    def query_points(self, collection_name, query, limit):
        self.query_args = (collection_name, query, limit)

        return FakeQueryResponse(self.records[:limit])

    def scroll(self, collection_name, scroll_filter, limit, with_vectors, with_payload, offset):
        self.scroll_offsets.append(offset)

        start = offset or 0
        end = start + self.page_size

        return self.records[start:end], (end if end < len(self.records) else None)

class FakeCrossEncoder:

    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs, batch_size, show_progress_bar):
        return [self.scores[text] for _, text in pairs]

def test_dense_search_merges_payload_and_score(monkeypatch):

    monkeypatch.setattr(ingest, "get_embedder", lambda: FakeEmbedder())

    client = FakeQdrant(
        [
            FakePoint(chunk("c1"), 0.9), 
            FakePoint(chunk("c2"), 0.4)
        ], page_size=10)

    hits = dense_search(client, "exp", "pregunta", k=2)

    assert client.query_args == ("exp", [0.1, 0.2], 2)
    assert [h["chunk_id"] for h in hits] == ["c1", "c2"]
    assert [h["score"] for h in hits] == [0.9, 0.4]

# fetch_all_chunks arma el índice BM25 con la colección completa: tiene que seguir el offset de
# scroll hasta que Qdrant devuelve None, sin perder ni repetir chunks entre páginas
def test_fetch_all_chunks_follows_scroll_pages():
    records = [FakePoint(chunk(f"c{i}")) for i in range(5)]
    client = FakeQdrant(records, page_size=2)

    chunks = fetch_all_chunks(client, "exp")

    assert [c["chunk_id"] for c in chunks] == ["c0", "c1", "c2", "c3", "c4"]
    assert client.scroll_offsets == [None, 2, 4]

def test_fetch_all_chunks_empty_collection():
    assert fetch_all_chunks(FakeQdrant([], page_size=2), "exp") == []

# rerank reordena por el score del cross-encoder (no por el de la búsqueda base), corta en top_k
# y conserva el payload original junto a rerank_score
def test_rerank_orders_by_cross_encoder_score(monkeypatch):
    scores = {"uno": 0.1, "dos": 0.9, "tres": 0.5}
    monkeypatch.setattr(retrieval, "get_rerank", lambda: FakeCrossEncoder(scores))
    candidates = [chunk("c1", "uno"), chunk("c2", "dos"), chunk("c3", "tres")]

    reranked = rerank("pregunta", candidates, top_k=2)

    assert [c["chunk_id"] for c in reranked] == ["c2", "c3"]
    assert [c["rerank_score"] for c in reranked] == [0.9, 0.5]
    assert reranked[0]["doc_id"] == "doc/a.md"

def test_rerank_empty_candidates_skips_model(monkeypatch):
    monkeypatch.setattr(retrieval, "get_rerank", lambda: pytest.fail("no debe cargar el modelo"))

    assert rerank("pregunta", [], top_k=5) == []
