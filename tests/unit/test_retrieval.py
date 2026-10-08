import math

import pytest

from shared.retrieval import BM25Index, reciprocal_rank_fusion


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

    assert index.search("hpss", k=2)[0]["chunk_id"] == "short"

def test_bm25_returns_k_results_even_with_zero_score(corpus):

    # sin umbral: la búsqueda siempre devuelve k candidatos, aunque no compartan términos
    results = BM25Index(corpus).search("término inexistente", k=3)

    assert len(results) == 3
    assert all(r["score"] == 0.0 for r in results)

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

    # "b" es 2.º en ambas listas (2/62) y supera a "a", 1.º en una sola (1/61)
    dense = [chunk("a"), chunk("b")]
    sparse = [chunk("c"), chunk("b")]

    fused = reciprocal_rank_fusion([dense, sparse])

    assert fused[0]["chunk_id"] == "b"

def test_rrf_top_k_and_unique_ids():

    rankings = [[chunk("a"), chunk("b"), chunk("c")], [chunk("c"), chunk("d")]]

    assert len(reciprocal_rank_fusion(rankings)) == 4
    assert [c["chunk_id"] for c in reciprocal_rank_fusion(rankings, top_k=2)] == [
        "c",
        "a",
    ]

def test_rrf_empty_rankings():

    assert reciprocal_rank_fusion([[], []]) == []
