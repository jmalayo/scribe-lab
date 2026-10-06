import pytest

from shared.eval.metrics import _find_numbers, calculate_hit_at_k, mrr, recall_at_k

DOC = "doc/benchmark.md"
SPAN = "Random Forest obtuvo 48.9 % ± 13.3 de balanced accuracy con 5 folds"

def test_find_numbers_keeps_order_and_ignores_space_before_percent():
    assert _find_numbers("37.5 % y 37.5%") == ["37.5%", "37.5%"]
    assert _find_numbers("22050, no 44100") == ["22050", "44100"]

@pytest.fixture
def ranking():

    q1, q2, q3 = (
        {
            "id": i, 
            "source_doc": DOC, 
            "gold_spans": [SPAN]
        } for i in ("q1", "q2", "q3")
    )

    miss = {
        "doc_id": DOC, 
        "text": "texto sin relación con la pregunta", 
        "tables": []
    }

    hit = {
        "doc_id": DOC, 
        "text": f"Contexto previo. {SPAN} Fin.", 
        "tables": []
    }

    results = {
        "q1": [miss, hit,   miss, miss, miss], # resultado 1 / 2
        "q2": [miss, miss, miss, hit, miss], # resultado 1 / 4
        "q3": [miss] * 5, # reusltado 0
    }

    return results, [q1, q2, q3]

def test_recall_at_k(ranking):
    results, questions = ranking

    assert recall_at_k(results, questions, 1) == 0
    assert recall_at_k(results, questions, 3) == pytest.approx(1 / 3)
    assert recall_at_k(results, questions, 5) == pytest.approx(2 / 3)

def test_mrr(ranking):
    results, questions = ranking

    assert mrr(results, questions, 10) == pytest.approx((1 / 2 + 1 / 4 + 0) / 3)
    assert mrr(results, questions, 3) == pytest.approx((1 / 2) / 3)

def test_calculate_hit_at_k(ranking):
    results, questions = ranking

    assert calculate_hit_at_k(results, questions, 1) == [0.0, 0.0, 0.0]
    assert calculate_hit_at_k(results, questions, 3) == [1.0, 0.0, 0.0]
    assert calculate_hit_at_k(results, questions, 5) == [1.0, 1.0, 0.0]
