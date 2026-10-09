import pytest

from shared.eval.metrics import (
    _find_numbers,
    bootstrap_ci,
    calculate_hit_at_k,
    groundedness_rate,
    hallucination_rate,
    latency_summary,
    mrr,
    percentile,
    recall_at_k,
)

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

def test_percentile_linear_interpolation():
    values = [7, 3, 10, 1, 5, 9, 2, 8, 4, 6]

    assert percentile(values, 50) == pytest.approx(5.5) # k = 9 * 0.50 = 4.5 -> 5 + (6 - 5) * 0.5, la mediana
    assert percentile(values, 95) == pytest.approx(9.55) # k = 9 * 0.95 = 8.55 -> 9 + (10 - 9) * 0.55
    assert percentile(values, 0) == 1
    assert percentile(values, 100) == 10 # freno: no hay vecino a la derecha

def test_percentile_edge_cases():
    assert percentile([42.0], 95) == 42.0
    assert percentile([], 50) == 0.0

def test_latency_summary():
    # p50: k = 1.5 -> 20 + 10 * 0.5 = 25; p95: k = 2.85 -> 30 + 10 * 0.85 = 38.5
    assert latency_summary([10, 20, 30, 40]) == {"p50_ms": 25.0, "p95_ms": 38.5, "mean_ms": 25}
    assert latency_summary([]) == {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0} # sin latencias no debe romper

def test_bootstrap_ci_reproducible_and_contains_mean():
    hits = [1.0] * 20 + [0.0] * 14
    mean = sum(hits) / len(hits)

    low, high = bootstrap_ci(hits)

    assert (low, high) == bootstrap_ci(hits) # reproducibilidad
    assert low <= mean <= high # media observada dentro de intervalo
    assert 0.0 <= low < high <= 1.0 # comprobación de limites

def test_bootstrap_ci_degenerate_inputs():
    assert bootstrap_ci([0.7] * 5) == (0.7, 0.7)
    assert bootstrap_ci([]) == (0.0, 0.0)

def test_groundedness_and_hallucination_rate():
    verdicts = [True, True, False, True]

    assert groundedness_rate(verdicts) == pytest.approx(0.75)
    assert hallucination_rate(verdicts) == pytest.approx(0.25)

    assert groundedness_rate([]) == 0.0
    assert hallucination_rate([]) == 0.0
