import json
from pathlib import Path

import pytest

from shared.eval.metrics import _find_numbers, is_chunk_correct, mrr, recall_at_k

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

DOC = "music-tagger/tagger-music-genesis.md"

# gold span real de q019, base de los casos sintéticos
SPAN = "Verificado 2026-08-18 contra el código real (`features/config.py`): el SR actual es 22050, no 44100"
INVERTED = SPAN.replace("22050", "@@").replace("44100", "22050").replace("@@", "44100")
TRUNCATED = SPAN[:int(len(SPAN) * 0.65)]
LONG_CONTEXT = "En esta sección se documenta la configuración de audio usada por el pipeline de extracción de features. " * 2

REAL_CHUNKS = json.loads((FIXTURES / "real_chunks.json").read_text(encoding="utf-8"))

def question(
    spans: list[str], 
    source_doc: str = DOC
) -> dict:
    return {
        "source_doc": source_doc, 
        "gold_spans": spans
    }

def chunk(
    text: str, 
    doc_id: str = DOC, 
    tables: list[dict] | None = None
) -> dict:
    return {
        "doc_id": doc_id, 
        "text": text, 
        "tables": tables or []
    }

@pytest.mark.parametrize(
    "text, expected",
    [
        (f"Contexto previo. {SPAN} Fin.", True),
        (f"Contexto previo. {SPAN.replace('Verificado', 'Verificaod')} Fin.", True),
        (f"Contexto previo. {INVERTED} Fin.", False),
        (f"Contexto previo. {INVERTED} Más adelante se probó 22050 y luego 44100.", False),
        (f"Contexto previo. {TRUNCATED}", False),
        (f"{LONG_CONTEXT}{TRUNCATED}", False),
    ],
    ids=[
        "identico",
        "typo_de_una_letra",
        "numeros_invertidos",
        "numeros_invertidos_con_correctos_en_otra_oracion",
        "truncado_chunk_corto",
        "truncado_chunk_largo",
    ],
)

def test_is_chunk_correct_synthetic(text, expected):
    assert is_chunk_correct(
        chunk(text), 
        question([SPAN])
    ) is expected

def test_wrong_source_doc_is_rejected():
    assert not is_chunk_correct(chunk(SPAN, doc_id="otro/doc.md"), question([SPAN]))

def test_multiple_source_docs():
    q = question([SPAN], source_doc=f"otro/doc.md; {DOC}")
    assert is_chunk_correct(chunk(SPAN), q)

def test_bold_is_normalized():
    assert is_chunk_correct(chunk("el SR actual es **22050**, no 44100 según el código"), question(["el SR actual es 22050, no 44100"]))

def test_span_inside_table_payload():
    row = "| 9 | 0.888 | 37.5 % | 353.0 % |"
    c = chunk("Validación de kernel_size para HPSS.", tables=[{"text_content": row}])
    assert is_chunk_correct(c, question([row]))

def test_find_numbers_keeps_order_and_ignores_space_before_percent():
    assert _find_numbers("37.5 % y 37.5%") == ["37.5%", "37.5%"]
    assert _find_numbers("22050, no 44100") == ["22050", "44100"]
    

@pytest.mark.parametrize("case", REAL_CHUNKS, ids=[c["id"] for c in REAL_CHUNKS])
def test_is_chunk_correct_real_chunks(case):
    q = question([case["gold_span"]], source_doc=case["source_doc"])
    assert is_chunk_correct(case["chunk"], q) is case["expected"], case["why"]

@pytest.fixture
def ranking():
    # q1: acierto en rank 2, q2: acierto en rank 4, q3: sin acierto
    q1, q2, q3 = (dict(question([SPAN]), id=i) for i in ("q1", "q2", "q3"))
    miss = chunk("texto sin relación con la pregunta")
    hit = chunk(f"Contexto previo. {SPAN} Fin.")
    results = {
        "q1": [miss, hit, miss, miss, miss],
        "q2": [miss, miss, miss, hit, miss],
        "q3": [miss] * 5,
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
