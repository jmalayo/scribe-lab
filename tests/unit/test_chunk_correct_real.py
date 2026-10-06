import json
from pathlib import Path

import pytest

from shared.eval.metrics import is_chunk_correct

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

REAL_CHUNKS = json.loads((FIXTURES / "real_chunks.json").read_text(encoding="utf-8"))

# chunks reales del corpus (barrido de chunking 2026-10-05): q002/q014/q028 son aciertos
# verificados a mano, q011 es un truncamiento real que sin guardas cambiaba la config ganadora
@pytest.mark.parametrize(
    "case",
    REAL_CHUNKS,
    ids=[c["id"] for c in REAL_CHUNKS]
)
def test_is_chunk_correct_real_chunks(case):
    q = {
        "source_doc": case["source_doc"],
        "gold_spans": [case["gold_span"]]
    }

    assert is_chunk_correct(
        case["chunk"],
        q
    ) is case["expected"], case["why"]
