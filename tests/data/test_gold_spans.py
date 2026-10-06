from pathlib import Path

from rapidfuzz import fuzz

from shared.eval.data import load_questions
from shared.eval.metrics import _normalize
from shared.settings import settings

CORPUS_DIR = Path(settings.corpus_dir)

def test_gold_spans_are_literal_in_source():

    docs = {}

    for reports_md in CORPUS_DIR.rglob("*.md"):
        relative_md = str(reports_md.relative_to(CORPUS_DIR))
        context_text = _normalize(reports_md.read_text(encoding="utf-8"))
        
        docs[relative_md] = context_text
        
    questions = load_questions()
    total = 0
    failures = []

    for q in questions:

        sources = [
            d.strip() 
                for d in q["source_doc"].split(";")
        ]

        missing = [
            d for d in sources
                if d not in docs
        ]

        for span in q["gold_spans"]:
            total += 1
            norm_span = _normalize(span)

            if missing:
                failures.append(f"  {q['id']} (documento inexistente: {', '.join(missing)}): {span[:80]}")

                continue

            if any(norm_span in docs[d] for d in sources):
                
                continue

            best = max(fuzz.partial_ratio(norm_span, docs[d]) / 100 for d in sources)
            failures.append(f"  {q['id']} ({best:.3f}): {span[:80]}")

    assert not failures, (
        f"{len(failures)} de {total} gold spans no aparecen literal en su source_doc:\n"
        + "\n".join(failures)
    )
