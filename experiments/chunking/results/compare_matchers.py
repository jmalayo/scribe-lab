from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from rapidfuzz import fuzz

from experiments.chunking.run import (
    K_MAX,
    OVERLAP_FRACS,
    evaluate_chunks_limits,
    get_tables_from_docs,
)
from shared.eval.data import load_questions
from shared.eval.metrics import _normalize, is_chunk_correct
from shared.ingest import (
    build_qdrant_client,
    chunk_documents,
    corpus_hash,
    index_chunks,
    load_corpus,
)
from shared.retrieval import dense_search

OUT_DIR = Path(__file__).resolve().parent
TMP_COLLECTION = "compare_matchers_tmp"

def _git_questions(rev: str) -> list[dict]:
    raw = subprocess.check_output(["git", "show", f"{rev}:shared/eval/questions.jsonl"]).decode()
    return [json.loads(line) for line in raw.splitlines() if line.strip()]

def _valid_doc(c, q):
    return c.get("doc_id") in [d.strip() for d in q["source_doc"].split(";")]

def _texts(c):
    return [c.get("text", "")] + [t.get("text_content", "") for t in c.get("tables", [])]

def strict(c, q):
    if not _valid_doc(c, q):
        return False
    ct = _normalize(" ".join(_texts(c)))
    return any(_normalize(s) in ct for s in q["gold_spans"])

def fuzzy_plain(c, q):
    if not _valid_doc(c, q):
        return False
    ct = _normalize("".join(_texts(c)))
    return any(fuzz.partial_ratio(_normalize(s), ct) / 100 >= 0.8 for s in q["gold_spans"])

def _first_rank(m, chunks, q):
    return next((i + 1 for i, c in enumerate(chunks) if m(c, q)), None)

def _metrics(m, results, qs):
    ranks = [_first_rank(m, results[q["id"]], q) for q in qs]
    out = {
        f"recall@{k}": round(sum(r is not None and r <= k for r in ranks) / len(qs), 4)
            for k in (1, 3, 5, 10)
    }
    out["mrr@10"] = round(sum(1 / r for r in ranks if r) / len(qs), 4)
    return out

def main():
    docs = load_corpus()
    qs = load_questions()

    text_wo, tables = [], []
    for d in docs:
        t, txt = get_tables_from_docs(d)
        text_wo.append(txt)
        tables.extend(t)
    sizes = evaluate_chunks_limits("".join(text_wo))

    matchers = {
        "in_gold_mlflow": (strict, _git_questions("2066799^")),
        "in_gold_actualizado": (strict, qs),
        "fuzzy_sin_guardas": (fuzzy_plain, qs),
        "fuzzy_actual": (is_chunk_correct, qs),
    }

    client = build_qdrant_client()
    rows = []

    for cs in sizes:
        for ov in OVERLAP_FRACS:
            name = f"cs{cs}_ov{int(ov * 100)}"
            chunks = chunk_documents(docs, tables, cs, int(cs * ov))
            index_chunks(client, chunks, TMP_COLLECTION, {"chunk_size": cs, "chunk_overlap": int(cs * ov), "corpus_hash": corpus_hash(docs)})
            results = {q["id"]: dense_search(client, TMP_COLLECTION, q["question"], K_MAX) for q in qs}

            for crit, (m, qq) in matchers.items():
                rows.append({"config": name, "criterio": crit, **_metrics(m, results, qq)})

            print(name, {r["criterio"]: r["recall@5"] for r in rows if r["config"] == name}, flush=True)

    client.delete_collection(TMP_COLLECTION)

    with (OUT_DIR / "matcher_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["config", "criterio", "recall@1", "recall@3", "recall@5", "recall@10", "mrr@10"])
        w.writeheader()
        w.writerows(rows)

if __name__ == "__main__":
    main()
