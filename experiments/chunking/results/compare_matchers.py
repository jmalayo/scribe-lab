"""Compara criterios de acierto de is_chunk_correct sobre el MISMO retrieval por config.

Re-indexa cada config del barrido de chunking en una colección temporal (no toca la
colección del experimento ni MLflow), congela el top-10 de cada pregunta y lo evalúa con:

- estricto_gold_mlflow : `in` estricto con questions.jsonl de 2066799^ (reproduce MLflow)
- estricto             : `in` estricto con el gold set actual
- fuzzy_sin_guardas    : partial_ratio >= 0.8 sobre "".join (versión de 2066799)
- fuzzy_numeros_chunk  : partial_ratio + longitud + números buscados en todo el chunk
- actual               : is_chunk_correct de shared/eval/metrics.py
- cobertura_<x>        : criterio actual con COVERAGE_THRESHOLD=x (calibración)

Además audita qué gold spans no aparecen literal en su source_doc, para el gold set de
HEAD y el actual.

Salida: matcher_comparison.json y matcher_comparison.csv en esta misma carpeta.
"""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from rapidfuzz import fuzz

from experiments.chunking.run import K_MAX, OVERLAP_FRACS, evaluate_chunks_limits, get_tables_from_docs
from shared.eval.data import load_questions
from shared.eval.metrics import _find_numbers, _is_subsequence, _normalize, is_chunk_correct
from shared.ingest import build_qdrant_client, chunk_documents, corpus_hash, index_chunks, load_corpus
from shared.retrieval import dense_search
from shared.settings import settings

OUT_DIR = Path(__file__).resolve().parent
TMP_COLLECTION = "compare_matchers_tmp"
COVERAGES = [0.7, 0.8, 0.85, 0.9, 0.95]

def _git_questions(rev: str) -> list[dict]:
    raw = subprocess.check_output(["git", "show", f"{rev}:shared/eval/questions.jsonl"]).decode()
    return [json.loads(l) for l in raw.splitlines() if l.strip()]

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

def fuzzy_numbers_whole_chunk(c, q):
    if not _valid_doc(c, q):
        return False
    ct = _normalize(" ".join(_texts(c)))
    for s in map(_normalize, q["gold_spans"]):
        if len(ct) >= len(s) and _is_subsequence(_find_numbers(s), _find_numbers(ct)) \
                and fuzz.partial_ratio(s, ct) / 100 >= 0.8:
            return True
    return False

def _align(s, ct):
    a = fuzz.partial_ratio_alignment(s, ct)
    coverage = min(a.src_end - a.src_start, a.dest_end - a.dest_start) / len(s)
    window = ct[a.dest_start:a.dest_end]
    return a.score / 100, coverage, window, _is_subsequence(_find_numbers(s), _find_numbers(window))

def coverage_variant(cov_min):
    def m(c, q):
        if not _valid_doc(c, q):
            return False
        ct = _normalize(" ".join(_texts(c)))
        for s in map(_normalize, q["gold_spans"]):
            score, coverage, _, nums_ok = _align(s, ct)
            if score >= 0.8 and coverage >= cov_min and nums_ok:
                return True
        return False
    return m

def _first_rank(m, chunks, q, k):
    return next((i + 1 for i, c in enumerate(chunks[:k]) if m(c, q)), None)

def _metrics(m, results, qs):
    ranks = {q["id"]: _first_rank(m, results[q["id"]], q, K_MAX) for q in qs}
    out = {
        f"recall@{k}": round(sum(r is not None and r <= k for r in ranks.values()) / len(qs), 4)
            for k in (1, 3, 5, 10)
    }
    out["mrr@10"] = round(sum(1 / r for r in ranks.values() if r) / len(qs), 4)
    out["ranks"] = ranks
    return out

def audit_gold(qs: list[dict]) -> list[dict]:
    root = Path(settings.corpus_dir)
    docs = {str(p.relative_to(root)): _normalize(p.read_text(encoding="utf-8")) for p in root.rglob("*.md")}
    out = []
    for q in qs:
        sources = [d.strip() for d in q["source_doc"].split(";")]
        for s in q["gold_spans"]:
            n = _normalize(s)
            if not any(n in docs.get(d, "") for d in sources):
                best = max(fuzz.partial_ratio(n, docs.get(d, "")) / 100 for d in sources)
                out.append({"id": q["id"], "best_score": round(best, 3), "gold_span": s})
    return out

def main():
    docs = load_corpus()
    qs = load_questions()
    qs_mlflow = _git_questions("2066799^")
    qs_head = _git_questions("HEAD")

    text_wo, tables = [], []
    for d in docs:
        t, txt = get_tables_from_docs(d)
        text_wo.append(txt)
        tables.extend(t)
    sizes = evaluate_chunks_limits("".join(text_wo))

    matchers = {
        "estricto_gold_mlflow": (strict, qs_mlflow),
        "estricto": (strict, qs),
        "fuzzy_sin_guardas": (fuzzy_plain, qs),
        "fuzzy_numeros_chunk": (fuzzy_numbers_whole_chunk, qs),
        "actual": (is_chunk_correct, qs),
        **{f"cobertura_{c}": (coverage_variant(c), qs) for c in COVERAGES},
    }

    client = build_qdrant_client()
    report = {
        "chunk_sizes": sizes,
        "gold_audit": {"HEAD": audit_gold(qs_head), "actual": audit_gold(qs)},
        "configs": {},
        "calibration_pairs": [],
    }

    for cs in sizes:
        for ov in OVERLAP_FRACS:
            name = f"cs{cs}_ov{int(ov * 100)}"
            chunks = chunk_documents(docs, tables, cs, int(cs * ov))
            index_chunks(client, chunks, TMP_COLLECTION, {"chunk_size": cs, "chunk_overlap": int(cs * ov), "corpus_hash": corpus_hash(docs)})
            results = {q["id"]: dense_search(client, TMP_COLLECTION, q["question"], K_MAX) for q in qs}

            report["configs"][name] = {
                "n_chunks": len(chunks),
                **{mn: _metrics(m, results, qq) for mn, (m, qq) in matchers.items()},
            }

            # pares que superan el score pero no cubren el span completo: base de la calibración
            for q in qs:
                for rank, c in enumerate(results[q["id"]], 1):
                    if not _valid_doc(c, q):
                        continue
                    ct = _normalize(" ".join(_texts(c)))
                    for s in map(_normalize, q["gold_spans"]):
                        score, coverage, window, nums_ok = _align(s, ct)
                        if score >= 0.8 and coverage < 1.0:
                            report["calibration_pairs"].append({
                                "config": name, "id": q["id"], "rank": rank, "score": round(score, 3),
                                "coverage": round(coverage, 3), "numbers_in_window": nums_ok,
                                "gold_span": s, "window": window,
                            })
            print(name, {k: v["recall@5"] for k, v in report["configs"][name].items() if isinstance(v, dict)}, flush=True)

    client.delete_collection(TMP_COLLECTION)

    (OUT_DIR / "matcher_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    with (OUT_DIR / "matcher_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "criterio", "recall@1", "recall@3", "recall@5", "recall@10", "mrr@10"])
        for cfg, row in report["configs"].items():
            for crit, m in row.items():
                if isinstance(m, dict):
                    w.writerow([cfg, crit, m["recall@1"], m["recall@3"], m["recall@5"], m["recall@10"], m["mrr@10"]])

if __name__ == "__main__":
    main()
