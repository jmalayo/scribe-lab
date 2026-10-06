import json
import time
from pathlib import Path

from experiments.chunking.run import get_tables_from_docs
from shared.eval.data import load_questions
from shared.eval.metrics import bootstrap_ci, calculate_hit_at_k, latency_summary, mrr, recall_at_k
from shared.ingest import build_qdrant_client, chunk_documents, index_chunks, load_corpus
from shared.retrieval import BM25Index, dense_search, fetch_all_chunks, reciprocal_rank_fusion, rerank
from shared.tracking import get_best_run, log_metrics, tracked_run

from qdrant_client import QdrantClient

POOL_SIZE = 20  # cuántos candidatos trae el retriever base antes de rerankear
TOP_K = 10
COLLECTION = "exp_reranking"
OUT_DIR = Path(__file__).resolve().parent

_best_chunking_run = get_best_run(
    "exp_chunking_dynamic_tables",
    recall_at_k="recall_at_10",
    tiebreak="mrr_at_10",
    filter_string="attributes.status = 'FINISHED'"
)
BEST_CHUNKING = {
    "chunk_size": int(_best_chunking_run["metrics.chunk_size"]),
    "chunk_overlap": int(_best_chunking_run["metrics.overlap_tokens"]),
}
BEST_METHOD = get_best_run(
    "hybrid-search",
    recall_at_k="recall_at_10",
    tiebreak="mrr_at_10",
    filter_string=(
        f"params.chunk_size = '{BEST_CHUNKING['chunk_size']}' "
        f"and params.chunk_overlap = '{BEST_CHUNKING['chunk_overlap']}'"
    )
)["params.method"]

def base_search(
    client: QdrantClient = None, 
    bm25: BM25Index = None, 
    question: str = None,
    collection: str = COLLECTION,
    k: int = POOL_SIZE
) -> list[dict]:

    if BEST_METHOD == "dense":
        return dense_search(client, collection, question, k)

    if BEST_METHOD == "bm25":
        return bm25.search(question, k)

    if BEST_METHOD == "hybrid_rrf":

        d = dense_search(client, collection, question, k)
        b = bm25.search(question, k)

        return reciprocal_rank_fusion([d, b], top_k=k)

    raise ValueError(f"Invalid method: {BEST_METHOD}")

def evaluate(name, results, questions, latencies):

    ci_low, ci_high = bootstrap_ci(calculate_hit_at_k(results, questions, TOP_K))
    return {
        "config": name,
        "recall@5": round(recall_at_k(results, questions, 5), 4),
        "recall@10": round(recall_at_k(results, questions, TOP_K), 4),
        "recall@10_ci95_low": ci_low,
        "recall@10_ci95_high": ci_high,
        "mrr@10": round(mrr(results, questions, TOP_K), 4),
        **latency_summary(latencies),
    }


def main():

    docs = load_corpus()
    questions = load_questions()

    client = build_qdrant_client()
    
    tables = [t for doc in docs for t in get_tables_from_docs(doc)[0]]

    chunks = chunk_documents(docs, tables, BEST_CHUNKING["chunk_size"], BEST_CHUNKING["chunk_overlap"])

    index_chunks(client, chunks, COLLECTION, BEST_CHUNKING)

    bm25 = BM25Index(
        fetch_all_chunks(client, COLLECTION)
    )

    baseline_results, reranked_results = {}, {}
    baseline_lat, reranked_lat = [], []

    for q in questions:

        t0 = time.perf_counter()

        candidates = base_search(
            client=client, 
            bm25=bm25, 
            question=q["question"], 
            k=POOL_SIZE
        )

        base_elapsed = (time.perf_counter() - t0) * 1000

        baseline_lat.append(base_elapsed)

        baseline_results[q["id"]] = candidates[:TOP_K]

        t0 = time.perf_counter()

        reranked = rerank(q["question"], candidates, TOP_K)

        rerank_elapsed = (time.perf_counter() - t0) * 1000

        reranked_lat.append(base_elapsed + rerank_elapsed)

        reranked_results[q["id"]] = reranked

    rows = [
        evaluate("baseline", baseline_results, questions, baseline_lat),
        evaluate("reranked", reranked_results, questions, reranked_lat),
    ]

    delta_recall_at_10 = rows[1]["recall@10"] - rows[0]["recall@10"]

    for row in rows:
        with tracked_run(
            "reranking",
            row["config"],
            {
                **BEST_CHUNKING, 
                "base_method": BEST_METHOD, 
                "use_reranker": row["config"] == "reranked"
            }
        ):
            log_metrics(row)

    results = {
        "chunking_config": BEST_CHUNKING,
        "base_method": BEST_METHOD,
        "n_questions": len(questions),
        "rows": rows,
        "delta_recall@10": round(delta_recall_at_10, 4),
        "delta_p95_ms": round(rows[1]["p95_ms"] - rows[0]["p95_ms"], 1),
        "negative_finding": delta_recall_at_10 < 0,
    }

    (OUT_DIR / "results").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results" / "results.json").write_text(
        json.dumps(
            results, 
            indent=2, 
            ensure_ascii=False
        ), encoding="utf-8"
    )

    for r in rows:
        print(f"{r['config']:>10}: recall@10={r['recall@10']:.3f} mrr@10={r['mrr@10']:.3f} "
              f"p50={r['p50_ms']}ms p95={r['p95_ms']}ms")

    print(f"\n¿Se adopta el reranker?: {delta_recall_at_10 > 0} -> guardado en reranking/results/results.json")

if __name__ == "__main__":
    main()
