import json
import time
from pathlib import Path

from experiments.chunking.run import get_tables_from_docs
from experiments.evaluation.prompts import prompt_v1
from experiments.reranking.run import (
    BEST_CHUNKING,
    BEST_METHOD,
    POOL_SIZE,
    TOP_K,
    base_search,
)
from shared.eval.data import load_questions
from shared.eval.judge import parse_verdict
from shared.eval.metrics import (
    bootstrap_ci,
    groundedness_rate,
    hallucination_rate,
    latency_summary,
)
from shared.ingest import (
    build_qdrant_client,
    chunk_documents,
    index_chunks,
    load_corpus,
)
from shared.llm import generate
from shared.pipeline import PipelineConfig
from shared.prompts import build_context_text
from shared.retrieval import BM25Index, fetch_all_chunks, rerank
from shared.settings import settings
from shared.tracking import log_metrics, tracked_run

COLLECTION = "exp_evaluation"
OUT_DIR = Path(__file__).resolve().parent

ANSWER_PROMPT, GROUNDEDNESS_PROMPT, RELEVANCE_PROMPT = prompt_v1()

USE_RERANKER = PipelineConfig.from_yaml(settings.pipeline_config_path).reranker

def retrieve_context(client, bm25, question: str) -> list[dict]:

    candidates = base_search(client, bm25, question, collection=COLLECTION, k=POOL_SIZE)

    if USE_RERANKER:
        return rerank(question, candidates, TOP_K)

    return candidates[:TOP_K]

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

    grounded_verdicts, relevant_verdicts, gen_latencies = [], [], []
    per_question_rows = []

    for q in questions:

        context_chunks = retrieve_context(client, bm25, q["question"])
        context_text = build_context_text(context_chunks)

        t0 = time.perf_counter()
        answer = generate(ANSWER_PROMPT.format(context=context_text, question=q["question"]))
        gen_latencies.append((time.perf_counter() - t0) * 1000)

        grounded_raw = generate(GROUNDEDNESS_PROMPT.format(context=context_text, answer=answer))
        grounded, _ = parse_verdict(grounded_raw)

        relevant_raw = generate(RELEVANCE_PROMPT.format(question=q["question"], answer=answer))
        relevant, _ = parse_verdict(relevant_raw)

        grounded_verdicts.append(grounded)
        relevant_verdicts.append(relevant)
        per_question_rows.append(
            {
                "id": q["id"],
                "question": q["question"],
                "answer": answer,
                "grounded": grounded,
                "grounded_raw": grounded_raw,
                "relevant": relevant,
                "relevant_raw": relevant_raw,
            }
        )

    g_rate = groundedness_rate(grounded_verdicts)
    h_rate = hallucination_rate(grounded_verdicts)
    r_rate = groundedness_rate(relevant_verdicts)  # mismo cálculo, distinto verdict list
    g_ci = bootstrap_ci([1.0 if v else 0.0 for v in grounded_verdicts])
    r_ci = bootstrap_ci([1.0 if v else 0.0 for v in relevant_verdicts])
    lat = latency_summary(gen_latencies)

    with tracked_run(
        "evaluation",
        "groundedness",
        {
            "base_method": BEST_METHOD,
            "use_reranker": USE_RERANKER,
            "llm_backend": __import__("shared.settings", fromlist=["settings"]).settings.llm_backend,
            **BEST_CHUNKING,
        },
    ):
        log_metrics(
            {
                "groundedness_rate": g_rate,
                "hallucination_rate": h_rate,
                "answer_relevance_rate": r_rate,
                **lat,
            }
        )

    (OUT_DIR / "per_question_results.json").write_text(
        json.dumps(per_question_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    results = {
        "chunking_config": BEST_CHUNKING,
        "base_method": BEST_METHOD,
        "use_reranker": USE_RERANKER,
        "n_questions": len(questions),
        "groundedness_rate": round(g_rate, 4),
        "groundedness_rate_ci95": [round(g_ci[0], 4), round(g_ci[1], 4)],
        "hallucination_rate": round(h_rate, 4),
        "answer_relevance_rate": round(r_rate, 4),
        "answer_relevance_rate_ci95": [round(r_ci[0], 4), round(r_ci[1], 4)],
        **lat,
    }

    (OUT_DIR / "results").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results" / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"\ngroundedness_rate={g_rate:.3f} hallucination_rate={h_rate:.3f} "
        f"answer_relevance_rate={r_rate:.3f} -> guardado en evaluation/results/results.json"
    )


if __name__ == "__main__":
    main()
