import os
import sys
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from experiments.chunking.run import get_tables_from_docs
from experiments.evaluation.prompts import ANSWER_PROMPT
from experiments.evaluation.run import (
    BEST_CHUNKING, BEST_METHOD, COLLECTION, POOL_SIZE, TOP_K, USE_RERANKER,
    build_context_text, retrieve_context,
)
from shared.eval.data import load_questions
from shared.ingest import build_qdrant_client, chunk_documents, index_chunks, load_corpus
from shared.retrieval import BM25Index, fetch_all_chunks
from shared.settings import settings

OUT_DIR = Path(__file__).resolve().parent
GENERATOR = "llama3.1:8b-instruct-q4_K_M"
NUM_CTX = 4096  # el mismo que fija shared/llm.py

def prompt_tokens(prompt: str) -> int:

    resp = requests.post(
        f"{settings.llm_host}/api/generate",
        json={
            "model": GENERATOR,
            "prompt": prompt,
            "stream": False,
            # num_ctx alto solo para contar: con 4096 Ollama trunca y el conteo saldría recortado
            "options": {
                "temperature": 0.0, 
                "num_ctx": 16384, 
                "num_predict": 1
            },
        },
        timeout=300
    )
    resp.raise_for_status()

    return resp.json()["prompt_eval_count"]

def main():

    docs = load_corpus()
    questions = load_questions()
    client = build_qdrant_client()

    tables = [
        t for doc in docs
            for t in get_tables_from_docs(doc)[0]
    ]

    chunks = chunk_documents(
        docs, 
        tables, 
        BEST_CHUNKING["chunk_size"], 
        BEST_CHUNKING["chunk_overlap"]
    )

    index_chunks(client, chunks, COLLECTION, BEST_CHUNKING)

    bm25 = BM25Index(fetch_all_chunks(client, COLLECTION))

    print(f"chunking={BEST_CHUNKING} method={BEST_METHOD} reranker={USE_RERANKER} pool={POOL_SIZE} top_k={TOP_K}")

    rows = []

    for q in questions:

        context_chunks = retrieve_context(client, bm25, q["question"])
        context_text = build_context_text(context_chunks)
        prompt = ANSWER_PROMPT.format(context=context_text, question=q["question"])
        n_tokens = prompt_tokens(prompt)

        rows.append({
            "id": q["id"],
            "n_chunks": len(context_chunks),
            "n_chunks_with_tables": sum(1 for c in context_chunks if c.get("tables")),
            "n_tables": sum(len(c.get("tables") or []) for c in context_chunks),
            "context_chars": len(context_text),
            "prompt_tokens": n_tokens,
            "exceeds_num_ctx": n_tokens > NUM_CTX,
        })
        print(rows[-1], flush=True)

    pd.DataFrame(rows).to_csv(OUT_DIR / "context_stats.csv", index=False, sep=";")


if __name__ == "__main__":
    main()
