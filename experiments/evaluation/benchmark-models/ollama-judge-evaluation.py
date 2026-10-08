"""Genera respuestas base con el pipeline vigente y, opcionalmente, las juzga con jueces de Ollama.

El contexto se arma igual que en `experiments/evaluation/run.py` (chunking y método elegidos en
MLflow, tablas del payload, reranker, `TOP_K` chunks). Se configura con las variables de abajo.
Salida en `results/<EXP>/<TAG>/`: `base_answers.csv` y, con `--use-judge True`,
`per_question_judge_verdicts.csv` y `results_summary.csv`.

    python experiments/evaluation/benchmark-models/ollama-judge-evaluation.py --use-judge False
    python experiments/evaluation/benchmark-models/ollama-judge-evaluation.py --use-judge True
"""

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from experiments.chunking.run import get_tables_from_docs
from experiments.evaluation.prompts import (
    ANSWER_PROMPT,
    GROUNDEDNESS_PROMPT,
    RELEVANCE_PROMPT,
)
from experiments.evaluation.run import (
    BEST_CHUNKING,
    COLLECTION,
    build_context_text,
    retrieve_context,
)
from shared.eval.data import load_questions
from shared.eval.judge import agreement_rate, parse_verdict
from shared.eval.metrics import latency_summary
from shared.ingest import (
    build_qdrant_client,
    chunk_documents,
    index_chunks,
    load_corpus,
)
from shared.llm import generate
from shared.retrieval import BM25Index, fetch_all_chunks
from shared.settings import settings

RESULTS_DIR = Path(__file__).resolve().parent / "results"

EXP = "exp_7"
TAG = "llama31_base"

ANSWER_MODEL = "llama3.1:8b-instruct-q4_K_M"
NUM_PREDICT = 500        # defaults de shared.llm.generate; exp_6 usó 1500 / 1.3 con DeepSeek
REPEAT_PENALTY = 1.0

# True: juzga el base_answers.csv existente en vez de generarlo
REUSE_ANSWERS = True

JUDGES = {
    "deepseek": settings.llm_model_judge.strip(),
    "llama": settings.llm_model_base.strip(),
}

# opciones de los jueces de exp_5/exp_6
JUDGE_NUM_PREDICT = 1500
JUDGE_REPEAT_PENALTY = 1.3

def build_answers(questions: list[dict], answer_model: str, answers_path: Path,
                  num_predict: int, repeat_penalty: float) -> list[dict]:

    docs = load_corpus()
    client = build_qdrant_client()

    tables = [t for doc in docs for t in get_tables_from_docs(doc)[0]]
    chunks = chunk_documents(docs, tables, BEST_CHUNKING["chunk_size"], BEST_CHUNKING["chunk_overlap"])

    index_chunks(client, chunks, COLLECTION, BEST_CHUNKING)

    bm25 = BM25Index(
        fetch_all_chunks(client, COLLECTION)
    )

    rows = []

    for i, q in enumerate(questions, start=1):

        context_text = build_context_text(retrieve_context(client, bm25, q["question"]))

        t0 = time.perf_counter()
        answer = generate(
            ANSWER_PROMPT.format(context=context_text, question=q["question"]),
            model=answer_model,
            num_predict=num_predict,
            repeat_penalty=repeat_penalty,
        )
        elapsed = time.perf_counter() - t0

        print(f"  [{i}/{len(questions)}] {q['id']} ({answer_model}): {elapsed:.1f}s", flush=True)

        rows.append(
            {
                "id": q["id"],
                "question": q["question"],
                "context": context_text,
                "answer": answer,
            }
        )

        pd.DataFrame(rows).to_csv(
            answers_path,
            index=False,
            encoding="utf-8",
            sep=";",
        )

    return rows

def judge_answers(rows: list[dict], judge_model: str) -> tuple[list[dict], list[float]]:

    verdicts, latencies = [], []

    for i, row in enumerate(rows, start=1):

        t0 = time.perf_counter()

        grounded_raw = generate(
            GROUNDEDNESS_PROMPT.format(context=row["context"], answer=row["answer"]),
            model=judge_model,
            num_predict=JUDGE_NUM_PREDICT,
            repeat_penalty=JUDGE_REPEAT_PENALTY,
        )
        relevant_raw = generate(
            RELEVANCE_PROMPT.format(question=row["question"], answer=row["answer"]),
            model=judge_model,
            num_predict=JUDGE_NUM_PREDICT,
            repeat_penalty=JUDGE_REPEAT_PENALTY,
        )

        elapsed = time.perf_counter() - t0
        latencies.append(elapsed * 1000)

        print(f"  [{i}/{len(rows)}] {row['id']} ({judge_model}): {elapsed:.1f}s", flush=True)

        grounded_verdict, grounded_valid = parse_verdict(grounded_raw)
        relevant_verdict, relevant_valid = parse_verdict(relevant_raw)

        verdicts.append(
            {
                "id": row["id"],
                "grounded": grounded_verdict,
                "relevant": relevant_verdict,
                "asw_valido": grounded_valid and relevant_valid,
            }
        )

    return verdicts, latencies


def summarize(judge_name: str, judge_model: str, answer_model: str,
              verdicts: list[dict], latencies: list[float]) -> dict:

    grounded = [v["grounded"] for v in verdicts]
    relevant = [v["relevant"] for v in verdicts]
    g_rate = sum(grounded) / len(grounded) if grounded else 0.0
    r_rate = sum(relevant) / len(relevant) if relevant else 0.0

    mode = "self-judging" if judge_model == answer_model else "cross-judging"

    return {
        "judge": judge_name,
        "judge_model": judge_model,
        "mode": mode,
        "groundedness_rate": round(g_rate, 4),
        "hallucination_rate": round(1 - g_rate, 4),
        "answer_relevance_rate": round(r_rate, 4),
        **latency_summary(latencies),
    }


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument("--use-judge", required=True, choices=["True", "False"],
                        help="False: solo genera base_answers.csv; True: además lo juzga con JUDGES")

    return parser.parse_args()

def main():

    use_judge = parse_args().use_judge == "True"

    questions = load_questions()

    out_dir = RESULTS_DIR / EXP / TAG
    answers_path = out_dir / "base_answers.csv"

    if REUSE_ANSWERS:
        print(f"Reusando respuestas base en {answers_path}", flush=True)

        answers_df = pd.read_csv(answers_path, encoding="utf-8", sep=";").fillna({"answer": ""})
        rows = answers_df.to_dict("records")

    else:
        if answers_path.exists():
            raise SystemExit(f"{answers_path} ya existe: usar REUSE_ANSWERS = True u otro EXP/TAG")

        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generando {len(questions)} respuestas base con {ANSWER_MODEL}...", flush=True)

        rows = build_answers(questions, ANSWER_MODEL, answers_path, NUM_PREDICT, REPEAT_PENALTY)

    if not use_judge:
        print(f"\n{len(rows)} respuestas -> {answers_path}")
        return

    summaries = []
    verdicts_by_judge = {}

    for judge_name, judge_model in JUDGES.items():

        print(f"Juzgando respuestas con {judge_name} ({judge_model})...", flush=True)

        verdicts, latencies = judge_answers(rows, judge_model)

        verdicts_by_judge[judge_name] = verdicts

        summaries.append(summarize(judge_name, judge_model, ANSWER_MODEL, verdicts, latencies))

    comparison_df = pd.DataFrame(rows)[["id", "question", "answer"]]

    for judge_name, verdicts in verdicts_by_judge.items():

        judge_df = pd.DataFrame(verdicts).rename(
            columns={
                "grounded": f"grounded_{judge_name}",
                "relevant": f"relevant_{judge_name}",
                "asw_valido": f"{judge_name}_asw_valido",
            }
        )

        comparison_df = comparison_df.merge(judge_df, on="id")

    comparison_df.to_csv(
        out_dir / "per_question_judge_verdicts.csv",
        index=False,
        encoding="utf-8",
        sep=";",
    )

    summary_df = pd.DataFrame(summaries)
    summary_df.insert(0, "base_model", ANSWER_MODEL)
    summary_df.insert(1, "n_questions", len(rows))

    summary_df.to_csv(
        out_dir / "results_summary.csv",
        index=False,
        encoding="utf-8",
        sep=";",
    )

    header = f"{'Modo':<16}{'Modelo juez':<22}{'Grounded':>10}{'Halluc.':>10}{'Relevant':>10}{'p50 ms':>10}"

    print("\n" + header)
    print("-" * len(header))

    for s in summaries:
        print(
            f"{s['mode']:<16}{s['judge_model']:<22}{s['groundedness_rate']:>10.1%}"
            f"{s['hallucination_rate']:>10.1%}{s['answer_relevance_rate']:>10.1%}{s['p50_ms']:>10.1f}"
        )

    print("-" * len(header))

    for (name_a, va), (name_b, vb) in itertools.combinations(verdicts_by_judge.items(), 2):
        print(
            f"Acuerdo {name_a} vs {name_b} (mismo set de respuestas) -> "
            f"groundedness: {agreement_rate(va, vb, 'grounded'):.1%} | "
            f"relevance: {agreement_rate(va, vb, 'relevant'):.1%}"
        )

if __name__ == "__main__":
    main()
