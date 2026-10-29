import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import delete, select

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import mlflow

from shared.db import (
    build_engine,
    chunks,
    corridas_eval,
    documentos,
    pregunta_documentos,
    preguntas,
    resultados_pregunta,
    upsert,
    veredictos,
)
from shared.eval.data import load_answers, load_questions
from shared.ingest import (
    chunk_documents,
    content_chunk_ids,
    get_embedder,
    get_tables_from_docs,
    load_corpus,
)
from shared.pipeline import PipelineConfig
from shared.settings import settings

COLECCION = "propia"

AGENT_EXPERIMENT = f"{settings.mlflow_experiment_prefix}-agent"
AGENT_EXP = "exp_2"
AGENT_RESULTS = REPO_ROOT / "experiments" / "agent" / "results" / AGENT_EXP
AGENT_SIDES = ("pipeline", "agent")


def sha256(text: str) -> str:

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def gold_hash(question: dict) -> str:

    return sha256(json.dumps(question["gold_spans"], ensure_ascii=False))


def served_chunks(config: PipelineConfig) -> tuple[list, list]:

    docs = load_corpus(config.corpus_dir)

    tables = [
        t for doc in docs
            for t in get_tables_from_docs(doc)[0]
    ]

    return docs, content_chunk_ids(chunk_documents(docs, tables, config.chunk_size, config.chunk_overlap))


def load_corpus_and_questions(conn, questions: list[dict]) -> dict:

    config = PipelineConfig.from_yaml(settings.pipeline_config_path)
    docs, served = served_chunks(config)
    tokenizer = get_embedder().tokenizer

    counts = {
        "documentos": upsert(
            conn,
            documentos,
            [
                {"coleccion": COLECCION, "nombre": d.doc_id, "sha256": sha256(d.text)}
                    for d in docs
            ],
            ["sha256"],
        )
    }

    doc_ids = dict(
        conn.execute(
            select(documentos.c.nombre, documentos.c.id).where(documentos.c.coleccion == COLECCION)
        ).all()
    )

    conn.execute(delete(chunks).where(chunks.c.documento_id.in_(doc_ids.values())))

    counts["chunks"] = upsert(
        conn,
        chunks,
        [
            {
                "id": c.chunk_id,
                "documento_id": doc_ids[c.doc_id],
                "idx": c.chunk_index,
                "texto": c.text,
                "n_tokens": len(tokenizer.encode(c.text, add_special_tokens=False)),
            }
                for c in served
        ],
        ["id"],
    )

    counts["preguntas"] = upsert(
        conn,
        preguntas,
        [
            {
                "id": q["id"],
                "biblioteca": q["library"],
                "tipo": q["tipo"],
                "pregunta": q["question"],
                "respuesta_esperada": q["expected_answer"],
                "gold_hash": gold_hash(q),
            }
                for q in questions
        ],
        ["id"],
    )

    counts["pregunta_documentos"] = upsert(
        conn,
        pregunta_documentos,
        [
            {"pregunta_id": q["id"], "documento_id": doc_ids[name]}
                for q in questions
                    if q["tipo"] != "sin_respuesta"
                        for name in q["source_doc"].split("; ")
        ],
        ["pregunta_id", "documento_id"],
    )

    return counts


def agent_runs() -> dict[str, dict]:

    runs = mlflow.search_runs(
        experiment_names=[AGENT_EXPERIMENT],
        filter_string=f"params.exp = '{AGENT_EXP}' and status = 'FINISHED'",
        output_format="list",
    )

    return {
        run.data.tags["mlflow.runName"].removeprefix(f"{AGENT_EXP}-"): run
            for run in runs
    }


def load_agent_benchmark(conn, questions: list[dict]) -> dict:

    hashes = {q["id"]: gold_hash(q) for q in questions}
    runs = agent_runs()

    corridas = upsert(
        conn,
        corridas_eval,
        [
            {
                "experimento": AGENT_EXP,
                "metodo": side,
                "config": runs[side].data.params,
                "git_commit": runs[side].data.tags["mlflow.source.git.commit"],
                "mlflow_run_id": runs[side].info.run_id,
                "creada_en": datetime.fromtimestamp(runs[side].info.start_time / 1000, UTC),
            }
                for side in AGENT_SIDES
        ],
        ["mlflow_run_id"],
    )

    corrida_ids = dict(
        conn.execute(
            select(corridas_eval.c.metodo, corridas_eval.c.id).where(corridas_eval.c.experimento == AGENT_EXP)
        ).all()
    )

    answers = {
        side: load_answers(AGENT_RESULTS / f"answers_{side}.csv")
            for side in AGENT_SIDES
    }

    resultados = upsert(
        conn,
        resultados_pregunta,
        [
            {
                "corrida_id": corrida_ids[side],
                "pregunta_id": r["id"],
                "gold_hash": hashes[r["id"]],
                "respuesta": r["answer"],
                "latencia_ms": r["latency_ms"],
                "tokens": r["tokens"],
                "gold_spans_at_10": r.get("gold_spans_at_10"),
            }
                for side, df in answers.items()
                    for r in df.to_dict(orient="records")
        ],
        ["corrida_id", "pregunta_id"],
    )

    labels = load_answers(AGENT_RESULTS / "verdicts_manual.csv")

    verdicts = upsert(
        conn,
        veredictos,
        [
            {
                "corrida_id": corrida_ids[r["lado"]],
                "pregunta_id": r["id"],
                "fuente": fuente,
                "veredicto": r[veredicto],
                "motivo": r[motivo],
            }
                for r in labels.to_dict(orient="records")
                    for fuente, veredicto, motivo in (
                        ("juez", "veredicto", "motivo"),
                        ("humano", "veredicto_manual", "motivo_manual"),
                    )
        ],
        ["corrida_id", "pregunta_id", "fuente"],
    )

    return {
        "corridas_eval": corridas,
        "resultados_pregunta": resultados,
        "veredictos": verdicts,
    }


def main() -> dict:

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    questions = load_questions()

    with build_engine().begin() as conn:
        counts = load_corpus_and_questions(conn, questions) | load_agent_benchmark(conn, questions)

    print(json.dumps(counts, indent=2))

    return counts


if __name__ == "__main__":
    main()
