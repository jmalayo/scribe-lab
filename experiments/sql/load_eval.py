import hashlib
import json
import os
import sys
from pathlib import Path

from sqlalchemy import select

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from shared.db import (
    build_engine,
    chunks,
    documentos,
    pregunta_documentos,
    preguntas,
    upsert,
)
from shared.eval.data import load_questions
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


def sha256(text: str) -> str:

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def served_chunks(config: PipelineConfig) -> tuple[list, list]:

    docs = load_corpus(config.corpus_dir)

    tables = [
        t for doc in docs
            for t in get_tables_from_docs(doc)[0]
    ]

    return docs, content_chunk_ids(chunk_documents(docs, tables, config.chunk_size, config.chunk_overlap))


def main() -> dict:

    config = PipelineConfig.from_yaml(settings.pipeline_config_path)
    docs, served = served_chunks(config)
    questions = load_questions()
    tokenizer = get_embedder().tokenizer

    with build_engine().begin() as conn:
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
                    "gold_hash": sha256(json.dumps(q["gold_spans"], ensure_ascii=False)),
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

    print(json.dumps(counts, indent=2))

    return counts


if __name__ == "__main__":
    main()
