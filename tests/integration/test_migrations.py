import uuid

import pytest
from alembic import command
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

TABLES = {
    "documentos",
    "chunks",
    "preguntas",
    "pregunta_documentos",
    "corridas_eval",
    "resultados_pregunta",
    "veredictos",
}


def tables(engine) -> set[str]:

    return set(inspect(engine).get_table_names()) - {"alembic_version"}


def test_upgrade_creates_the_tables_and_downgrade_removes_them(db):

    config, engine = db

    command.upgrade(config, "head")
    assert tables(engine) == TABLES

    command.downgrade(config, "base")
    assert tables(engine) == set()


def test_a_document_with_chunks_cannot_be_deleted(db):

    config, engine = db
    command.upgrade(config, "head")

    with engine.begin() as c:
        doc = c.execute(
            text(
                "INSERT INTO documentos (coleccion, nombre, sha256) "
                "VALUES ('propia', 'music-tagger/a.md', 'h1') RETURNING id"
            )
        ).scalar()

        c.execute(
            text(
                "INSERT INTO chunks (id, documento_id, idx, texto, n_tokens) "
                "VALUES (:id, :doc, 0, 'texto', 3)"
            ),
            {"id": uuid.uuid4(), "doc": doc},
        )

    with pytest.raises(IntegrityError), engine.begin() as c:
        c.execute(text("DELETE FROM documentos WHERE id = :doc"), {"doc": doc})


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO documentos (coleccion, nombre, sha256) VALUES ('otra', 'a.md', 'h1')",
        "INSERT INTO chunks (id, documento_id, idx, texto, n_tokens) "
        "VALUES (gen_random_uuid(), gen_random_uuid(), 0, 'texto', 3)",
        "INSERT INTO preguntas (id, biblioteca, tipo, pregunta, respuesta_esperada, gold_hash) "
        "VALUES ('q001', 'music-tagger', 'abierta', '¿?', 'r', 'h')",
    ],
    ids=["coleccion_desconocida", "documento_inexistente", "tipo_desconocido"],
)
def test_constraints_reject_invalid_rows(db, statement):

    config, engine = db
    command.upgrade(config, "head")

    with pytest.raises(IntegrityError), engine.begin() as c:
        c.execute(text(statement))
