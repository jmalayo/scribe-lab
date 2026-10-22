from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import func, select

from shared.db import documentos, metadata, preguntas, upsert

PREGUNTA = {
    "id": "q001",
    "biblioteca": "music-tagger",
    "tipo": "simple",
    "pregunta": "¿Qué modelo?",
    "respuesta_esperada": "r",
    "gold_hash": "h1",
}


def test_declared_tables_match_the_migrations(db):

    config, engine = db
    command.upgrade(config, "head")

    with engine.connect() as c:
        assert compare_metadata(MigrationContext.configure(c), metadata) == []


def test_upsert_twice_keeps_one_row_with_the_latest_values(db):

    config, engine = db
    command.upgrade(config, "head")

    with engine.begin() as c:
        upsert(c, preguntas, [PREGUNTA], ["id"])
        upsert(c, preguntas, [{**PREGUNTA, "gold_hash": "h2"}], ["id"])

    with engine.connect() as c:
        assert c.execute(select(preguntas.c.gold_hash)).scalars().all() == ["h2"]


def test_upsert_splits_large_loads_into_batches(db, monkeypatch):

    config, engine = db
    command.upgrade(config, "head")
    monkeypatch.setattr("shared.db.BATCH_SIZE", 2)

    rows = [
        {"coleccion": "propia", "nombre": f"d{i}.md", "sha256": f"h{i}"}
            for i in range(5)
    ]

    with engine.begin() as c:
        assert upsert(c, documentos, rows, ["sha256"]) == 5
        upsert(c, documentos, rows, ["sha256"])

    with engine.connect() as c:
        assert c.execute(select(func.count()).select_from(documentos)).scalar() == 5
