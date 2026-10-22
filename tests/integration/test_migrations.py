import uuid

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError, OperationalError

from shared.db import build_engine
from shared.settings import settings

TEST_DB = "scribe_test"
TABLES = {"documentos", "chunks", "preguntas"}


@pytest.fixture
def db():

    admin = build_engine().execution_options(isolation_level="AUTOCOMMIT")

    try:
        with admin.connect() as c:
            c.execute(text(f"DROP DATABASE IF EXISTS {TEST_DB}"))
            c.execute(text(f"CREATE DATABASE {TEST_DB}"))

    except OperationalError:
        pytest.skip("Postgres de docker-compose-dev.yml no está levantado")

    url = make_url(settings.database_url).set(database=TEST_DB).render_as_string(hide_password=False)

    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", url)

    engine = build_engine(url)

    yield config, engine

    engine.dispose()

    with admin.connect() as c:
        c.execute(text(f"DROP DATABASE {TEST_DB} WITH (FORCE)"))

    admin.dispose()


def tables(engine) -> set[str]:

    return set(inspect(engine).get_table_names()) - {"alembic_version"}


def test_upgrade_creates_the_base_tables_and_downgrade_removes_them(db):

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
