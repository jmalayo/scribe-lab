from datetime import UTC, datetime

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import func, insert, select
from sqlalchemy.exc import IntegrityError

from shared.db import (
    corridas_eval,
    documentos,
    metadata,
    preguntas,
    resultados_pregunta,
    upsert,
    veredictos,
)

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


@pytest.fixture
def resultado(db):

    config, engine = db
    command.upgrade(config, "head")

    with engine.begin() as c:
        upsert(c, preguntas, [PREGUNTA], ["id"])

        corrida = c.execute(
            insert(corridas_eval)
                .values(
                    experimento="exp_2",
                    metodo="pipeline",
                    config={"model": "m"},
                    git_commit="abc",
                    mlflow_run_id="r1",
                    creada_en=datetime.now(UTC),
                )
                .returning(corridas_eval.c.id)
        ).scalar()

        c.execute(
            insert(resultados_pregunta).values(
                corrida_id=corrida,
                pregunta_id="q001",
                gold_hash="h1",
                respuesta="r",
                latencia_ms=1.0,
                tokens=10,
            )
        )

    return engine, corrida


@pytest.mark.parametrize(
    ("fuente", "veredicto", "pregunta"),
    [
        ("humano", "dudosa", "q001"),
        ("lector", "correcta", "q001"),
        ("humano", "correcta", "q999"),
    ],
    ids=["veredicto_desconocido", "fuente_desconocida", "sin_resultado"],
)
def test_verdicts_reject_unknown_labels_and_answers_without_a_result(resultado, fuente, veredicto, pregunta):

    engine, corrida = resultado

    with pytest.raises(IntegrityError), engine.begin() as c:
        c.execute(
            insert(veredictos).values(
                corrida_id=corrida,
                pregunta_id=pregunta,
                fuente=fuente,
                veredicto=veredicto,
                motivo="m",
            )
        )
