import os
import subprocess
import sys
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError, ProgrammingError

from shared.db import build_engine, build_writer_engine, insert_query
from shared.settings import settings

TEST_DB = "scribe_test_servicio"


@pytest.fixture(scope="module")
def service_db():

    admin_url = make_url(settings.database_url)

    try:
        admin = build_engine(admin_url.render_as_string(hide_password=False)).execution_options(isolation_level="AUTOCOMMIT")

        with admin.connect() as conn:
            conn.execute(text(f"DROP DATABASE IF EXISTS {TEST_DB}"))
            conn.execute(text(f"CREATE DATABASE {TEST_DB}"))

    except OperationalError:
        pytest.skip("Postgres de docker-compose-dev.yml no está levantado")

    url = admin_url.set(database=TEST_DB).render_as_string(hide_password=False)
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "servicio@head"],
        check=True,
        env={**os.environ, "DATABASE_URL": url},
    )

    yield url

    with admin.connect() as conn:
        conn.execute(text(f"DROP DATABASE {TEST_DB} WITH (FORCE)"))

    admin.dispose()


def test_service_branch_creates_only_service_tables(service_db):

    with build_engine(service_db).connect() as conn:
        tables = set(
            conn.execute(
                text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
            ).scalars()
        )

    assert tables == {
        "alembic_version",
        "api_keys",
        "sesiones",
        "consultas",
        "consulta_chunks",
        "feedback",
        "v_latencia_movil",
        "v_errores_por_dia",
        "v_drift_semanal",
        "v_sesiones",
        "v_rachas_sin_grounding",
    }


def test_writer_records_a_query_and_cannot_read_it_back(service_db):

    consulta_id = uuid.uuid4()
    sesion_id = uuid.uuid4()
    row = {
        "id": consulta_id,
        "request_id": "rid-db",
        "sesion_id": sesion_id,
        "turno": 1,
        "pregunta": "¿qué pasó?",
        "respuesta": "respuesta",
        "estado": 200,
        "abstencion": False,
        "latencia_total_ms": 1200.0,
        "pipeline_version": "test",
        "prompt_version": "v1",
    }
    chunk_rows = [
        {
            "consulta_id": consulta_id,
            "posicion": position,
            "chunk_id": f"c{position}",
            "doc_id": "doc.md",
            "score": 1 / position,
        }
            for position in range(1, 11)
    ]

    writer = build_writer_engine(service_db)
    insert_query(writer, row, chunk_rows)
    insert_query(writer, {**row, "id": uuid.uuid4(), "turno": 2}, [])

    with pytest.raises(ProgrammingError, match="permission denied"), writer.connect() as conn:
        conn.execute(text("SELECT count(*) FROM consultas"))

    writer.dispose()

    with build_engine(service_db).connect() as conn:
        counts = conn.execute(
            text(
                "SELECT"
                " (SELECT count(*) FROM consultas WHERE id = :id),"
                " (SELECT count(*) FROM consulta_chunks WHERE consulta_id = :id),"
                " (SELECT consultas FROM v_sesiones WHERE sesion_id = :sesion)"
            ),
            {"id": consulta_id, "sesion": sesion_id},
        ).one()

    assert tuple(counts) == (1, 10, 2)
