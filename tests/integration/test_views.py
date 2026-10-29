from datetime import UTC, datetime

import pytest
from alembic import command
from sqlalchemy import insert, text

from shared.db import corridas_eval, preguntas, resultados_pregunta, upsert, veredictos
from shared.eval.metrics import cohen_kappa

VIEWS = [
    "v_metricas_por_metodo",
    "v_peores_preguntas",
    "v_ultima_corrida",
    "v_gold_nunca_recuperado",
    "v_calidad_chunks",
    "v_acuerdo_juez_humano",
]

HUMANO = ["correcta", "correcta", "parcial", "incorrecta", "abstencion", "parcial"]
JUEZ = ["correcta", "parcial", "parcial", "incorrecta", "correcta", "correcta"]


@pytest.fixture
def cargada(db):

    config, engine = db
    command.upgrade(config, "head")

    ids = [f"q{i:03d}" for i in range(1, len(HUMANO) + 1)]

    with engine.begin() as c:
        upsert(
            c,
            preguntas,
            [
                {
                    "id": q,
                    "biblioteca": "music-tagger",
                    "tipo": "simple",
                    "pregunta": "¿?",
                    "respuesta_esperada": "r",
                    "gold_hash": "h",
                }
                    for q in ids
            ],
            ["id"],
        )

        corrida = c.execute(
            insert(corridas_eval)
                .values(
                    experimento="exp_t",
                    metodo="pipeline",
                    config={},
                    git_commit="abc",
                    mlflow_run_id="r1",
                    creada_en=datetime.now(UTC),
                )
                .returning(corridas_eval.c.id)
        ).scalar()

        c.execute(
            insert(resultados_pregunta),
            [
                {
                    "corrida_id": corrida,
                    "pregunta_id": q,
                    "gold_hash": "h",
                    "respuesta": "r",
                    "latencia_ms": 1.0,
                    "tokens": 10,
                    "gold_spans_at_10": i % 2,
                }
                    for i, q in enumerate(ids)
            ],
        )

        c.execute(
            insert(veredictos),
            [
                {
                    "corrida_id": corrida,
                    "pregunta_id": q,
                    "fuente": fuente,
                    "veredicto": veredicto,
                    "motivo": "m",
                }
                    for q, humano, juez in zip(ids, HUMANO, JUEZ, strict=True)
                        for fuente, veredicto in (("humano", humano), ("juez", juez))
            ],
        )

    return engine


@pytest.mark.parametrize("view", VIEWS)
def test_every_view_runs_after_the_migrations(cargada, view):

    with cargada.connect() as c:
        c.execute(text(f"SELECT * FROM {view}")).all()


def test_kappa_in_sql_matches_the_python_metric(cargada):

    with cargada.connect() as c:
        kappa = c.execute(
            text("SELECT kappa FROM v_acuerdo_juez_humano WHERE metodo = 'todos'")
        ).scalar()

    assert float(kappa) == pytest.approx(cohen_kappa(HUMANO, JUEZ), abs=1e-3)


def test_questions_without_any_retrieved_gold_span_are_listed(cargada):

    with cargada.connect() as c:
        listed = c.execute(text("SELECT pregunta_id FROM v_gold_nunca_recuperado ORDER BY 1")).scalars().all()

    assert listed == ["q001", "q003", "q005"]


def test_downgrade_drops_the_views(cargada, db):

    config, _ = db
    command.downgrade(config, "0007_veredictos")

    with cargada.connect() as c:
        names = c.execute(text("SELECT table_name FROM information_schema.views WHERE table_schema = 'public'")).scalars().all()

    assert names == []
