from collections import Counter
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError

from shared.db import build_engine, corridas_eval, veredictos
from shared.eval.data import load_answers

AGENT_RESULTS = Path("experiments/agent/results/exp_2")


def loaded_verdicts() -> Counter:

    query = (
        select(
            corridas_eval.c.metodo,
            veredictos.c.fuente,
            veredictos.c.veredicto,
            func.count(),
        )
            .join(corridas_eval, corridas_eval.c.id == veredictos.c.corrida_id)
            .where(corridas_eval.c.experimento == "exp_2")
            .group_by(corridas_eval.c.metodo, veredictos.c.fuente, veredictos.c.veredicto)
    )

    try:
        with build_engine().connect() as c:
            rows = c.execute(query).all()

    except OperationalError:
        pytest.skip("Postgres de docker-compose-dev.yml no está levantado")

    return Counter({(lado, fuente, veredicto): n for lado, fuente, veredicto, n in rows})


def test_loaded_verdicts_match_the_labels_file():

    labels = load_answers(AGENT_RESULTS / "verdicts_manual.csv")

    expected = Counter(
        (r["lado"], fuente, r[columna])
            for r in labels.to_dict(orient="records")
                for fuente, columna in (("juez", "veredicto"), ("humano", "veredicto_manual"))
    )

    assert loaded_verdicts() == expected
