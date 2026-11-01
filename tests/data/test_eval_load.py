from collections import Counter
from pathlib import Path

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import OperationalError

from shared.db import build_engine, corridas_eval, veredictos
from shared.eval.data import load_answers
from shared.eval.metrics import cohen_kappa

AGENT_RESULTS = Path("experiments/agent/results/exp_2")


@pytest.fixture
def conn():

    try:
        with build_engine().connect() as c:
            yield c

    except OperationalError:
        pytest.skip("Postgres de docker-compose-dev.yml no está levantado")


def loaded_verdicts(conn) -> Counter:

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

    rows = conn.execute(query).all()

    return Counter({(lado, fuente, veredicto): n for lado, fuente, veredicto, n in rows})


def test_loaded_verdicts_match_the_labels_file(conn):

    labels = load_answers(AGENT_RESULTS / "verdicts_manual.csv")

    expected = Counter(
        (r["lado"], fuente, r[columna])
            for r in labels.to_dict(orient="records")
                for fuente, columna in (("juez", "veredicto"), ("humano", "veredicto_manual"))
    )

    assert loaded_verdicts(conn) == expected


def test_kappa_view_matches_the_labels_file(conn):

    labels = load_answers(AGENT_RESULTS / "verdicts_manual.csv")
    expected = cohen_kappa(labels["veredicto_manual"].tolist(), labels["veredicto"].tolist())

    kappa = conn.execute(
        text("SELECT kappa FROM v_acuerdo_juez_humano WHERE experimento = 'exp_2' AND metodo = 'todos'")
    ).scalar()

    assert float(kappa) == pytest.approx(expected, abs=1e-3)


def test_multiturn_sets_match_their_files(conn):

    import json

    from shared.db import conversaciones_eval, turnos_eval

    expected = {}

    for path in sorted(Path("shared/eval/multiturn").glob("multiturno_*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        expected[path.stem.removeprefix("multiturno_")] = (len(rows), sum(len(r["turnos"]) for r in rows))

    loaded = {}

    for table, position in ((conversaciones_eval, 0), (turnos_eval, 1)):
        for conjunto, count in conn.execute(select(table.c.conjunto, func.count()).group_by(table.c.conjunto)):
            loaded.setdefault(conjunto, [0, 0])[position] = count

    if not loaded:
        pytest.skip("los sets multiturno no están cargados (experiments/sql/load_eval.py)")

    assert {k: tuple(v) for k, v in loaded.items()} == expected
