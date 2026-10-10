import pandas as pd

from shared.eval.data import load_answers, load_chunks


def _write(path, rows):
    pd.DataFrame(rows).to_csv(path, sep=";", index=False, encoding="utf-8")


def _files(tmp_path):
    answers, chunks = tmp_path / "base_answers.csv", tmp_path / "base_chunks.csv"
    _write(answers, [
        {"id": "q1", "question": "¿A?", "answer": "Sí; con\nsalto"},
        {"id": "q2", "question": "¿B?", "answer": ""},
    ])
    _write(chunks, [
        {"id": "q1", "context": "[a/x.md] uno\n\n| t | 1 |"},
        {"id": "q2", "context": "[c.md] tres"},
        {"id": "q1", "context": "[b.md] dos"},
    ])
    return answers, chunks


def test_load_chunks_agrupa_por_id_en_orden_de_archivo(tmp_path):
    _, chunks = _files(tmp_path)

    assert load_chunks(chunks) == {
        "q1": ["[a/x.md] uno\n\n| t | 1 |", "[b.md] dos"],
        "q2": ["[c.md] tres"],
    }


def test_load_answers_con_chunks_arma_el_contexto(tmp_path):
    answers, chunks = _files(tmp_path)

    assert load_answers(answers, chunks).to_dict("records") == [
        {"id": "q1", "question": "¿A?", "answer": "Sí; con\nsalto", "context": "[a/x.md] uno\n\n| t | 1 |\n\n[b.md] dos"},
        {"id": "q2", "question": "¿B?", "answer": "", "context": "[c.md] tres"},
    ]


def test_load_answers_sin_chunks_se_devuelve_tal_cual(tmp_path):
    path = tmp_path / "base_answers.csv"
    context = "[a/x.md] uno\n\n| t | 1 |\n\n[b.md] dos"
    _write(path, [{"id": "q1", "question": "¿A?", "context": context, "answer": ""}])

    assert load_answers(path).to_dict("records") == [{"id": "q1", "question": "¿A?", "context": context, "answer": ""}]
