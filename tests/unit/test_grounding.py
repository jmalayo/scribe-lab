import pytest

from shared.eval.grounding import (
    GroundingResult,
    ground_answer,
    nli_scores,
    sentence_ranges,
    sentence_scores,
    should_abstain,
    split_sentences,
    unsupported_numbers,
)


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("", []),
        ("No lo sé según el contexto proporcionado.", ["No lo sé según el contexto proporcionado."]),
        (
            "Random Forest obtuvo 48.9 % ± 13.3. El árbol quedó en 46.1 %.",
            ["Random Forest obtuvo 48.9 % ± 13.3.", "El árbol quedó en 46.1 %."],
        ),
        (
            "La `baja-contemplativa` tiene 100 %. `baja-ritmica` es el patrón inverso.",
            ["La `baja-contemplativa` tiene 100 %.", "`baja-ritmica` es el patrón inverso."],
        ),
        (
            "Primer párrafo.\n\nResumen: segundo párrafo.",
            ["Primer párrafo.", "Resumen: segundo párrafo."],
        ),
        (
            "Pasos:\n- bloques de 3 s\n- gate absoluto",
            ["Pasos:", "- bloques de 3 s", "- gate absoluto"],
        ),
        ("kernel_size=9 se rechazó (Spearman=0.86).", ["kernel_size=9 se rechazó (Spearman=0.86)."]),
    ],
    ids=["vacia", "abstencion", "decimales", "backticks", "parrafos", "lista", "una_oracion"],
)
def test_split_sentences(answer, expected):
    assert split_sentences(answer) == expected


def test_sentence_ranges_tolera_espacios_colapsados():
    answer = "Primera   oración.\n\nSegunda\noración."

    ranges = sentence_ranges(answer, split_sentences(answer))

    assert [answer[s:e] for s, e in ranges] == ["Primera   oración.", "Segunda\noración."]


def test_sentence_ranges_oracion_repetida_avanza_en_orden():
    answer = "Sí. Sí."

    assert sentence_ranges(answer, ["Sí.", "Sí."]) == [(0, 3), (4, 7)]


def test_sentence_scores_toma_el_maximo_de_sus_tokens():
    answer = "Tarda 5 minutos. Es correcto."
    tokens = [
        {"start": 0, "end": 5, "prob": 0.1},
        {"start": 6, "end": 7, "prob": 0.8},
        {"start": 8, "end": 16, "prob": 0.3},
        {"start": 17, "end": 19, "prob": 0.2},
        {"start": 29, "end": 29, "prob": 0.99},   # token de largo cero: no cuenta
    ]

    assert [round(s["score"], 2) for s in sentence_scores(answer, tokens)] == [0.8, 0.2]


@pytest.mark.parametrize(
    "sentence, context, question, expected",
    [
        ("Obtuvo 48.9 %.", "Random Forest 48.9 % ± 13.3", "", []),
        ("Obtuvo 48,9 %.", "Random Forest 48.9 %", "", []),
        ("Tarda ~5 minutos.", "tarda ~1.5 minutos", "", ["5"]),
        ("1. El SR es 22050.", "SR=22050", "", []),
        ("Pasó de 22050 a 44100.", "SR=22050", "¿44100?", []),
        ("Spearman=0.91.", "Spearman=0.86", "", ["0.91"]),
    ],
    ids=["en_contexto", "coma_decimal", "fabricada", "marcador_lista", "de_la_pregunta", "decimal_cambiado"],
)
def test_unsupported_numbers(sentence, context, question, expected):
    assert unsupported_numbers(sentence, context, question) == expected


@pytest.mark.parametrize(
    "scores, min_score, expected",
    [([0.9, 0.1], 0.5, False), ([0.4, 0.1], 0.5, True), ([], 0.5, True)],
    ids=["supera", "no_supera", "sin_chunks"],
)
def test_should_abstain(scores, min_score, expected):
    assert should_abstain([{"rerank_score": s} for s in scores], min_score) is expected


def test_ground_answer_combina_lettucedetect_y_cifras():
    answer = "Tarda 5 minutos. Usa RRF. Es rápido."
    tokens = [
        {"start": 0, "end": 16, "prob": 0.2},
        {"start": 17, "end": 25, "prob": 0.7},
        {"start": 26, "end": 36, "prob": 0.1},
    ]

    result = ground_answer(answer, ["usa RRF y tarda 2 minutos"], "¿cuánto tarda?", tokens, threshold=0.5)

    assert [(u["sentence"], u["reasons"]) for u in result["unsupported_sentences"]] == [
        ("Tarda 5 minutos.", ["numeric"]),
        ("Usa RRF.", ["lettucedetect"]),
    ]
    assert result["score"] == pytest.approx(1 / 3)


def _fake_predict(table):
    def predict(pairs):
        return [table[(c, h)] for c, h in pairs]
    return predict


def test_nli_scores_toma_el_maximo_sobre_los_chunks():
    table = {
        ("c1", "s1"): {"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05},
        ("c2", "s1"): {"entailment": 0.1, "neutral": 0.2, "contradiction": 0.7},
        ("c1", "s2"): {"entailment": 0.2, "neutral": 0.7, "contradiction": 0.1},
        ("c2", "s2"): {"entailment": 0.3, "neutral": 0.6, "contradiction": 0.1},
    }

    assert nli_scores(["s1", "s2"], ["c1", "c2"], _fake_predict(table)) == [
        {"entailment": 0.9, "contradiction": 0.7},
        {"entailment": 0.3, "contradiction": 0.1},
    ]


def test_nli_scores_sin_chunks_no_llama_al_modelo():
    def predict(pairs):
        raise AssertionError("no debería llamarse")

    assert nli_scores(["s1"], [], predict) == [{"entailment": 0.0, "contradiction": 0.0}]


def test_ground_answer_suma_las_razones_del_nli():
    answer = "Usa RRF. Es rápido."
    tokens = [{"start": 0, "end": 8, "prob": 0.1}, {"start": 9, "end": 19, "prob": 0.1}]
    nli = [{"entailment": 0.9, "contradiction": 0.6}, {"entailment": 0.4, "contradiction": 0.1}]

    result = ground_answer(answer, ["usa RRF"], "", tokens, threshold=0.5, nli=nli)

    assert [u["reasons"] for u in result["unsupported_sentences"]] == [
        ["nli_contradiction"],
        ["nli_not_entailed"],
    ]
    assert result["score"] == 0.0


def test_ground_answer_output_matches_the_service_schema():
    answer = "Tarda 5 minutos. Usa RRF."
    tokens = [{"start": 0, "end": 16, "prob": 0.2}, {"start": 17, "end": 25, "prob": 0.7}]
    nli = [{"entailment": 0.9, "contradiction": 0.1}, {"entailment": 0.4, "contradiction": 0.1}]

    raw = ground_answer(answer, ["usa RRF y tarda 2 minutos"], "", tokens, threshold=0.5, nli=nli)
    result = GroundingResult.model_validate(raw)

    assert result.model_dump() == raw
    assert result.unsupported_sentences[1].nli.entailment == 0.4
