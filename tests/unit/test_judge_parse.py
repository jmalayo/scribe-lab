import pytest

from shared.eval.judge import agreement_rate, parse_verdict


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("TRUE", (True, True)),
        ("FALSE", (False, True)),
        ("true", (True, True)),
        ("La respuesta cita cifras del contexto. Por lo tanto: TRUE", (True, True)),
        ("SÍ", (False, False)),
        ("", (False, False)),
        (None, (False, False)),
    ],
    ids=["true", "false", "minusculas", "razonamiento_previo", "formato_si_no", "vacia", "none"],
)
def test_parse_verdict(raw, expected):
    assert parse_verdict(raw) == expected

# fija el comportamiento actual: con dos veredictos en el texto gana el primero que aparece
def test_parse_verdict_takes_the_first_verdict():
    assert parse_verdict("FALSE. Aunque parte de la respuesta podría ser TRUE") == (False, True)

# falla conocida: el regex no exige palabra completa, así que un razonamiento en español con
@pytest.mark.xfail(strict=True, reason="busca TRUE/FALSE como substring: 'falsedad' contiene 'false'")
def test_parse_verdict_ignores_words_that_contain_false():
    assert parse_verdict("No hay falsedad en la respuesta. TRUE") == (True, True)

# falla conocida: si el juez repite la consigna ("TRUE o FALSE") antes de decidir, gana el TRUE de la consigna y no el veredicto final;
@pytest.mark.xfail(strict=True, reason="toma la primera coincidencia: falla si el juez repite 'TRUE o FALSE' antes de decidir")
def test_parse_verdict_uses_the_final_verdict():
    assert parse_verdict("Debo responder TRUE o FALSE. La respuesta inventa una cifra: FALSE") == (False, True)

# arma veredictos con el mismo formato que judge_answers(): un dict por pregunta
def verdicts(*grounded: bool) -> list[dict]:
    return [{"id": f"q{i}", "grounded": g, "relevant": True} for i, g in enumerate(grounded)]

# acuerdo = fracción de preguntas con el mismo veredicto en la clave pedida, redondeado a 4 decimales
# como en results_summary.csv; sin veredictos devuelve 0.0 en vez de dividir por cero
def test_agreement_rate():

    assert agreement_rate(verdicts(True, False, True), verdicts(True, False, True), "grounded") == 1.0
    assert agreement_rate(verdicts(True, False, True), verdicts(True, True, True), "grounded") == 0.6667
    assert agreement_rate(verdicts(True, False), verdicts(False, True), "relevant") == 1.0
    assert agreement_rate([], [], "grounded") == 0.0

# strict=True: si a un juez le falta una respuesta, el acuerdo se calcularía sobre pares desalineados
def test_agreement_rate_rejects_unaligned_verdicts():

    with pytest.raises(ValueError):
        agreement_rate(
            verdicts(True, False, True), 
            verdicts(True, False), 
            "grounded"
        )
