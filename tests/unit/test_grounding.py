import pytest

from shared.eval.grounding import split_sentences


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
