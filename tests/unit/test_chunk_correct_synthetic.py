import pytest

from shared.eval.metrics import is_chunk_correct

DOC = "doc/benchmark.md"

SPAN = "Random Forest obtuvo 48.9 % ± 13.3 de balanced accuracy con 5 folds"
ROW = "| Random Forest | 48.9 % | 13.3 |"

PRE = "Resultados del modelo de producción sobre el dataset completo. "
POST = " Esto se usa como referencia para el resto del reporte."

SWAPPED = SPAN.replace("48.9", "@").replace("13.3", "48.9").replace("@", "13.3")

def question(
    spans: list[str],
    source_doc: str = DOC
) -> dict:
    return {
        "source_doc": source_doc,
        "gold_spans": spans
    }

def chunk(
    text: str,
    doc_id: str = DOC,
    tables: list[dict] | None = None
) -> dict:
    return {
        "doc_id": doc_id,
        "text": text,
        "tables": tables or []
    }

# formato distinto, mismo dato: debe contar como acierto
@pytest.mark.parametrize(
    "span, c",
    [
        (SPAN, chunk(PRE + SPAN + POST)),
        (SPAN, chunk(PRE + SPAN.replace("obtuvo ", "obtuvo  \n") + POST)),
        (SPAN, chunk(PRE + SPAN.replace("48.9 %", "**48.9 %**") + POST)),
        (SPAN, chunk(PRE + SPAN.replace("Random Forest", "`Random Forest`") + POST)),
        (SPAN, chunk(PRE + SPAN.replace("obtuvo", "obtuov") + POST)),
        (ROW, chunk(PRE, tables=[{"text_content": "| Random Forest   | 48.9 %   | 13.3   |"}])),
    ],
    ids=[
        "identico",
        "espacios_y_salto_de_linea",
        "negrita_en_la_fuente",
        "backticks_en_la_fuente",
        "typo_de_una_letra",
        "fila_de_tabla_con_otro_padding",
    ],
)
def test_same_data_different_format_is_hit(span, c):
    assert is_chunk_correct(
        c,
        question([span])
    ) is True

# dato alterado, incompleto o ajeno: no debe contar como acierto
@pytest.mark.parametrize(
    "c",
    [
        chunk(PRE + SPAN.replace("48.9", "49.8") + POST),
        chunk(PRE + SWAPPED + POST),
        chunk(PRE + SWAPPED + " Antes se citaba 48.9 % y 13.3 de desvío."),
        chunk(PRE + SPAN.replace(" ± 13.3", "") + POST),
        chunk(PRE + SPAN[:SPAN.index("balanced") + len("balanced")]),
        chunk(SPAN[SPAN.index("48.9"):] + POST),
        chunk(PRE + "Árbol de decisión obtuvo 37.2 % ± 13.1 de balanced accuracy con 5 folds" + POST),
        chunk(PRE + SPAN + POST, doc_id="doc/otro.md"),
    ],
    ids=[
        "numero_cambiado",
        "numeros_invertidos",
        "numeros_invertidos_con_correctos_en_otra_oracion",
        "falta_un_numero",
        "truncado_al_final",
        "truncado_al_inicio",
        "misma_estructura_otro_modelo",
        "documento_equivocado",
    ],
)
def test_altered_or_incomplete_data_is_miss(c):
    assert is_chunk_correct(
        c,
        question([SPAN])
    ) is False

def test_any_of_multiple_source_docs():
    q = question([SPAN], source_doc=f"doc/otro.md; {DOC}")

    assert is_chunk_correct(
        chunk(PRE + SPAN + POST),
        q
    )
