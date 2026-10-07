import re

# corte después de . ! ? seguido de espacio y de algo que abre oración; los decimales ("48.9")
# y las abreviaturas sin mayúscula siguiente no cortan
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡`\"'(*\-•\d])")
_BLOCK_RE = re.compile(r"\n\s*\n|\n(?=\s*(?:[-*•]|\d+[.)])\s)")


def split_sentences(answer: str) -> list[str]:
    """Unidad de etiquetado y de evaluación: oraciones de la respuesta, en orden.

    Primero separa bloques (párrafos e ítems de lista) y después oraciones dentro de cada
    bloque. Las etiquetas humanas y los detectores usan esta misma función.
    """

    sentences = []

    for block in _BLOCK_RE.split(answer or ""):
        for sentence in _SENTENCE_END_RE.split(block.strip()):
            sentence = " ".join(sentence.split())
            if sentence:
                sentences.append(sentence)

    return sentences
