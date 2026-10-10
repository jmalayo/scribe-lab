from textwrap import dedent


def prompt_v1():

    ANSWER_PROMPT = """
    Responde la pregunta usando ÚNICAMENTE el contexto de abajo. Si el contexto no contiene la respuesta, di "No lo sé según el contexto proporcionado."

    Contexto: {context}
    Pregunta: {question}

    Responde en español, de forma concisa (2-3 oraciones):
    """

    GROUNDEDNESS_PROMPT = """
    ¿Cada afirmación de la "Respuesta a evaluar" de abajo está directamente respaldada por el Contexto de abajo? Responde TRUE o FALSE.

    Contexto: {context}

    Respuesta a evaluar: {answer}
    """

    RELEVANCE_PROMPT = """
    ¿La "Respuesta a evaluar" de abajo realmente aborda la Pregunta de abajo, sin importar si el contenido es correcto? Responde TRUE o FALSE.

    Pregunta: {question}

    Respuesta a evaluar: {answer}
    """

    return dedent(ANSWER_PROMPT), dedent(GROUNDEDNESS_PROMPT), dedent(RELEVANCE_PROMPT)


def prompt_v2():

    RELEVANCE_PROMPT = """\
    Pregunta: {question}

    Respuesta: {answer}

    ¿La respuesta aborda la pregunta, sin importar si es correcta? Devuelve JSON con las claves "relevante" (bool) y "motivo" (str)."""

    RELEVANCE_SCHEMA = {
        "type": "object",
        "required": ["relevante", "motivo"],
        "properties": {
            "relevante": {"type": "boolean"},
            "motivo": {"type": "string"}
        }
    }

    return dedent(RELEVANCE_PROMPT), RELEVANCE_SCHEMA
