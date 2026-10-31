from textwrap import dedent

REWRITE_PROMPT_V1 = dedent("""
    Tu tarea es reescribir la última pregunta del usuario para que se entienda sin leer la conversación.

    Reglas:
    - Si la última pregunta ya se entiende sola, devuélvela sin cambios.
    - Si depende de la conversación (pronombres como "eso" o "ese modelo", un documento omitido como "¿y en genesis?", un valor nuevo como "¿y con 9?", o una corrección como "no, me refería a..."), reemplaza esas referencias por lo que significan según la conversación.
    - Si la última pregunta cambia de tema, no le agregues nada de los turnos anteriores.
    - Conserva nombres de documentos, modelos, cifras y términos técnicos tal como aparecen.
    - No respondas la pregunta y no agregues información que no esté en la conversación.
    - Usa como máximo 60 palabras.

    Conversación:
    {history}

    Última pregunta del usuario: {question}

    Responde solo con un JSON de la forma {{"pregunta": "<pregunta reescrita>"}}.
    """)

REWRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "pregunta": {"type": "string"},
    },
    "required": ["pregunta"],
}

FIDELITY_PROMPT_V1 = dedent("""
    Compara dos preguntas sobre la misma documentación técnica.

    Pregunta de referencia: {reference}
    Pregunta reescrita: {rewritten}

    Decide si la pregunta reescrita pide lo mismo que la de referencia:
    - "equivalente": pide la misma información, sobre el mismo documento, objeto y valores, aunque use otras palabras.
    - "parcial": apunta al mismo tema pero le falta o le sobra algo que cambia qué se busca (otro documento, otro valor, una parte omitida).
    - "distinta": pide otra cosa, o sigue dependiendo de una conversación que no se ve.

    Responde solo con un JSON de la forma {{"veredicto": "equivalente|parcial|distinta", "motivo": "<una oración>"}}.
    """)

FIDELITY_SCHEMA = {
    "type": "object",
    "properties": {
        "veredicto": {"type": "string", "enum": ["equivalente", "parcial", "distinta"]},
        "motivo": {"type": "string"},
    },
    "required": ["veredicto", "motivo"],
}
