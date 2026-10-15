from textwrap import dedent

PROMPT_VERSION = "v1"

ANSWER_PROMPT_V1 = dedent("""
    Responde la pregunta usando ÚNICAMENTE el contexto de abajo. Si el contexto no contiene la respuesta, di "No lo sé según el contexto proporcionado."

    Contexto: {context}
    Pregunta: {question}

    Responde en español, de forma concisa (2-3 oraciones):
    """)

ANSWER_PROMPTS = {
    "v1": ANSWER_PROMPT_V1
}

def chunk_block(chunk: dict) -> str:

    tables_str = "\n\n".join(
        t["text_content"] 
            for t in chunk.get("tables") or []
    )

    block = f"[{chunk['doc_id']}] {chunk['text']}"

    if tables_str:
        block += f"\n\n{tables_str}"

    return block

def build_context_text(context_chunks: list[dict]) -> str:

    return "\n\n".join(
        chunk_block(chunk) 
            for chunk in context_chunks
    )
