import json
import re
from collections.abc import Sequence

from shared.prompts import REWRITE_PROMPTS

MAX_TURNS = 2
MAX_ANSWER_CHARS = 400
MAX_WORDS_ACCEPTED = 60

Turns = Sequence[tuple[str, str]]


def format_history(history: Turns) -> str:

    lines = []

    for question, answer in list(history)[-MAX_TURNS:]:

        if len(answer) > MAX_ANSWER_CHARS:
            answer = answer[:MAX_ANSWER_CHARS] + "…"

        lines.append(f"Usuario: {question}\nAsistente: {answer}")

    return "\n".join(lines)


def user_message(history: Turns, question: str) -> str:

    return f"Historial:\n{format_history(history)}\n\nÚltima pregunta: {question}"


def build_messages(history: Turns, question: str, prompt_version: str = "v2") -> list[dict]:

    system, few_shot = REWRITE_PROMPTS[prompt_version]
    messages = [{"role": "system", "content": system}]

    for example in few_shot:
        messages.append({"role": "user", "content": user_message(example["historial"], example["pregunta"])})
        messages.append({"role": "assistant", "content": json.dumps(example["salida"], ensure_ascii=False)})

    messages.append({"role": "user", "content": user_message(history, question)})

    return messages


def norm(text: str) -> str:

    return re.sub(r"\s+", " ", text.lower()).strip()


def parse_rewrite(raw: str, history: Turns, question: str) -> dict:

    base = {
        "necesita_historial": False,
        "pregunta": question,
        "terminos_clave": [],
        "terminos_descartados": [],
        "fallback": None,
    }

    try:
        data = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
    except ValueError:
        return {**base, "fallback": "json_invalido"}

    if not isinstance(data, dict):
        return {**base, "fallback": "json_invalido"}

    rewritten = str(data.get("pregunta", "")).strip()

    if not rewritten:
        return {**base, "fallback": "vacia"}

    if len(rewritten.split()) > MAX_WORDS_ACCEPTED:
        return {**base, "fallback": "demasiado_larga"}

    source = norm(question + " " + " ".join(f"{q} {a}" for q, a in history))
    anchored, discarded = [], []

    for term in data.get("terminos_clave") or []:

        term = str(term).strip()

        if not term:
            continue

        if norm(term) in source:
            anchored.append(term)
        else:
            discarded.append(term)

    return {
        **base,
        "necesita_historial": bool(data.get("necesita_historial", True)),
        "pregunta": rewritten,
        "terminos_clave": anchored,
        "terminos_descartados": discarded,
    }
