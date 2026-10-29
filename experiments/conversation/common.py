import json
import re
import time
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from shared.eval.data import load_questions
from shared.eval.metrics import bootstrap_ci, is_chunk_correct, mcnemar_exact
from shared.llm import LLMClient
from shared.memory import build_messages, norm, parse_rewrite
from shared.pipeline import MemoryConfig, Pipeline, PipelineConfig
from shared.settings import settings

STAGE = "conversation-multiturn"
RESULTS_BASE = Path("experiments/conversation/results")
COLLECTION = "exp_paired_293_73"
TOP_K = 10
TOKEN_LIMIT = 126
EMBEDDING_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
ANSWER_MODEL = "llama3.2:3b"
REWRITE_MODEL = "llama3.2:3b"
REWRITE_PROMPT_VERSION = "rewrite_v2"
REWRITE_NUM_PREDICT = 120
TERMS_TOKEN_LIMIT = 110
BOOTSTRAP_RESAMPLES = 10_000

STOPWORDS = set(
    "el la los las un una unos unas de del al a en y o u que qué cuál cuáles cómo cuánto cuántos cuántas "
    "cuándo dónde por para con sin se su sus es son fue era lo le les este esta estos estas ese esa eso ya "
    "no sí si según entre sobre más menos muy como también pero cada otro otra mismo misma hay ha han tiene "
    "tienen dice da dan".split()
)


@cache
def tokenizer():

    return AutoTokenizer.from_pretrained(EMBEDDING_MODEL_ID, revision=EMBEDDING_REVISION)


def count_tokens(text: str) -> int:

    return len(tokenizer().encode(text, add_special_tokens=False))


def load_conversations(path: Path) -> list[dict]:

    questions = {
        q["id"]: q
            for q in load_questions()
    }

    conversations = [
        json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
    ]

    for conv in conversations:

        target = conv["objetivo"]
        source = questions[target["ref_id"]]

        for key in ("gold_spans", "source_doc", "expected_answer"):
            if (source.get(key) or None) != (target.get(key) or None):
                raise SystemExit(f"{conv['conv_id']}: {key} no coincide con {target['ref_id']}")

    return conversations


def base_config(**update) -> PipelineConfig:

    config = PipelineConfig.from_yaml(settings.pipeline_config_path)

    return config.model_copy(
        update={
            "collection": COLLECTION, 
            "memory": MemoryConfig(), 
            **update
        }
    )


def cached(resutls_dir: Path, build: Callable[[], pd.DataFrame], reuse: bool) -> pd.DataFrame:

    if reuse and resutls_dir.exists():
        return pd.read_csv(resutls_dir, sep=";", keep_default_na=False)

    df = build()
    df.to_csv(resutls_dir, sep=";", index=False)

    return df


def is_hit(chunks: list[dict], conv: dict) -> bool:

    target = {
        "gold_spans": conv["objetivo"]["gold_spans"],
        "source_doc": conv["objetivo"]["source_doc"],
    }

    if not target["gold_spans"]:
        return False

    return any(is_chunk_correct(c, target) for c in chunks)


def hit_table(retrieval: pd.DataFrame, method: str | None = None) -> dict[tuple[str, str], bool]:

    if method is not None:
        retrieval = retrieval[retrieval["metodo"] == method]

    hits = {}

    for r in retrieval.itertuples():
        hits[(r.conv_id, r.condicion)] = r.acierto in (True, "True")

    return hits


def hits_by_type(conversations: list[dict], hits: dict, types: list[str], conditions: tuple[str, ...]) -> dict:

    out = {}

    for tipo in types:

        ids = [c["conv_id"] for c in conversations if c["tipo"] == tipo]
        out[tipo] = {"n": len(ids)}

        for condition in conditions:
            out[tipo][condition] = sum(hits[(i, condition)] for i in ids)

    return out


def last_question(conv: dict) -> str:

    return conv["turnos"][-1]["texto"]


def history_text(conv: dict) -> str:

    roles = {
        "user": "Usuario",
        "assistant": "Asistente",
    }

    return "\n".join(
        f"{roles[t['rol']]}: {t['texto']}"
            for t in conv["turnos"][:-1] # Todos los turnos anteriores a la última pregunta del usuario
    )


def history_pairs(conv: dict) -> list[tuple[str, str]]:

    turns = conv["turnos"][:-1]
    pairs = []

    for i in range(0, len(turns), 2):

        question = turns[i]["texto"]
        answer = turns[i + 1]["texto"] if i + 1 < len(turns) else ""

        pairs.append((question, answer))

    return pairs


def strong_points(text: str) -> str:

    words = re.findall(r"[\wÀ-ÿ.%/-]+", text.lower())

    return " ".join(
        w for w in words
            if w not in STOPWORDS and len(w) > 1
    )


def concat_query(conv: dict) -> str:

    text = last_question(conv)
    previous_q = [
        t["texto"] 
            for t in conv["turnos"][:-1] 
                if t["rol"] == "user"
    ]

    for turn in reversed(previous_q):

        candidate = f"{text} {strong_points(turn)}"

        if count_tokens(candidate) > TOKEN_LIMIT:
            break

        text = candidate

    return text


def base_queries(conv: dict) -> dict[str, str]:

    return {
        "literal": last_question(conv),
        "concat": concat_query(conv),
        "oraculo": conv["objetivo"]["pregunta_autonoma"],
    }


def search(pipeline: Pipeline, queries: dict[str, str], condition: str) -> tuple[list[dict], float]:

    t0 = time.perf_counter()

    if condition == "multi_consulta":
        chunks = pipeline.retrieve_fused(queries["literal"], queries["reescritura_v2"], TOP_K).chunks
    else:
        chunks = pipeline.retrieve(queries[condition], top_k=TOP_K).chunks

    return chunks, (time.perf_counter() - t0) * 1000


def query_with_terms(question: str, terms: Sequence[str], max_tokens: int = TERMS_TOKEN_LIMIT) -> str:

    query = question

    for term in terms:

        if norm(term) in norm(query):
            continue

        candidate = f"{query} {term}"

        if count_tokens(candidate) > max_tokens:
            break

        query = candidate

    return query


def ask_json(client: LLMClient, messages: list[dict], schema: dict | None = None) -> dict:

    t0 = time.perf_counter()
    result = client.chat(messages, json_format=schema or True)

    try:
        data = json.loads(result.text)
    except json.JSONDecodeError:
        data = None

    return {
        "raw": result.text,
        "data": data,
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
        "latencia_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


def build_rewrites_v2(conversations: list[dict], client: LLMClient) -> pd.DataFrame:

    ask_json(client, [{"role": "user", "content": "Responde {}"}])
    rows = []

    for conv in conversations:

        history, question = history_pairs(conv), last_question(conv)
        out = ask_json(client, build_messages(history, question))
        parsed = parse_rewrite(out["raw"], history, question)

        rows.append({
            "conv_id": conv["conv_id"],
            "prompt": REWRITE_PROMPT_VERSION,
            "modelo": client.model,
            "fallback": parsed["fallback"] or "",
            "necesita_historial_modelo": parsed["necesita_historial"],
            "necesita_historial_gold": conv["objetivo"]["usa_historial"],
            "pregunta": parsed["pregunta"],
            "consulta_terminos": query_with_terms(parsed["pregunta"], parsed["terminos_clave"]),
            "terminos_clave": json.dumps(parsed["terminos_clave"], ensure_ascii=False),
            "terminos_descartados": json.dumps(parsed["terminos_descartados"], ensure_ascii=False),
            "tokens_in": out["tokens_in"],
            "tokens_out": out["tokens_out"],
            "latencia_ms": out["latencia_ms"],
            "salida_cruda": out["raw"],
        })

    return pd.DataFrame(rows)


def group_ids(conversations: list[dict], split: str | None = None) -> dict[str, list[str]]:

    convs = [c for c in conversations if split is None or c["split"] == split]
    answerable = [c for c in convs if c["objetivo"]["gold_spans"]]

    return {
        "usa_historial": [c["conv_id"] for c in answerable if c["objetivo"]["usa_historial"]],
        "no_usa_historial": [c["conv_id"] for c in answerable if not c["objetivo"]["usa_historial"]],
    }


def paired(hits: dict, ids: list[str], a: str, b: str) -> dict:

    ha = [bool(hits[(i, a)]) for i in ids]
    hb = [bool(hits[(i, b)]) for i in ids]
    only_a = sum(x and not y for x, y in zip(ha, hb, strict=True))
    only_b = sum(y and not x for x, y in zip(ha, hb, strict=True))
    diffs = [float(y) - float(x) for x, y in zip(ha, hb, strict=True)]

    return {
        "a": a,
        "b": b,
        "aciertos_a": sum(ha),
        "aciertos_b": sum(hb),
        "solo_a": only_a,
        "solo_b": only_b,
        "p_mcnemar": mcnemar_exact(only_a, only_b),
        "dif_media_b_menos_a": round(sum(diffs) / len(diffs), 3) if diffs else 0.0,
        "ic95_bootstrap": bootstrap_ci(diffs, n_resamples=BOOTSTRAP_RESAMPLES),
    }
