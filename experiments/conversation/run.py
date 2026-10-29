import json
from pathlib import Path

import pandas as pd

from experiments.conversation.common import (
    ANSWER_MODEL,
    COLLECTION,
    RESULTS_BASE,
    STAGE,
    TOKEN_LIMIT,
    TOP_K,
    ask_json,
    base_config,
    base_queries,
    cached,
    count_tokens,
    group_ids,
    history_text,
    hit_table,
    hits_by_type,
    is_hit,
    last_question,
    load_conversations,
    paired,
)
from experiments.conversation.prompts import (
    FIDELITY_PROMPT_V1,
    FIDELITY_SCHEMA,
    REWRITE_PROMPT_V1,
    REWRITE_SCHEMA,
)
from shared.eval.data import load_questions
from shared.eval.grounding import abstained
from shared.eval.metrics import percentile, wilson_ci
from shared.ingest import build_qdrant_client, get_embedder
from shared.llm import LLMClient, build_llm
from shared.pipeline import Pipeline
from shared.settings import settings
from shared.tracking import log_metrics, tracked_run

EXP = "multiturno_v1"
DATASET = Path("shared/eval/multiturn/multiturno_v1.jsonl")
RESULTS_DIR = RESULTS_BASE / EXP
METHODS = ("hybrid_rrf", "dense")
REWRITERS = {
    "reescritura_3b": "llama3.2:3b",
    "reescritura_8b": "llama3.1:8b-instruct-q4_K_M",
}
CONDITIONS = ("literal", "concat", *REWRITERS, "oraculo")
REWRITE_V1_NUM_PREDICT = 150
JUDGE_NUM_PREDICT = 300
ORACLE_SHARE_MIN = 0.80
ALPHA = 0.05
REUSE = True


def build_rewrites(conversations: list[dict], rewriters: dict[str, LLMClient]) -> pd.DataFrame:

    rows = []

    for condition, client in rewriters.items():

        ask_json(client, [{"role": "user", "content": "Responde {\"pregunta\": \"ok\"}"}], REWRITE_SCHEMA) # calentamiento

        for conv in conversations:

            prompt = REWRITE_PROMPT_V1.format(history=history_text(conv), question=last_question(conv))

            out = ask_json(client, [{"role": "user", "content": prompt}], REWRITE_SCHEMA) # consulta

            text = (out["data"] or {}).get("pregunta", "").strip()
            valid = bool(text)

            rows.append({
                "conv_id": conv["conv_id"],
                "condicion": condition,
                "modelo": client.model,
                "valido": valid,
                "pregunta": text if valid else last_question(conv),
                "tokens_in": out["tokens_in"],
                "tokens_out": out["tokens_out"],
                "latencia_ms": out["latencia_ms"],
                "tokens_embedder": count_tokens(text if valid else last_question(conv)),
            })

    return pd.DataFrame(rows)


def queries_for(conversations: list[dict], rewrites: pd.DataFrame) -> dict[str, dict[str, str]]:

    rewritten = {
        (r.conv_id, r.condicion): r.pregunta
            for r in rewrites.itertuples()
    }
    
    queries = {}

    for conv in conversations:

        cid = conv["conv_id"]
        queries[cid] = base_queries(conv)

        for condition in REWRITERS:
            queries[cid][condition] = rewritten[(cid, condition)]

    return queries


def build_retrieval(conversations: list[dict], queries: dict) -> pd.DataFrame:

    base = base_config()
    client, embedder = build_qdrant_client(), get_embedder()
    rows = []

    for method in METHODS:

        pipeline = Pipeline(base.model_copy(update={"method": method}), client, embedder, None, None)

        for conv in conversations:
            for condition in CONDITIONS:

                text = queries[conv["conv_id"]][condition]
                chunks = pipeline.retrieve(text, top_k=TOP_K).chunks

                rows.append({
                    "conv_id": conv["conv_id"],
                    "metodo": method,
                    "condicion": condition,
                    "acierto": is_hit(chunks, conv),
                    "tokens_embedder": count_tokens(text),
                    "consulta": text,
                })

    return pd.DataFrame(rows)


def build_fidelity(conversations: list[dict], rewrites: pd.DataFrame, judge: LLMClient) -> pd.DataFrame:

    reference = {c["conv_id"]: c["objetivo"]["pregunta_autonoma"] for c in conversations}
    rows = []

    for r in rewrites.itertuples():

        prompt = FIDELITY_PROMPT_V1.format(reference=reference[r.conv_id], rewritten=r.pregunta)
        out = ask_json(judge, [{"role": "user", "content": prompt}], FIDELITY_SCHEMA)
        data = out["data"] or {}

        rows.append({
            "conv_id": r.conv_id,
            "condicion": r.condicion,
            "pregunta_autonoma": reference[r.conv_id],
            "pregunta_reescrita": r.pregunta,
            "veredicto_juez": data.get("veredicto", ""),
            "motivo_juez": data.get("motivo", ""),
            "etiqueta_humana": "",
        })

    return pd.DataFrame(rows)


def build_abstention(conversations: list[dict], queries: dict, llm: LLMClient) -> pd.DataFrame:

    pipeline = Pipeline(base_config(), build_qdrant_client(), get_embedder(), llm, None)
    rows = []

    unanswerable_turns = [c for c in conversations if c["objetivo"]["debe_abstenerse"]]

    for conv in unanswerable_turns:
        for condition in CONDITIONS:

            answer = pipeline.run(queries[conv["conv_id"]][condition], top_k=TOP_K).answer

            rows.append({
                "origen": "multiturno",
                "id": conv["conv_id"],
                "condicion": condition,
                "respuesta": answer,
                "abstencion": abstained(answer),
            })

    for q in load_questions():

        if q.get("gold_spans"):
            continue

        answer = pipeline.run(q["question"], top_k=TOP_K).answer

        rows.append({
            "origen": "un_turno",
            "id": q["id"],
            "condicion": "literal",
            "respuesta": answer,
            "abstencion": abstained(answer),
        })

    return pd.DataFrame(rows)


def summarize_method(conversations: list[dict], retrieval: pd.DataFrame, method: str) -> dict:

    hits = hit_table(retrieval, method)
    tipos = sorted({c["tipo"] for c in conversations if c["objetivo"]["gold_spans"]})
    out = {}

    for split in (None, "test"):

        groups = group_ids(conversations, split)
        key = "todas" if split is None else split

        with_history = groups["usa_historial"]
        without_history = groups["no_usa_historial"]

        out[key] = {
            "n_usa_historial": len(with_history),
            "n_no_usa_historial": len(without_history),
            "aciertos_usa_historial": {
                c: sum(hits[(i, c)] for i in with_history)
                    for c in CONDITIONS
            },
            "aciertos_no_usa_historial": {
                c: sum(hits[(i, c)] for i in without_history)
                    for c in CONDITIONS
            },
            "pares_usa_historial": [
                paired(hits, with_history, "literal", r) for r in REWRITERS
            ] + [
                paired(hits, with_history, "concat", r) for r in REWRITERS
            ] + [
                paired(hits, with_history, "literal", "concat"),
            ],
            "pares_no_usa_historial": [
                paired(hits, without_history, "literal", r) for r in REWRITERS
            ],
        }

    out["por_tipo"] = hits_by_type(conversations, hits, tipos, CONDITIONS)

    return out


def acceptance(summary: dict) -> dict:

    verdicts = {}

    for rewriter in REWRITERS:

        result = {}

        for split in ("todas", "test"):

            block = summary[split]
            hist = block["aciertos_usa_historial"]
            pairs = {(p["a"], p["b"]): p for p in block["pares_usa_historial"]}
            no_hist = {(p["a"], p["b"]): p for p in block["pares_no_usa_historial"]}
            vs_literal = pairs[("literal", rewriter)]
            vs_concat = pairs[("concat", rewriter)]
            contamination = no_hist[("literal", rewriter)]

            result[split] = {
                "fraccion_del_oraculo": round(hist[rewriter] / hist["oraculo"], 3) if hist["oraculo"] else None,
                "c1_alcanza_80_del_oraculo": hist["oraculo"] > 0 and hist[rewriter] >= ORACLE_SHARE_MIN * hist["oraculo"],
                "c2_supera_literal_significativo": vs_literal["solo_b"] > vs_literal["solo_a"] and vs_literal["p_mcnemar"] < ALPHA,
                "c3_supera_concat": vs_concat["solo_b"] > vs_concat["solo_a"],
                "c4_sin_perdida_significativa_sin_historial": not (
                    contamination["solo_a"] > contamination["solo_b"] and contamination["p_mcnemar"] < ALPHA
                ),
            }

        verdicts[rewriter] = result

    return verdicts


def summarize_costs(rewrites: pd.DataFrame) -> dict:

    out = {}

    for condition, group in rewrites.groupby("condicion"):

        latencies = group["latencia_ms"].astype(float).tolist()

        out[condition] = {
            "modelo": group["modelo"].iloc[0],
            "validas": int(group["valido"].astype(str).eq("True").sum()),
            "n": len(group),
            "latencia_p50_ms": round(percentile(latencies, 50), 1),
            "latencia_p95_ms": round(percentile(latencies, 95), 1),
            "tokens_in_p50": float(group["tokens_in"].astype(float).median()),
            "tokens_out_p50": float(group["tokens_out"].astype(float).median()),
            "tokens_embedder_p50": float(group["tokens_embedder"].astype(float).median()),
            "tokens_embedder_max": int(group["tokens_embedder"].astype(int).max()),
            "superan_limite_embedder": int((group["tokens_embedder"].astype(int) > TOKEN_LIMIT).sum()),
        }

    return out


def summarize_fidelity(conversations: list[dict], fidelity: pd.DataFrame) -> dict:

    needs_history = {c["conv_id"]: c["objetivo"]["usa_historial"] for c in conversations}
    out = {}

    for condition, group in fidelity.groupby("condicion"):

        group = group.assign(usa_historial=group["conv_id"].map(needs_history))

        out[condition] = {
            "todas": group["veredicto_juez"].value_counts().to_dict(),
            "usa_historial": group[group["usa_historial"]]["veredicto_juez"].value_counts().to_dict(),
            "no_usa_historial": group[~group["usa_historial"]]["veredicto_juez"].value_counts().to_dict(),
        }

    return out


def summarize_abstention(abstention: pd.DataFrame) -> dict:

    flags = abstention["abstencion"].astype(str).eq("True")
    single = abstention[abstention["origen"] == "un_turno"]
    single_hits = int(flags[single.index].sum())
    out = {
        "un_turno": {
            "abstenciones": single_hits,
            "n": len(single),
            "ic95_wilson": wilson_ci(single_hits, len(single)),
        },
    }

    multi = abstention[abstention["origen"] == "multiturno"]

    for condition, group in multi.groupby("condicion"):

        hits = int(flags[group.index].sum())

        out[condition] = {
            "abstenciones": hits,
            "n": len(group),
            "ic95_wilson": wilson_ci(hits, len(group)),
        }

    return out


def log_to_mlflow(summary: dict) -> None:

    hybrid = summary["retrieval"]["hybrid_rrf"]["todas"]

    for condition in CONDITIONS:

        params = {
            "dataset": DATASET.name,
            "collection": COLLECTION,
            "method": "hybrid_rrf",
            "top_k": TOP_K,
            "condition": condition,
            "rewrite_model": REWRITERS.get(condition, ""),
            "rewrite_prompt": "v1" if condition in REWRITERS else "",
        }

        with tracked_run(STAGE, condition, params):
            log_metrics({
                "hits_needs_history": hybrid["aciertos_usa_historial"][condition],
                "n_needs_history": hybrid["n_usa_historial"],
                "hits_no_history": hybrid["aciertos_no_usa_historial"][condition],
                "n_no_history": hybrid["n_no_usa_historial"],
            })


def main() -> dict:

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "summary.json"

    if summary_path.exists() and not REUSE:
        raise SystemExit(f"{summary_path} ya existe; cambiar EXP para no sobrescribir resultados.")

    conversations = load_conversations(DATASET)
    rewriters = {
        condition: build_llm(
            provider="ollama", 
            model=model, 
            max_tokens=REWRITE_V1_NUM_PREDICT,
            repeat_penalty=None
        ) for condition, model in REWRITERS.items()
    }

    judge = build_llm(
        provider="ollama", 
        model=settings.llm_model_judge, 
        max_tokens=JUDGE_NUM_PREDICT,
        repeat_penalty=None
    )

    answer = build_llm(
        provider="ollama",
        model=ANSWER_MODEL, 
        max_tokens=base_config().llm.max_tokens
    )

    rewrites = cached(RESULTS_DIR / "rewrites.csv", lambda: build_rewrites(conversations, rewriters), REUSE)
    queries = queries_for(conversations, rewrites)
    retrieval = cached(RESULTS_DIR / "retrieval.csv", lambda: build_retrieval(conversations, queries), REUSE)
    fidelity = cached(RESULTS_DIR / "fidelity.csv", lambda: build_fidelity(conversations, rewrites, judge), REUSE)
    abstention = cached(RESULTS_DIR / "abstention.csv", lambda: build_abstention(conversations, queries, answer), REUSE)

    retrieval_summary = {
        m: summarize_method(conversations, retrieval, m)
            for m in METHODS
    }

    summary = {
        "exp": EXP,
        "n_conversaciones": len(conversations),
        "retrieval": retrieval_summary,
        "criterios": {m: acceptance(retrieval_summary[m]) for m in METHODS},
        "costo_reescritura": summarize_costs(rewrites),
        "fidelidad_juez": summarize_fidelity(conversations, fidelity),
        "abstencion": summarize_abstention(abstention),
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    log_to_mlflow(summary)

    return summary

if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=1))
