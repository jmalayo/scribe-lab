import json
from pathlib import Path

import pandas as pd

from experiments.conversation.common import (
    COLLECTION,
    RESULTS_BASE,
    REWRITE_MODEL,
    REWRITE_NUM_PREDICT,
    REWRITE_PROMPT_VERSION,
    STAGE,
    TOP_K,
    base_config,
    base_queries,
    build_rewrites_v2,
    cached,
    count_tokens,
    group_ids,
    hit_table,
    is_hit,
    load_conversations,
    paired,
    search,
)
from shared.eval.metrics import percentile
from shared.ingest import build_qdrant_client, get_embedder
from shared.llm import build_llm
from shared.memory import norm
from shared.pipeline import Pipeline
from shared.tracking import log_metrics, tracked_run

EXP = "multiturno_v1_fase3"
SPLIT = "test"
DATASET = Path("shared/eval/multiturn/multiturno_v1.jsonl")
RESULTS_ROOT = RESULTS_BASE / EXP / REWRITE_PROMPT_VERSION
RESULTS_DIR = RESULTS_ROOT / SPLIT
PHASE2_REWRITES = RESULTS_BASE / "multiturno_v1" / "rewrites.csv"
PHASE2_CONDITION = "reescritura_3b"
METHOD = "hybrid_rrf"
FALLBACK_MAX = 0.10
CANDIDATES = ("reescritura_v2", "reescritura_terminos", "multi_consulta")
BASELINES = ("literal", "concat", "reescritura_v1")
PHASE2_FAILURES = {
    "c04": ["Fase 1"],
    "c09": ["perc_ratio"],
    "c10": ["perc_ratio"],
    "c12": ["balanced accuracy"],
    "c36": ["balanced accuracy"],
    "c39": ["embed_mert.py"],
    "c40": ["22050", "44100"],
}
REUSE = True


def winner_from_dev() -> str:

    dev = json.loads((RESULTS_ROOT / "dev" / "summary.json").read_text(encoding="utf-8"))
    winner = dev["ganadora"]

    if not winner:
        raise SystemExit("Ninguna condición pasó la regla de elección en dev; no se corre test.")

    return winner


def conditions_for_split() -> tuple[str, ...]:

    if SPLIT == "dev":
        return (*BASELINES, *CANDIDATES, "oraculo")

    return (*BASELINES, winner_from_dev(), "oraculo")


def build_retrieval(conversations: list[dict], rewrites: pd.DataFrame, conditions: tuple[str, ...]) -> pd.DataFrame:

    pipeline = Pipeline(base_config(method=METHOD), build_qdrant_client(), get_embedder(), None, None)
    v1 = pd.read_csv(PHASE2_REWRITES, sep=";", keep_default_na=False)
    v1 = {r.conv_id: r.pregunta for r in v1[v1["condicion"] == PHASE2_CONDITION].itertuples()}
    v2 = {r.conv_id: r for r in rewrites.itertuples()}
    pipeline.retrieve("calentamiento", top_k=TOP_K)
    rows = []

    for conv in conversations:

        cid = conv["conv_id"]
        queries = {
            **base_queries(conv),
            "reescritura_v1": v1[cid],
            "reescritura_v2": v2[cid].pregunta,
            "reescritura_terminos": v2[cid].consulta_terminos,
        }

        for condition in conditions:

            chunks, latency = search(pipeline, queries, condition)

            if condition == "multi_consulta":
                searched = [queries["literal"], queries["reescritura_v2"]]
            else:
                searched = [queries[condition]]

            rows.append({
                "conv_id": cid,
                "metodo": METHOD,
                "condicion": condition,
                "acierto": is_hit(chunks, conv),
                "latencia_busqueda_ms": round(latency, 1),
                "tokens_embedder": max(count_tokens(q) for q in searched),
                "consulta": " || ".join(searched),
            })

    return pd.DataFrame(rows)


def summarize_hits(conversations: list[dict], hits: dict, conditions: tuple[str, ...], split: str | None) -> dict:

    groups = group_ids(conversations, split)
    with_history, without_history = groups["usa_historial"], groups["no_usa_historial"]
    count = {c: sum(hits[(i, c)] for i in with_history) for c in conditions}
    oracle = count["oraculo"]
    new = [c for c in conditions if c not in (*BASELINES, "oraculo")]

    return {
        "n_usa_historial": len(with_history),
        "n_no_usa_historial": len(without_history),
        "aciertos_usa_historial": count,
        "fraccion_del_oraculo": {c: round(count[c] / oracle, 3) if oracle else None for c in conditions},
        "aciertos_no_usa_historial": {c: sum(hits[(i, c)] for i in without_history) for c in conditions},
        "pares_usa_historial": [
            paired(hits, with_history, a, b)
                for b in new
                    for a in BASELINES + ("oraculo",)
        ] + [
            paired(hits, with_history, "literal", "oraculo"),
        ],
        "pares_no_usa_historial": [
            paired(hits, without_history, "literal", c)
                for c in conditions if c != "literal"
        ],
    }


def summarize_rewrites(rewrites: pd.DataFrame) -> dict:

    anchored = rewrites["terminos_clave"].map(json.loads)
    discarded = rewrites["terminos_descartados"].map(json.loads)
    fallbacks = rewrites["fallback"][rewrites["fallback"] != ""]
    latencies = rewrites["latencia_ms"].astype(float).tolist()
    model_flag = rewrites["necesita_historial_modelo"].astype(str).eq("True")
    gold_flag = rewrites["necesita_historial_gold"].astype(str).eq("True")

    return {
        "n": len(rewrites),
        "fallbacks": int(len(fallbacks)),
        "tasa_fallback": round(len(fallbacks) / len(rewrites), 3),
        "motivos_fallback": fallbacks.value_counts().to_dict(),
        "necesita_historial_coincide_con_gold": int((model_flag == gold_flag).sum()),
        "terminos_anclados": int(anchored.map(len).sum()),
        "terminos_descartados": int(discarded.map(len).sum()),
        "reescrituras_con_descartados": int(discarded.map(bool).sum()),
        "descartados": {
            r.conv_id: d
                for r, d in zip(rewrites.itertuples(), discarded, strict=True) if d
        },
        "latencia_p50_ms": round(percentile(latencies, 50), 1),
        "latencia_p95_ms": round(percentile(latencies, 95), 1),
        "tokens_in_p50": float(rewrites["tokens_in"].astype(float).median()),
        "tokens_out_p50": float(rewrites["tokens_out"].astype(float).median()),
    }


def summarize_failures(retrieval: pd.DataFrame, hits: dict, conditions: tuple[str, ...]) -> dict:

    queries = {(r.conv_id, r.condicion): r.consulta for r in retrieval.itertuples()}
    present = sorted(set(retrieval["conv_id"]) & set(PHASE2_FAILURES))

    return {
        cid: {
            "terminos": PHASE2_FAILURES[cid],
            "conserva_terminos": {
                c: all(norm(t) in norm(queries[(cid, c)]) for t in PHASE2_FAILURES[cid])
                    for c in conditions if c != "oraculo"
            },
            "aciertos": {c: hits[(cid, c)] for c in conditions},
        }
            for cid in present
    }


def summarize_costs(retrieval: pd.DataFrame) -> dict:

    out = {}

    for condition, group in retrieval.groupby("condicion"):

        latencies = group["latencia_busqueda_ms"].astype(float).tolist()

        out[condition] = {
            "latencia_busqueda_p50_ms": round(percentile(latencies, 50), 1),
            "tokens_embedder_p50": float(group["tokens_embedder"].astype(float).median()),
            "tokens_embedder_max": int(group["tokens_embedder"].astype(int).max()),
        }

    return out


def choose_winner(hits_summary: dict, rewrite_summary: dict) -> tuple[str | None, dict]:

    contamination = {(p["a"], p["b"]): p for p in hits_summary["pares_no_usa_historial"]}
    rule = {}

    for condition in CANDIDATES:

        losses = contamination[("literal", condition)]["solo_a"]

        rule[condition] = {
            "aciertos_usa_historial": hits_summary["aciertos_usa_historial"][condition],
            "perdidas_en_autocontenidas": losses,
            "tasa_fallback": rewrite_summary["tasa_fallback"],
            "elegible": losses == 0 and rewrite_summary["tasa_fallback"] < FALLBACK_MAX,
        }

    eligible = [c for c in CANDIDATES if rule[c]["elegible"]]

    if not eligible:
        return None, rule

    best = max(rule[c]["aciertos_usa_historial"] for c in eligible)

    return next(c for c in eligible if rule[c]["aciertos_usa_historial"] == best), rule


def combined_with_dev(conversations: list[dict], retrieval: pd.DataFrame, conditions: tuple[str, ...]) -> dict:

    dev = pd.read_csv(RESULTS_ROOT / "dev" / "retrieval.csv", sep=";", keep_default_na=False)
    both = pd.concat([dev[dev["condicion"].isin(conditions)], retrieval])

    return summarize_hits(conversations, hit_table(both), conditions, None)


def log_to_mlflow(hits_summary: dict, conditions: tuple[str, ...]) -> None:

    for condition in conditions:

        params = {
            "dataset": DATASET.name,
            "split": SPLIT,
            "collection": COLLECTION,
            "method": METHOD,
            "top_k": TOP_K,
            "condition": condition,
            "rewrite_model": REWRITE_MODEL if condition in CANDIDATES else "",
            "rewrite_prompt": REWRITE_PROMPT_VERSION if condition in CANDIDATES else "",
        }

        with tracked_run(STAGE, f"{REWRITE_PROMPT_VERSION}-{SPLIT}-{condition}", params):
            log_metrics({
                "hits_needs_history": hits_summary["aciertos_usa_historial"][condition],
                "n_needs_history": hits_summary["n_usa_historial"],
                "hits_no_history": hits_summary["aciertos_no_usa_historial"][condition],
                "n_no_history": hits_summary["n_no_usa_historial"],
            })


def main() -> dict:

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "summary.json"

    if summary_path.exists() and not REUSE:
        raise SystemExit(f"{summary_path} ya existe; cambiar EXP para no sobrescribir resultados.")

    if SPLIT == "test" and not (RESULTS_ROOT / "dev" / "summary.json").exists():
        raise SystemExit("Falta la corrida de dev: la condición de test se elige ahí.")

    everything = load_conversations(DATASET)
    conversations = [c for c in everything if c["split"] == SPLIT]
    conditions = conditions_for_split()

    rewriter = build_llm("ollama", REWRITE_MODEL, max_tokens=REWRITE_NUM_PREDICT, repeat_penalty=None)

    rewrites = cached(RESULTS_DIR / "rewrites.csv", lambda: build_rewrites_v2(conversations, rewriter), REUSE)
    retrieval = cached(RESULTS_DIR / "retrieval.csv", lambda: build_retrieval(conversations, rewrites, conditions), REUSE)
    hits = hit_table(retrieval)
    hits_summary = summarize_hits(conversations, hits, conditions, SPLIT)
    rewrite_summary = summarize_rewrites(rewrites)

    summary = {
        "exp": EXP,
        "prompt": REWRITE_PROMPT_VERSION,
        "split": SPLIT,
        "modelo": REWRITE_MODEL,
        "metodo": METHOD,
        "n_conversaciones": len(conversations),
        "condiciones": list(conditions),
        "aciertos": hits_summary,
        "reescritura": rewrite_summary,
        "fallos_fase2": summarize_failures(retrieval, hits, conditions),
        "costo_busqueda": summarize_costs(retrieval),
    }

    if SPLIT == "dev":
        summary["ganadora"], summary["regla_eleccion"] = choose_winner(hits_summary, rewrite_summary)
    else:
        summary["ganadora_dev"] = conditions[len(BASELINES)]
        summary["dev_mas_test"] = combined_with_dev(everything, retrieval, conditions)

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    log_to_mlflow(hits_summary, conditions)

    return summary


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=1))
