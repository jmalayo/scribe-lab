import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

from experiments.conversation.common import (
    ANSWER_MODEL,
    BOOTSTRAP_RESAMPLES,
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
    hit_table,
    hits_by_type,
    is_hit,
    load_conversations,
    paired,
    search,
)
from shared.eval.grounding import abstained
from shared.eval.metrics import mcnemar_exact, percentile, wilson_ci
from shared.ingest import build_qdrant_client, get_embedder
from shared.llm import LLMClient, build_llm
from shared.pipeline import Pipeline
from shared.prompts import ANSWER_PROMPTS, build_context_text
from shared.tracking import log_metrics, tracked_run

EXP = "multiturno_v2"
SPLIT = "test"
DATASET = Path("shared/eval/multiturn/multiturno_v2.jsonl")
V1_RETRIEVAL = RESULTS_BASE / "multiturno_v1" / "retrieval.csv"
V1_PHASE3 = RESULTS_BASE / "multiturno_v1_fase3" / REWRITE_PROMPT_VERSION
RESULTS_ROOT = RESULTS_BASE / EXP
RESULTS_DIR = RESULTS_ROOT / SPLIT
METHODS = ("hybrid_rrf", "dense")
CONDITIONS = ("literal", "concat", "reescritura_v2", "multi_consulta", "oraculo")
ANSWERED = ("literal", "multi_consulta")
ORACLE_GAP_MAX = 10.0
AUTO_LOSSES_MAX = 2
SEED = 42
REUSE = True


def group_of(conv: dict) -> str:

    target = conv["objetivo"]

    if target["debe_abstenerse"]:
        return "sin_respuesta"

    return "historial" if target["usa_historial"] else "autocontenida"


def build_pipelines(llm: LLMClient | None = None) -> dict[str, Pipeline]:

    base = base_config()
    client, embedder = build_qdrant_client(), get_embedder()

    return {
        m: Pipeline(base.model_copy(update={"method": m}), client, embedder, llm, None)
            for m in METHODS
    }


def queries_for(conversations: list[dict], rewrites: pd.DataFrame) -> dict[str, dict[str, str]]:

    rewritten = {r.conv_id: r.pregunta for r in rewrites.itertuples()}
    queries = {}

    for conv in conversations:

        cid = conv["conv_id"]
        queries[cid] = {**base_queries(conv), "reescritura_v2": rewritten[cid]}

    return queries


def build_retrieval(conversations: list[dict], queries: dict) -> pd.DataFrame:

    pipelines = build_pipelines()
    rows = []

    for method, pipeline in pipelines.items():

        pipeline.retrieve("calentamiento", top_k=TOP_K)

        for conv in conversations:
            for condition in CONDITIONS:

                chunks, latency = search(pipeline, queries[conv["conv_id"]], condition)

                rows.append({
                    "conv_id": conv["conv_id"],
                    "ref_id": conv["objetivo"]["ref_id"],
                    "tipo": conv["tipo"],
                    "grupo": group_of(conv),
                    "metodo": method,
                    "condicion": condition,
                    "acierto": is_hit(chunks, conv),
                    "latencia_busqueda_ms": round(latency, 1),
                })

    return pd.DataFrame(rows)


def build_abstention(conversations: list[dict], queries: dict, llm: LLMClient) -> pd.DataFrame:

    pipeline = build_pipelines(llm)["hybrid_rrf"]
    prompt = ANSWER_PROMPTS[pipeline.config.prompt_version]
    rows = []

    for conv in conversations:

        if group_of(conv) != "sin_respuesta":
            continue

        for condition in ANSWERED:

            question = queries[conv["conv_id"]]["literal" if condition == "literal" else "reescritura_v2"]
            chunks, _ = search(pipeline, queries[conv["conv_id"]], condition)
            answer = pipeline.llm.complete(prompt.format(context=build_context_text(chunks), question=question)).text

            rows.append({
                "conv_id": conv["conv_id"],
                "condicion": condition,
                "pregunta": question,
                "respuesta": answer,
                "abstencion": abstained(answer),
            })

    return pd.DataFrame(rows)


def clustered(hits: dict, ids: list[str], clusters: dict[str, str], a: str, b: str) -> dict:

    by_cluster = defaultdict(list)

    for i in ids:
        by_cluster[clusters[i]].append(float(hits[(i, b)]) - float(hits[(i, a)]))

    sums = [sum(v) for v in by_cluster.values()]
    rng = random.Random(SEED)
    groups = list(by_cluster.values())
    means = []

    for _ in range(BOOTSTRAP_RESAMPLES):

        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        means.append(sum(sum(g) for g in sample) / sum(len(g) for g in sample))

    means.sort()
    only_a, only_b = sum(s < 0 for s in sums), sum(s > 0 for s in sums)

    return {
        "objetivos": len(groups),
        "objetivos_a_favor_a": only_a,
        "objetivos_a_favor_b": only_b,
        "p_signo_por_objetivo": mcnemar_exact(only_a, only_b),
        "ic95_bootstrap_por_objetivo": (
            round(means[int(0.025 * BOOTSTRAP_RESAMPLES)], 4),
            round(means[int(0.975 * BOOTSTRAP_RESAMPLES) - 1], 4),
        ),
    }


def summarize_method(conversations: list[dict], hits: dict) -> dict:

    clusters = {c["conv_id"]: c["objetivo"]["ref_id"] for c in conversations}
    groups = defaultdict(list)

    for c in conversations:
        groups[group_of(c)].append(c["conv_id"])

    history, auto = groups["historial"], groups["autocontenida"]
    types = sorted({c["tipo"] for c in conversations if group_of(c) != "sin_respuesta"})
    count = {c: sum(hits[(i, c)] for i in history) for c in CONDITIONS}
    pairs = [("literal", c) for c in CONDITIONS if c != "literal"] + [
        ("concat", "multi_consulta"),
        ("reescritura_v2", "multi_consulta"),
        ("multi_consulta", "oraculo"),
    ]

    return {
        "n_historial": len(history),
        "objetivos_historial": len({clusters[i] for i in history}),
        "n_autocontenidas": len(auto),
        "aciertos_historial": count,
        "recall_historial": {c: round(count[c] / len(history), 3) for c in CONDITIONS},
        "aciertos_autocontenidas": {c: sum(hits[(i, c)] for i in auto) for c in CONDITIONS},
        "pares_historial": [
            {**paired(hits, history, a, b), **clustered(hits, history, clusters, a, b)}
                for a, b in pairs
        ],
        "pares_autocontenidas": [
            {**paired(hits, auto, "literal", c), **clustered(hits, auto, clusters, "literal", c)}
                for c in CONDITIONS if c != "literal"
        ],
        "por_tipo": hits_by_type(conversations, hits, types, CONDITIONS),
    }


def summarize_abstention(abstention: pd.DataFrame) -> dict:

    out = {}

    for condition, group in abstention.groupby("condicion"):

        flags = group["abstencion"].astype(str).eq("True")
        hits = int(flags.sum())

        out[condition] = {
            "abstenciones": hits,
            "n": len(group),
            "tasa_abstencion": round(hits / len(group), 3),
            "tasa_alucinacion": round(1 - hits / len(group), 3),
            "ic95_wilson": wilson_ci(hits, len(group)),
        }

    return out


def acceptance(summary: dict, abstention: dict) -> dict:

    block = summary["hybrid_rrf"]
    recall = block["recall_historial"]
    history_pairs = {(p["a"], p["b"]): p for p in block["pares_historial"]}
    auto_pairs = {(p["a"], p["b"]): p for p in block["pares_autocontenidas"]}
    vs_literal = history_pairs[("literal", "multi_consulta")]
    losses = auto_pairs[("literal", "multi_consulta")]["solo_a"]
    gap = round((recall["oraculo"] - recall["multi_consulta"]) * 100, 1)

    return {
        "c1_multi_consulta_no_peor_que_literal": recall["multi_consulta"] >= recall["literal"],
        "c1_significativo_por_objetivo": vs_literal["p_signo_por_objetivo"] < 0.05
            and vs_literal["objetivos_a_favor_b"] > vs_literal["objetivos_a_favor_a"],
        "c2_distancia_al_oraculo_puntos": gap,
        "c2_distancia_al_oraculo_max_10": gap <= ORACLE_GAP_MAX,
        "c4_perdidas_autocontenidas": losses,
        "c4_perdidas_max_2": losses <= AUTO_LOSSES_MAX,
        "c5_abstencion_no_baja": abstention["multi_consulta"]["abstenciones"] >= abstention["literal"]["abstenciones"],
    }


def summarize_costs(rewrites: pd.DataFrame, retrieval: pd.DataFrame) -> dict:

    latencies = rewrites["latencia_ms"].astype(float).tolist()
    hybrid = retrieval[retrieval["metodo"] == "hybrid_rrf"]

    return {
        "reescritura_latencia_p50_ms": round(percentile(latencies, 50), 1),
        "reescritura_latencia_p95_ms": round(percentile(latencies, 95), 1),
        "reescritura_tokens_in_p50": float(rewrites["tokens_in"].astype(float).median()),
        "reescritura_tokens_out_p50": float(rewrites["tokens_out"].astype(float).median()),
        "fallbacks": rewrites["fallback"][rewrites["fallback"] != ""].value_counts().to_dict(),
        "tokens_embedder_max_reescritura": int(rewrites["pregunta"].map(count_tokens).max()),
        "busqueda_p50_ms": {
            c: round(percentile(g["latencia_busqueda_ms"].astype(float).tolist(), 50), 1)
                for c, g in hybrid.groupby("condicion")
        },
    }


def revalidate_v1_oracle(conversations: list[dict]) -> dict:

    redo = [c for c in conversations if c.get("oraculo_a_rehacer")]
    pipelines = build_pipelines()
    old = pd.read_csv(V1_RETRIEVAL, sep=";", keep_default_na=False)
    old = old[old["condicion"] == "oraculo"]
    old_hits = {(r.conv_id, r.metodo): r.acierto in (True, "True") for r in old.itertuples()}
    rows = {}

    for conv in redo:

        cid = conv["conv_id"]
        oracle = conv["objetivo"]["pregunta_autonoma"]
        row = {
            "split": conv["split"],
            "pregunta_autonoma_v1": conv["objetivo"]["pregunta_autonoma_v1"],
            "pregunta_autonoma": oracle,
        }

        for m in METHODS:
            row[f"{m}_antes"] = old_hits[(cid, m)]

        for m in METHODS:
            chunks = pipelines[m].retrieve(oracle, top_k=TOP_K).chunks
            row[f"{m}_ahora"] = is_hit(chunks, conv)

        rows[cid] = row

    v1_history = old[old["conv_id"].isin([c["conv_id"] for c in conversations if group_of(c) == "historial"])]
    totals = {}

    for m in METHODS:

        before = int(v1_history[v1_history["metodo"] == m]["acierto"].astype(str).eq("True").sum())
        delta = sum(int(r[f"{m}_ahora"]) - int(r[f"{m}_antes"]) for r in rows.values())
        totals[m] = {"oraculo_fase2_antes": before, "oraculo_fase2_ahora": before + delta}

    phase3 = {}

    for split in ("dev", "test"):

        summary = json.loads((V1_PHASE3 / split / "summary.json").read_text(encoding="utf-8"))
        before = summary["aciertos"]["aciertos_usa_historial"]["oraculo"]
        delta = sum(
            int(r["hybrid_rrf_ahora"]) - int(r["hybrid_rrf_antes"])
                for r in rows.values() if r["split"] == split
        )
        phase3[split] = {"oraculo_antes": before, "oraculo_ahora": before + delta}

    return {"conversaciones": rows, "fase2_28": totals, "fase3": phase3}


def log_to_mlflow(summary: dict) -> None:

    block = summary["recuperacion"]["hybrid_rrf"]

    for condition in CONDITIONS:

        params = {
            "dataset": DATASET.name,
            "split": SPLIT,
            "collection": COLLECTION,
            "method": "hybrid_rrf",
            "top_k": TOP_K,
            "condition": condition,
            "rewrite_model": REWRITE_MODEL if condition in ("reescritura_v2", "multi_consulta") else "",
            "rewrite_prompt": REWRITE_PROMPT_VERSION if condition in ("reescritura_v2", "multi_consulta") else "",
        }

        with tracked_run(STAGE, f"{EXP}-{SPLIT}-{condition}", params):
            log_metrics({
                "hits_needs_history": block["aciertos_historial"][condition],
                "n_needs_history": block["n_historial"],
                "hits_no_history": block["aciertos_autocontenidas"][condition],
                "n_no_history": block["n_autocontenidas"],
            })


def main() -> dict:

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "summary.json"

    if summary_path.exists() and not REUSE:
        raise SystemExit(f"{summary_path} ya existe; cambiar EXP para no sobrescribir resultados.")

    everything = load_conversations(DATASET)
    conversations = [c for c in everything if c["split"] == SPLIT]

    rewriter = build_llm("ollama", REWRITE_MODEL, max_tokens=REWRITE_NUM_PREDICT, repeat_penalty=None)
    answer = build_llm("ollama", ANSWER_MODEL, max_tokens=base_config().llm.max_tokens)

    rewrites = cached(RESULTS_DIR / "rewrites.csv", lambda: build_rewrites_v2(conversations, rewriter), REUSE)
    queries = queries_for(conversations, rewrites)
    retrieval = cached(RESULTS_DIR / "retrieval.csv", lambda: build_retrieval(conversations, queries), REUSE)
    abstention_rows = cached(RESULTS_DIR / "abstention.csv", lambda: build_abstention(conversations, queries, answer), REUSE)

    recovery = {m: summarize_method(conversations, hit_table(retrieval, m)) for m in METHODS}
    abstention = summarize_abstention(abstention_rows)

    summary = {
        "exp": EXP,
        "split": SPLIT,
        "prompt": REWRITE_PROMPT_VERSION,
        "modelo_reescritura": REWRITE_MODEL,
        "n_conversaciones": len(conversations),
        "recuperacion": recovery,
        "abstencion": abstention,
        "criterios": acceptance(recovery, abstention),
        "costo": summarize_costs(rewrites, retrieval),
        "oraculo_v1_revalidado": revalidate_v1_oracle(everything),
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    log_to_mlflow(summary)

    return summary


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=1))
