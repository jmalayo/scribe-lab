import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from shared.eval.data import load_questions
from shared.eval.grounding import abstained
from shared.eval.metrics import auroc, is_chunk_correct, percentile, wilson_ci
from shared.ingest import build_qdrant_client, get_embedder
from shared.llm import build_llm
from shared.pipeline import AbstentionConfig, MemoryConfig, Pipeline, PipelineConfig
from shared.retrieval import rerank
from shared.settings import settings
from shared.tracking import log_metrics, tracked_run

EXP = "abstencion_v1"
STAGE = "abstention"
RESULTS_DIR = Path("experiments/abstention/results") / EXP
COLLECTION = "exp_paired_293_73"
ANSWER_MODEL = "llama3.2:3b"
SCORES = {
    "embedder": "max_dense_score",
    "cross_encoder": "max_rerank_score",
    "rrf": "max_rrf_score",
}
SPLITS = ("calibracion", "verificacion")
MAX_FALSE_ABSTENTION = 0.05
SEED = 42


def assign_splits(questions: list[dict]) -> dict[str, str]:

    strata = defaultdict(list)

    for q in questions:
        strata[(bool(q["gold_spans"]), q["tipo"])].append(q["id"])

    rng = random.Random(SEED)
    splits = {}

    for key in sorted(strata):

        ids = sorted(strata[key])
        rng.shuffle(ids)

        for position, qid in enumerate(ids):
            splits[qid] = SPLITS[position % 2]

    return splits


def build_pipeline() -> Pipeline:

    config = PipelineConfig.from_yaml(settings.pipeline_config_path).model_copy(
        update={
            "collection": COLLECTION,
            "memory": MemoryConfig(),
            "abstention": AbstentionConfig(),
        }
    )
    llm = build_llm("ollama", ANSWER_MODEL, max_tokens=config.llm.max_tokens)

    return Pipeline(config, build_qdrant_client(), get_embedder(), llm, None)


def best(chunks: list[dict], key: str) -> float:

    values = [c[key] for c in chunks if key in c]

    return max(values) if values else float("nan")


def build_runs(questions: list[dict], splits: dict[str, str]) -> pd.DataFrame:

    pipeline = build_pipeline()
    pipeline.run("calentamiento")
    rerank("calentamiento", [{"text": "calentamiento"}], 1)
    rows = []

    for q in questions:

        result = pipeline.run(q["question"])

        t0 = time.perf_counter()
        reranked = rerank(q["question"], result.chunks, len(result.chunks))
        rerank_ms = (time.perf_counter() - t0) * 1000

        rows.append({
            "id": q["id"],
            "tipo": q["tipo"],
            "sin_respuesta": not q["gold_spans"],
            "split": splits[q["id"]],
            "max_dense_score": best(result.chunks, "dense_score"),
            "max_rerank_score": best(reranked, "rerank_score"),
            "max_rrf_score": best(result.chunks, "rrf_score"),
            "recupera_gold": any(is_chunk_correct(c, q) for c in result.chunks) if q["gold_spans"] else False,
            "abstencion_llm": abstained(result.answer),
            "latencia_busqueda_ms": round(result.latencies_ms["embedding"] + result.latencies_ms["retrieval"], 1),
            "latencia_rerank_ms": round(rerank_ms, 1),
            "latencia_llm_ms": result.latencies_ms["llm"],
            "respuesta": result.answer,
        })

    return pd.DataFrame(rows)


def flags(df: pd.DataFrame, column: str) -> pd.Series:

    return df[column].astype(str).eq("True")


def choose_threshold(calibration: pd.DataFrame, column: str) -> dict:

    unanswerable = flags(calibration, "sin_respuesta")
    scores = calibration[column].astype(float)
    allowed = math.floor(MAX_FALSE_ABSTENTION * int((~unanswerable).sum()))
    best_rule = {
        "umbral": None,
        "atrapadas": 0,
        "bloqueadas": 0,
    }

    for threshold in sorted(scores.unique()):

        abstains = scores < threshold
        caught = int((abstains & unanswerable).sum())
        blocked = int((abstains & ~unanswerable).sum())

        if blocked > allowed:
            break

        if caught > best_rule["atrapadas"]:
            best_rule = {
                "umbral": float(threshold),
                "atrapadas": caught,
                "bloqueadas": blocked,
            }

    best_rule["bloqueos_permitidos"] = allowed

    return best_rule


def apply_threshold(df: pd.DataFrame, column: str, threshold: float | None) -> dict:

    unanswerable = flags(df, "sin_respuesta")
    llm_abstains = flags(df, "abstencion_llm")
    retrieved_gold = flags(df, "recupera_gold")
    if threshold is None:
        abstains = pd.Series(False, index=df.index)
    else:
        abstains = df[column].astype(float) < threshold

    n_unanswerable = int(unanswerable.sum())
    n_answerable = int((~unanswerable).sum())
    caught = int((abstains & unanswerable).sum())
    blocked = int((abstains & ~unanswerable).sum())

    return {
        "sin_respuesta_n": n_unanswerable,
        "con_respuesta_n": n_answerable,
        "ganadas": caught,
        "ganadas_ic95": wilson_ci(caught, n_unanswerable),
        "perdidas": blocked,
        "perdidas_ic95": wilson_ci(blocked, n_answerable),
        "alucinaciones_evitadas": int((abstains & unanswerable & ~llm_abstains).sum()),
        "respuestas_perdidas": int((abstains & ~unanswerable & ~llm_abstains).sum()),
        "respuestas_perdidas_con_gold": int((abstains & ~unanswerable & ~llm_abstains & retrieved_gold).sum()),
        "ids_ganadas": sorted(df.loc[abstains & unanswerable, "id"]),
        "ids_perdidas": sorted(df.loc[abstains & ~unanswerable, "id"]),
    }


def separation(df: pd.DataFrame) -> dict:

    unanswerable = flags(df, "sin_respuesta").tolist()
    out = {}

    for name, column in SCORES.items():

        scores = [-float(s) for s in df[column]]
        out[name] = auroc(scores, unanswerable)

    return out


def score_ranges(df: pd.DataFrame) -> dict:

    unanswerable = flags(df, "sin_respuesta")
    out = {}

    for name, column in SCORES.items():

        scores = df[column].astype(float)

        out[name] = {
            "sin_respuesta": (round(scores[unanswerable].min(), 4), round(scores[unanswerable].max(), 4)),
            "con_respuesta": (round(scores[~unanswerable].min(), 4), round(scores[~unanswerable].max(), 4)),
        }

    return out


def llm_only(df: pd.DataFrame) -> dict:

    unanswerable = flags(df, "sin_respuesta")
    llm_abstains = flags(df, "abstencion_llm")
    retrieved_gold = flags(df, "recupera_gold")
    correct = int((llm_abstains & unanswerable).sum())
    wrong = int((llm_abstains & ~unanswerable).sum())

    return {
        "sin_respuesta_abstiene": correct,
        "sin_respuesta_abstiene_ic95": wilson_ci(correct, int(unanswerable.sum())),
        "con_respuesta_abstiene": wrong,
        "con_respuesta_abstiene_ic95": wilson_ci(wrong, int((~unanswerable).sum())),
        "con_respuesta_abstiene_con_gold": int((llm_abstains & ~unanswerable & retrieved_gold).sum()),
        "con_respuesta_sin_gold": int((~unanswerable & ~retrieved_gold).sum()),
    }


def llm_misses(df: pd.DataFrame) -> dict:

    unanswerable = flags(df, "sin_respuesta")
    missed = df[unanswerable & ~flags(df, "abstencion_llm")]
    answerable = df[~unanswerable]
    out = {}

    for _, row in missed.iterrows():

        scores = {"split": row["split"]}

        for name, column in SCORES.items():

            value = float(row[column])
            below = float((answerable[column].astype(float) < value).mean())

            scores[name] = {
                "score": round(value, 4),
                "con_respuesta_por_debajo": round(below, 3),
            }

        out[row["id"]] = scores

    return out


def costs(df: pd.DataFrame) -> dict:

    out = {}

    for column in ("latencia_busqueda_ms", "latencia_rerank_ms", "latencia_llm_ms"):

        values = df[column].astype(float).tolist()

        out[column] = {
            "p50": round(percentile(values, 50), 1),
            "p95": round(percentile(values, 95), 1),
        }

    return out


def log_to_mlflow(summary: dict) -> None:

    for name, column in SCORES.items():

        rule = summary["umbrales"][name]
        result = summary["verificacion"][name]
        params = {
            "collection": COLLECTION,
            "score": column,
            "threshold": rule["umbral"],
            "max_false_abstention": MAX_FALSE_ABSTENTION,
            "answer_model": ANSWER_MODEL,
        }

        with tracked_run(STAGE, f"{EXP}-{name}", params):
            log_metrics({
                "auroc": summary["auroc"]["todas"][name],
                "caught_unanswerable": result["ganadas"],
                "n_unanswerable": result["sin_respuesta_n"],
                "blocked_answerable": result["perdidas"],
                "n_answerable": result["con_respuesta_n"],
                "hallucinations_avoided": result["alucinaciones_evitadas"],
                "answers_lost": result["respuestas_perdidas"],
            })


def summarize(runs: pd.DataFrame) -> dict:

    calibration = runs[runs["split"] == "calibracion"]
    verification = runs[runs["split"] == "verificacion"]

    thresholds = {
        name: choose_threshold(calibration, column)
            for name, column in SCORES.items()
    }

    summary = {
        "exp": EXP,
        "n_preguntas": len(runs),
        "n_por_split": runs.groupby(["split", "sin_respuesta"]).size().to_dict(),
        "auroc": {
            "todas": separation(runs),
            "calibracion": separation(calibration),
            "verificacion": separation(verification),
        },
        "rangos": score_ranges(runs),
        "umbrales": thresholds,
        "verificacion": {
            name: apply_threshold(verification, column, thresholds[name]["umbral"])
                for name, column in SCORES.items()
        },
        "solo_llm": {
            "todas": llm_only(runs),
            "verificacion": llm_only(verification),
            "sin_respuesta_no_se_abstiene": llm_misses(runs),
        },
        "costo": costs(runs),
    }

    summary["n_por_split"] = {f"{k[0]}|{k[1]}": v for k, v in summary["n_por_split"].items()}

    return summary


def main() -> dict:

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "summary.json"

    if summary_path.exists():
        raise SystemExit(f"{summary_path} ya existe; cambiar EXP para no sobrescribir resultados.")

    questions = load_questions()
    runs = build_runs(questions, assign_splits(questions))
    summary = summarize(runs)

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log_to_mlflow(summary)

    return summary


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=1, default=str))
