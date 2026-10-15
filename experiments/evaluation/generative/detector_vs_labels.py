import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)


from shared.eval.data import load_answers, load_chunks
from shared.eval.grounding import (
    detect_tokens,
    ground_answer,
    nli_http_predictor,
    nli_scores,
    sentence_scores,
    split_sentences,
)
from shared.eval.metrics import bootstrap_ci, percentile
from shared.settings import settings

OUT_DIR = Path(__file__).resolve().parent / "llama_3_1_8_B" / "grounding_v3"
ANSWERS = REPO_ROOT / "experiments/evaluation/benchmark-models/results/exp_7/llama31_base/base_answers.csv"
CHUNKS = REPO_ROOT / "experiments/evaluation/benchmark-models/results/exp_7/llama31_base/base_chunks.csv"
LABELS_SENTENCES = Path(__file__).resolve().parent / "llama_3_1_8_B/labels_sentences.csv"
LABELS_ANSWERS = Path(__file__).resolve().parent / "llama_3_1_8_B/labels_answers.csv"

LETTUCE_THRESHOLD = 0.5
ENTAILMENT_THRESHOLD = 0.5
CONTRADICTION_THRESHOLD = 0.5
POSITIVE = {
    "no_soportada", 
    "contradice"
}

RUN_ABSTENTION = False

def score_sentences(predict) -> tuple[pd.DataFrame, pd.DataFrame]:

    answers = load_answers(ANSWERS)
    chunks_by_id = load_chunks(CHUNKS)

    labels = pd.read_csv(LABELS_SENTENCES, sep=";")
    rows, timings = [], []

    for r in answers.itertuples(): # 'itertuples' consume menos
        
        chunks = chunks_by_id[r.id]
        human = labels[labels["id"] == r.id].sort_values("oracion_idx")

        sentences = split_sentences(r.answer)

        if sentences != human["oracion"].tolist():
            raise ValueError(f"{r.id}: las oraciones no coinciden con la etiqueta humana")

        t0 = time.perf_counter()
        tokens = detect_tokens(settings.detector_url, chunks, r.question, r.answer)

        t1 = time.perf_counter()
        nli = nli_scores(sentences, chunks, predict)

        t2 = time.perf_counter()

        result = ground_answer(
            answer = r.answer, 
            chunks = chunks, 
            question = r.question, 
            tokens = tokens, 
            nli = nli,
            threshold = LETTUCE_THRESHOLD, 
            entailment_threshold = ENTAILMENT_THRESHOLD,
            contradiction_threshold = CONTRADICTION_THRESHOLD
        )

        t3 = time.perf_counter()

        timings.append({
            "id": r.id, 
            "n_sentences": len(sentences), 
            "n_chunks": len(chunks),
            "lettuce_s": t1 - t0, 
            "nli_s": t2 - t1, 
            "ground": t3 - t2, 
            "total_s": t3 - t0
        })

        flagged = {
            u["start"]: u 
                for u in result["unsupported_sentences"]
        }

        for s, n, h in zip(sentence_scores(r.answer, tokens), nli, human.itertuples(), strict=True):
            u = flagged.get(s["start"], {})
            rows.append({
                "id": r.id,
                "oracion_idx": h.oracion_idx,
                "humano_sustento": h.humano_sustento,
                "lettuce_score": round(s["score"], 4),
                "nli_entailment": round(n["entailment"], 4),
                "nli_contradiction": round(n["contradiction"], 4),
                "numeros_sin_sustento": " ".join(u.get("numbers", [])),
                "razones": " ".join(u.get("reasons", [])),
                "oracion": s["sentence"]
            })

    timing_df = pd.DataFrame(timings)
    sentence_df = pd.DataFrame(rows)

    # Debug: Filtro para saber cuantos casos debería filtrar LettuceDetect hasta en este breakpoint return
    # df[df["humano_sustento"].isin(["no_soportada", "contradice"])][["humano_sustento", "razones"]] 

    return sentence_df, timing_df

def detection(flags: pd.Series, positive: pd.Series) -> dict:
    
    tp = int((flags & positive).sum())
    fp = int((flags & ~positive).sum())
    fn = int((~flags & positive).sum())

    return {
        "tp": tp, 
        "fp": fp, 
        "fn": fn, 
        "tn": int((~flags & ~positive).sum()),
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "precision_ci95": (
            bootstrap_ci([
                float(p) 
                    for p in positive[flags]
            ]) if tp + fp else None
        ),
        "recall": round(tp / (tp + fn), 3),
        "recall_ci95": (
            bootstrap_ci([
                float(f) 
                    for f in flags[positive]
            ])
        ),
    }

def auroc(scores: pd.Series, positive: pd.Series) -> float:
    pos = scores[positive].tolist()
    neg = scores[~positive].tolist()

    return round(
        sum(
            (p > q) + 0.5 * (p == q) 
                for p in pos 
                    for q in neg) / (len(pos) * len(neg)
        ), 3
    )

def ranges(df: pd.DataFrame, column: str) -> dict:
    return {
        label: {
            "n": len(g), 
            "min": float(g.min()), 
            "median": round(float(g.median()), 4), 
            "max": float(g.max())
        } for label, g in df.groupby("humano_sustento")[column]
    }

def summarize_sentences(df: pd.DataFrame) -> dict:
    df = df[df["humano_sustento"] != "no_aplica"]
    positive = df["humano_sustento"].isin(POSITIVE)
    reasons = df["razones"].fillna("")

    signals = {
        "lettuce": reasons.str.contains("lettucedetect"),
        "numeric": reasons.str.contains("numeric"),
        "nli_contradiction": reasons.str.contains("nli_contradiction"),
        "nli_not_entailed": reasons.str.contains("nli_not_entailed"),
    }
    signals["nli"] = signals["nli_contradiction"] | signals["nli_not_entailed"]
    signals["lettuce_or_nli"] = signals["lettuce"] | signals["nli"]
    signals["all"] = reasons != ""

    by_answer = df.assign(pos=positive, flag=signals["all"]).groupby("id")[["pos", "flag"]].any()

    return {
        "n_sentences": len(df),
        "n_positive": int(positive.sum()),
        "thresholds": {
            "lettuce": LETTUCE_THRESHOLD, 
            "entailment": ENTAILMENT_THRESHOLD,
            "contradiction": CONTRADICTION_THRESHOLD
        },
        "ranges": {
            "lettuce_score": ranges(df, "lettuce_score"),
            "nli_entailment": ranges(df, "nli_entailment"),
            "nli_contradiction": ranges(df, "nli_contradiction"),
        },
        "auroc": {
            "lettuce_score": auroc(df["lettuce_score"], positive),
            "nli_1_minus_entailment": auroc(1 - df["nli_entailment"], positive),
            "nli_contradiction": auroc(df["nli_contradiction"], positive),
        },
        "detection": {name: detection(flags, positive) for name, flags in signals.items()},
        "answers_with_problem": int(by_answer["pos"].sum()),
        "answers_flagged_correctly": int((by_answer["pos"] & by_answer["flag"]).sum()),
        "answers_flagged_without_problem": int((~by_answer["pos"] & by_answer["flag"]).sum()),
    }


def summarize_timings(t: pd.DataFrame) -> dict:
    
    def stats(col: str) -> dict:

        values = t[col].tolist()

        return {
            "p50_s": round(percentile(values, 50), 2), 
            "p95_s": round(percentile(values, 95), 2),
            "max_s": round(max(values), 2), 
            "total_s": round(sum(values), 1)
        }

    return {
        "n_questions": len(t),
        "per_question": {
            col: stats(col) 
                for col in ["lettuce_s", "nli_s", "ground", "total_s"]
        },
    }


def abstention_scores() -> pd.DataFrame:

    labels_df = pd.read_csv(LABELS_ANSWERS, sep=";")
    no_answer = set(labels_df.loc[labels_df["contexto_suficiente"] == "n.a.", "id"])
    max_score = pd.read_csv(CHUNKS, sep=";").groupby("id", sort=False)["score"].max()

    return pd.DataFrame({
        "id": max_score.index,
        "sin_respuesta_en_corpus": max_score.index.isin(no_answer),
        "max_score": max_score.round(4).to_numpy(),
    })

def main():
    
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    predict = nli_http_predictor(settings.nli_url)

    sentences, timings = score_sentences(predict)

    sentences.to_csv(OUT_DIR / "sentences.csv", sep=";", index=False)

    summary = {
        "sentences": summarize_sentences(sentences), 
        "timings": summarize_timings(timings)
    }

    if RUN_ABSTENTION:

        abst = abstention_scores()
        abst.to_csv(OUT_DIR / "abstention.csv", sep=";", index=False)

        summary["abstention"] = {
            str(label): {
                "n": len(g), 
                "min": float(g.min()), 
                "max": float(g.max())
            } for label, g in abst.groupby("sin_respuesta_en_corpus")["max_score"]
        }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
