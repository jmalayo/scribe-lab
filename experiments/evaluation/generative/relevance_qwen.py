import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from experiments.evaluation.prompts import prompt_v2
from shared.eval.data import load_answers
from shared.eval.metrics import bootstrap_ci, percentile
from shared.settings import settings

OUT_DIR = Path(__file__).resolve().parent / "llama_3_1_8_B" / "relevance_qwen3"
ANSWERS = REPO_ROOT / "experiments/evaluation/benchmark-models/results/exp_7/llama31_base/base_answers.csv"
LABELS_ANSWERS = Path(__file__).resolve().parent / "llama_3_1_8_B/labels_answers.csv"

RELEVANCE_PROMPT, RELEVANCE_SCHEMA = prompt_v2()

OPTIONS = {
    "temperature": 0.0, 
    "num_ctx": 4096, 
    "num_predict": 3000
}

MODES = {
    "think": {
        "think": True
    },
    "json": {
        "think": False, 
        "format": RELEVANCE_SCHEMA
    }
}

def parse(text: str) -> dict | None:

    match = re.search(r"\{.*\}", text or "", re.DOTALL)

    if match is None:
        return None

    try:
        data = json.loads(match.group(0))

    except json.JSONDecodeError:
        return None

    if not isinstance(data.get("relevante"), bool):
        return None

    return data


def judge(mode: str, answers: pd.DataFrame) -> list[dict]:
    rows = []

    for r in answers.itertuples():

        prompt = RELEVANCE_PROMPT.format(question=r.question, answer=r.answer or "(vacía)")

        t0 = time.perf_counter()
        
        resp = requests.post(
            f"{settings.llm_host}/api/generate", 
            timeout=600, 
            json={
                "model": settings.llm_model_judge, 
                "prompt": prompt, 
                "stream": False, 
                "options": OPTIONS, 
                **MODES[mode]
            }
        )

        resp.raise_for_status()

        latency = time.perf_counter() - t0

        body = resp.json()

        verdict = parse(body.get("response", ""))

        rows.append({
            "id": r.id,
            "modo": mode,
            "valido": verdict is not None,
            "qwen_relevante": verdict["relevante"] if verdict else None,
            "qwen_motivo": verdict.get("motivo", "") if verdict else "",
            "latencia_s": round(latency, 2),
            "eval_count": body.get("eval_count"),
            "respuesta_cruda": body.get("response", ""),
        })

    return rows


def summarize(df: pd.DataFrame) -> dict:

    out = {}

    for mode, g in df.groupby("modo"):
        valid = g[g["valido"]]

        agree = (valid["qwen_relevante"] == (valid["humano_relevancia"] == "si")).tolist()

        lat = g["latencia_s"].tolist()

        qwen_no_relevante = ~valid["qwen_relevante"].astype(bool)

        out[mode] = {
            "n": len(g),
            "invalidos": int((~g["valido"]).sum()),
            "acuerdo_con_humano": round(sum(agree) / len(agree), 3) if agree else None,
            "acuerdo_ci95": bootstrap_ci([float(a) for a in agree]),
            "qwen_no_relevante": valid.loc[qwen_no_relevante, "id"].tolist(),
            "latencia": {
                "p50_s": round(percentile(lat, 50), 2), 
                "p95_s": round(percentile(lat, 95), 2),
                "max_s": round(max(lat), 2), 
                "total_s": round(sum(lat), 1)
            }
        }

    return out

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=False)

    answers = load_answers(ANSWERS)
    labels = pd.read_csv(LABELS_ANSWERS, sep=";")[["id", "humano_relevancia", "humano_abstencion"]]

    rows = [
        row 
            for mode in MODES 
                for row in judge(mode, answers)
    ]

    df = pd.DataFrame(rows).merge(labels, on="id", how="left")

    df.to_csv(OUT_DIR / "verdicts.csv", sep=";", index=False)

    summary = summarize(df)

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
