import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from agent.graph import build_graph, run_agent
from agent.graph import build_llm as build_agent_llm
from agent.tools import agent_tools
from experiments.evaluation.prompts import prompt_v3
from shared.eval.data import load_answers, load_questions
from shared.eval.metrics import bootstrap_ci, latency_summary
from shared.llm import build_llm
from shared.pipeline import Pipeline, PipelineConfig, RetrievalResult
from shared.settings import settings
from shared.tracking import log_metrics, tracked_run

EXP = "exp_1"
MODEL = "llama3.1:8b-instruct-q4_K_M"
REUSE_ANSWERS = True
RESULTS_DIR = Path(__file__).resolve().parent / "results" / EXP

EXP7_ANSWERS = REPO_ROOT / "experiments/evaluation/benchmark-models/results/exp_7/llama31_base/base_answers.csv"
LABELS_ANSWERS = REPO_ROOT / "experiments/evaluation/generative/llama_3_1_8_B/labels_answers.csv"

CORRECTNESS_PROMPT, CORRECTNESS_SCHEMA = prompt_v3()

JUDGE_OPTIONS = {
    "temperature": 0.0,
    "num_ctx": 4096,
    "num_predict": 1000
}


class ApiRetrievalPipeline(Pipeline):

    def retrieve(self, question: str, top_k: int | None = None) -> RetrievalResult:

        t0 = time.perf_counter()

        resp = requests.post(
            f"{settings.scribe_api_url}/v1/retrieve",
            json={"query": question, "top_k": top_k or self.config.top_k},
            timeout=30,
        )
        resp.raise_for_status()

        chunks = [
            {**c, "tables": [{"text_content": t} for t in c["tables"]]}
            for c in resp.json()["chunks"]
        ]

        return RetrievalResult(
            chunks=chunks,
            latencies_ms={"retrieval": round((time.perf_counter() - t0) * 1000, 2)},
        )


def run_pipeline(questions: list[dict]) -> pd.DataFrame:

    config = PipelineConfig.from_yaml(settings.pipeline_config_path)
    config.llm.model = MODEL

    pipeline = ApiRetrievalPipeline(
        config,
        client=None,
        embedder=None,
        llm=build_llm(config.llm.provider, MODEL, config.llm.max_tokens),
    )

    rows = []

    for q in questions:
        result = pipeline.run(q["question"])

        rows.append({
            "id": q["id"],
            "question": q["question"],
            "answer": result.answer,
            "tokens": (result.tokens_in or 0) + (result.tokens_out or 0),
            "latency_ms": result.latencies_ms["total"],
            "retrieval_ms": result.latencies_ms["retrieval"],
            "llm_ms": result.latencies_ms["llm"],
        })

    return pd.DataFrame(rows)


async def run_agent_side(questions: list[dict]) -> pd.DataFrame:

    rows = []

    async with agent_tools() as tools:
        graph = build_graph(build_agent_llm(MODEL), tools)

        for q in questions:
            result = await run_agent(graph, q["question"])

            rows.append({
                "id": q["id"],
                "question": q["question"],
                "answer": result.answer or "",
                "outcome": result.outcome,
                "reason": result.reason or "",
                "steps": result.steps,
                "tokens": result.tokens,
                "latency_ms": result.latency_ms,
                "tool_ms": round(sum(c["latency_ms"] for c in result.tool_calls), 1),
                "n_tool_calls": len(result.tool_calls),
                "tool_calls": json.dumps(result.tool_calls, ensure_ascii=False),
            })

    return pd.DataFrame(rows)


def answers_for(side: str, questions: list[dict], out_dir: Path) -> pd.DataFrame:

    path = out_dir / f"answers_{side}.csv"

    if path.exists():
        if not REUSE_ANSWERS:
            raise FileExistsError(f"{path} ya existe; cambiar EXP o activar REUSE_ANSWERS")

        return load_answers(path)

    df = run_pipeline(questions) if side == "pipeline" else asyncio.run(run_agent_side(questions))
    df.to_csv(path, sep=";", index=False)

    return df


def parse(text: str) -> dict | None:

    match = re.search(r"\{.*\}", text or "", re.DOTALL)

    if match is None:
        return None

    try:
        data = json.loads(match.group(0))

    except json.JSONDecodeError:
        return None

    if data.get("veredicto") not in CORRECTNESS_SCHEMA["properties"]["veredicto"]["enum"]:
        return None

    return data


def judge(side: str, answers: pd.DataFrame, expected: dict[str, str]) -> list[dict]:

    rows = []

    for r in answers.itertuples():

        prompt = CORRECTNESS_PROMPT.format(
            question=r.question,
            expected=expected[r.id],
            answer=r.answer or "(vacía)",
        )

        t0 = time.perf_counter()

        resp = requests.post(
            f"{settings.llm_host}/api/generate",
            timeout=600,
            json={
                "model": settings.llm_model_judge,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "format": CORRECTNESS_SCHEMA,
                "options": JUDGE_OPTIONS,
            },
        )
        resp.raise_for_status()

        verdict = parse(resp.json().get("response", ""))

        rows.append({
            "lado": side,
            "id": r.id,
            "valido": verdict is not None,
            "veredicto": verdict["veredicto"] if verdict else None,
            "motivo": verdict.get("motivo", "") if verdict else "",
            "latencia_s": round(time.perf_counter() - t0, 2),
        })

    return rows


def summarize(verdicts: pd.DataFrame, answers: pd.DataFrame | None, unanswerable: set[str], multi_span: set[str]) -> dict:

    answerable = verdicts[~verdicts["id"].isin(unanswerable)]
    without = verdicts[verdicts["id"].isin(unanswerable)]
    multi = answerable[answerable["id"].isin(multi_span)]

    correct = (answerable["veredicto"] == "correcta").astype(float).tolist()
    low, high = bootstrap_ci(correct)

    metrics = {
        "n_answerable": len(answerable),
        "correct_rate": round(sum(correct) / len(correct), 3),
        "correct_rate_ci_low": low,
        "correct_rate_ci_high": high,
        "correct_or_partial_rate": round(answerable["veredicto"].isin(["correcta", "parcial"]).mean(), 3),
        "incorrect_rate": round((answerable["veredicto"] == "incorrecta").mean(), 3),
        "false_abstentions": int((answerable["veredicto"] == "abstencion").sum()),
        "correct_abstentions": int((without["veredicto"] == "abstencion").sum()),
        "n_unanswerable": len(without),
        "multi_span_correct_rate": round((multi["veredicto"] == "correcta").mean(), 3),
        "n_multi_span": len(multi),
        "judge_invalid": int((~verdicts["valido"]).sum()),
    }

    if answers is not None:
        metrics |= {
            **latency_summary(answers["latency_ms"].astype(float).tolist()),
            "tokens_mean": round(answers["tokens"].astype(float).mean(), 1),
        }

    if answers is not None and "outcome" in answers:
        metrics |= {
            "steps_mean": round(answers["steps"].astype(float).mean(), 2),
            "tool_calls_mean": round(answers["n_tool_calls"].astype(float).mean(), 2),
            "tool_ms_p50": round(answers["tool_ms"].astype(float).median(), 1),
            "tool_abstentions": int((answers["outcome"] == "abstencion").sum()),
            "limit_stops": int((answers["outcome"] == "limite").sum()),
        }

    return metrics


def main(out_dir: Path = RESULTS_DIR) -> dict:

    out_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions()
    expected = {q["id"]: q["expected_answer"] for q in questions}
    multi_span = {q["id"] for q in questions if len(q.get("gold_spans") or []) > 1}

    labels = pd.read_csv(LABELS_ANSWERS, sep=";", keep_default_na=False)
    unanswerable = set(labels.loc[labels["contexto_suficiente"] == "n.a.", "id"])

    sides = {
        "pipeline": answers_for("pipeline", questions, out_dir),
        "agent": answers_for("agent", questions, out_dir),
        "exp7": load_answers(EXP7_ANSWERS),
    }

    verdicts_path = out_dir / "verdicts.csv"

    if verdicts_path.exists():
        verdicts = load_answers(verdicts_path).assign(valido=lambda d: d["valido"] == "True")
    else:
        verdicts = pd.DataFrame([
            row
            for side, answers in sides.items()
            for row in judge(side, answers, expected)
        ])
        verdicts.to_csv(verdicts_path, sep=";", index=False)

    summary = {}

    for side, answers in sides.items():
        metrics = summarize(
            verdicts[verdicts["lado"] == side],
            answers if side != "exp7" else None,
            unanswerable,
            multi_span,
        )

        with tracked_run(
            "agent",
            side,
            {
                "exp": EXP,
                "model": MODEL,
                "judge": settings.llm_model_judge,
                "retrieval": "exp_7 (hybrid_rrf + ms-marco)" if side == "exp7" else "/v1/retrieve dense top-10",
            },
        ):
            log_metrics(metrics)

        summary[side] = metrics

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    return summary


if __name__ == "__main__":
    main()
