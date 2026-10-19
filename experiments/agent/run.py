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
from shared.eval.metrics import (
    bootstrap_ci,
    cohen_kappa,
    is_chunk_correct,
    latency_summary,
    mcnemar_exact,
    wilson_ci,
)
from shared.llm import build_llm
from shared.pipeline import Pipeline, PipelineConfig, RetrievalResult
from shared.settings import settings
from shared.tracking import log_metrics, tracked_run

EXP = "exp_2"
MODEL = "llama3.1:8b-instruct-q4_K_M"
REUSE_ANSWERS = True
SUMMARIZE_MANUAL = False
RESULTS_DIR = Path(__file__).resolve().parent / "results" / EXP

BASE_QUESTIONS = Path(settings.questions_path) / "questions_base.jsonl"
TIPOS = ("simple", "comparativa_mismo_doc", "comparativa_entre_docs", "sin_respuesta")
COMPARATIVAS = ("comparativa_mismo_doc", "comparativa_entre_docs")
SIDES = ("pipeline", "agent")
TOP_K = 10
ALPHA = 0.05
FALSE_ABSTENTION_MAX = 0.20

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


def spans_found(chunks: list[dict], question: dict) -> int:

    return sum(
        any(is_chunk_correct(c, {**question, "gold_spans": [span]}) for c in chunks)
            for span in question["gold_spans"]
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
        chunks = result.chunks[:TOP_K]

        rows.append({
            "id": q["id"],
            "question": q["question"],
            "answer": result.answer,
            "tokens": (result.tokens_in or 0) + (result.tokens_out or 0),
            "latency_ms": result.latencies_ms["total"],
            "retrieval_ms": result.latencies_ms["retrieval"],
            "llm_ms": result.latencies_ms["llm"],
            "n_gold_spans": len(q["gold_spans"]),
            "gold_spans_at_10": spans_found(chunks, q),
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


def question_sets() -> tuple[list[dict], dict[str, set[str]]]:

    questions = load_questions()

    base_ids = {
        json.loads(line)["id"]
            for line in BASE_QUESTIONS.read_text(encoding="utf-8").splitlines()
                if line.strip()
    }

    groups = {
        tipo: {q["id"] for q in questions if q["tipo"] == tipo}
            for tipo in TIPOS
    }

    comparativas = groups["comparativa_mismo_doc"] | groups["comparativa_entre_docs"]

    groups |= {
        "comparativas_nuevas": comparativas - base_ids,
        "comparativas_base": comparativas & base_ids,
        "con_respuesta": {q["id"] for q in questions} - groups["sin_respuesta"],
    }

    return questions, groups


def rate(verdicts: pd.DataFrame, ids: set[str], labels: tuple[str, ...]) -> tuple[int, int]:

    subset = verdicts[verdicts["id"].isin(ids)]

    return int(subset["veredicto"].isin(labels).sum()), len(subset)


def summarize(verdicts: pd.DataFrame, answers: pd.DataFrame | None, groups: dict[str, set[str]]) -> dict:

    answerable = verdicts[verdicts["id"].isin(groups["con_respuesta"])]
    correct = (answerable["veredicto"] == "correcta").astype(float).tolist()
    low, high = bootstrap_ci(correct)

    false_abst, n_simple = rate(verdicts, groups["simple"], ("abstencion",))
    right_abst, n_unanswerable = rate(verdicts, groups["sin_respuesta"], ("abstencion",))

    metrics = {
        "n_answerable": len(answerable),
        "correct_rate": round(sum(correct) / len(correct), 3),
        "correct_rate_ci_low": low,
        "correct_rate_ci_high": high,
        "correct_or_partial_rate": round(answerable["veredicto"].isin(["correcta", "parcial"]).mean(), 3),
        "incorrect_rate": round((answerable["veredicto"] == "incorrecta").mean(), 3),
        "false_abstentions": int((answerable["veredicto"] == "abstencion").sum()),
        "false_abstentions_simple": false_abst,
        "false_abstention_rate_simple": round(false_abst / n_simple, 3),
        "false_abstention_ci_high_simple": wilson_ci(false_abst, n_simple)[1],
        "correct_abstentions": right_abst,
        "correct_abstention_rate": round(right_abst / n_unanswerable, 3),
        "correct_abstention_ci_low": wilson_ci(right_abst, n_unanswerable)[0],
        "judge_invalid": int((~verdicts["valido"]).sum()),
    }

    for group in (*TIPOS[:3], "comparativas_nuevas", "comparativas_base"):
        hits, n = rate(verdicts, groups[group], ("correcta",))
        metrics |= {f"correct_rate_{group}": round(hits / n, 3), f"n_{group}": n}

    if answers is not None:
        metrics |= {
            **latency_summary(answers["latency_ms"].astype(float).tolist()),
            "tokens_mean": round(answers["tokens"].astype(float).mean(), 1),
        }

    if answers is not None and "gold_spans_at_10" in answers:
        retrieved = answers.set_index("id")

        for tipo in TIPOS[:3]:
            subset = retrieved.loc[sorted(groups[tipo])]
            found = subset["gold_spans_at_10"].astype(int)
            metrics |= {
                f"hit_at_10_{tipo}": round((found > 0).mean(), 3),
                f"all_spans_at_10_{tipo}": round((found == subset["n_gold_spans"].astype(int)).mean(), 3),
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


def paired(verdicts: pd.DataFrame, side_a: str, side_b: str, ids: set[str], hits: tuple[str, ...]) -> dict:

    wide = verdicts.pivot(index="id", columns="lado", values="veredicto").loc[sorted(ids)]
    a = wide[side_a].isin(hits)
    b = wide[side_b].isin(hits)

    only_a = int((a & ~b).sum())
    only_b = int((b & ~a).sum())

    return {
        "n": len(wide),
        f"only_{side_a}": only_a,
        f"only_{side_b}": only_b,
        "mcnemar_p": mcnemar_exact(only_a, only_b),
    }


def decision(verdicts: pd.DataFrame, summary: dict, groups: dict[str, set[str]]) -> dict:

    new, base = groups["comparativas_nuevas"], groups["comparativas_base"]
    p1 = paired(verdicts, "pipeline", "agent", new, ("correcta",))

    return {
        "p1_comparativas_nuevas": p1,
        "p1_comparativas_nuevas_con_parcial": paired(verdicts, "pipeline", "agent", new, ("correcta", "parcial")),
        "p1_comparativas_base": paired(verdicts, "pipeline", "agent", base, ("correcta",)),
        "p1_pasa": p1["mcnemar_p"] < ALPHA and p1["only_agent"] > p1["only_pipeline"],
        "p2_pasa": summary["agent"]["false_abstention_ci_high_simple"] < FALSE_ABSTENTION_MAX,
        "p3_pasa": summary["agent"]["correct_abstentions"] >= summary["pipeline"]["correct_abstentions"],
    }


def manual_summary(out_dir: Path = RESULTS_DIR) -> dict:

    _, groups = question_sets()

    manual = load_answers(out_dir / "verdicts_manual.csv")
    human = manual.assign(veredicto=manual["veredicto_manual"], valido=True)

    summary = {
        side: summarize(human[human["lado"] == side], None, groups)
            for side in SIDES
    }

    summary["decision"] = decision(human, summary, groups)

    agree = (manual["veredicto"] == manual["veredicto_manual"]).astype(float).tolist()

    summary["judge"] = {
        "n": len(agree),
        "agreement": round(sum(agree) / len(agree), 3),
        "agreement_ci": bootstrap_ci(agree),
        "kappa": cohen_kappa(manual["veredicto_manual"].tolist(), manual["veredicto"].tolist()),
        "confusion_human_x_judge": pd.crosstab(manual["veredicto_manual"], manual["veredicto"]).to_dict(orient="index"),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    return summary


def main(out_dir: Path = RESULTS_DIR) -> dict:

    out_dir.mkdir(parents=True, exist_ok=True)

    questions, groups = question_sets()
    expected = {q["id"]: q["expected_answer"] for q in questions}

    sides = {
        side: answers_for(side, questions, out_dir)
            for side in SIDES
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
        metrics = summarize(verdicts[verdicts["lado"] == side], answers, groups)

        with tracked_run(
            "agent",
            f"{EXP}-{side}",
            {
                "exp": EXP,
                "model": MODEL,
                "judge": settings.llm_model_judge,
                "retrieval": f"/v1/retrieve dense top-{TOP_K}",
                "questions": len(questions),
            },
        ):
            log_metrics(metrics)

        summary[side] = metrics

    summary["decision"] = decision(verdicts, summary, groups)

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    return summary


if __name__ == "__main__":
    manual_summary() if SUMMARIZE_MANUAL else main()
