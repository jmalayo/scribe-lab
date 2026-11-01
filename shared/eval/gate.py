import csv
import os
import sys
from pathlib import Path

from shared.eval.data import load_questions
from shared.eval.metrics import calculate_hit_at_k, mcnemar_exact
from shared.ingest import build_qdrant_client, get_embedder
from shared.llm import NoLLM
from shared.pipeline import Pipeline, PipelineConfig
from shared.settings import settings

MODE = os.environ.get("GATE_MODE", "check")
BASELINE_PATH = Path(os.environ.get("GATE_BASELINE", "shared/eval/gate_baseline.csv"))
OUTPUT_PATH = Path(os.environ.get("GATE_OUTPUT", "gate_hits.csv"))
K = 10
ALPHA = 0.05


def current_hits(pipeline: Pipeline, questions: list[dict]) -> dict[str, float]:

    results = {
        q["id"]: pipeline.retrieve(q["question"], top_k=K).chunks
            for q in questions
    }

    return dict(zip([q["id"] for q in questions], calculate_hit_at_k(results, questions, K), strict=True))


def read_hits(path: Path) -> dict[str, float]:

    with path.open(encoding="utf-8") as f:
        return {row["question_id"]: float(row["hit_at_10"]) for row in csv.DictReader(f)}


def write_hits(path: Path, hits: dict[str, float]) -> None:

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question_id", "hit_at_10"])
        writer.writerows((qid, int(hit)) for qid, hit in sorted(hits.items()))


def compare(baseline: dict[str, float], current: dict[str, float]) -> dict:

    shared_ids = sorted(baseline.keys() & current.keys())
    lost = [qid for qid in shared_ids if baseline[qid] and not current[qid]]
    gained = [qid for qid in shared_ids if current[qid] and not baseline[qid]]
    p_value = mcnemar_exact(len(gained), len(lost))

    return {
        "questions": len(shared_ids),
        "baseline_recall": sum(baseline[q] for q in shared_ids) / len(shared_ids),
        "current_recall": sum(current[q] for q in shared_ids) / len(shared_ids),
        "lost": lost,
        "gained": gained,
        "p_value": p_value,
        "blocked": len(lost) > len(gained) and p_value < ALPHA,
    }


def summary(result: dict) -> str:

    verdict = "BLOQUEA" if result["blocked"] else "PASA"

    return "\n".join(
        [
            f"## Gate de evaluación: {verdict}",
            "",
            "| Métrica | Línea base | Este cambio |",
            "| --- | --- | --- |",
            f"| recall@10 ({result['questions']} preguntas) | {result['baseline_recall']:.3f} | {result['current_recall']:.3f} |",
            "",
            f"Preguntas ganadas: {len(result['gained'])}. Preguntas perdidas: {len(result['lost'])}.",
            f"McNemar exacta: p = {result['p_value']}. Bloquea si se pierden más de las que se ganan con p < {ALPHA}.",
            "",
            f"Perdidas: {', '.join(result['lost']) or 'ninguna'}",
            f"Ganadas: {', '.join(result['gained']) or 'ninguna'}",
        ]
    )


def main() -> int:

    config = PipelineConfig.from_yaml(settings.pipeline_config_path)
    pipeline = Pipeline(config, build_qdrant_client(), get_embedder(), NoLLM(), None, NoLLM())
    hits = current_hits(pipeline, load_questions())
    write_hits(OUTPUT_PATH, hits)

    if MODE == "baseline" or not BASELINE_PATH.exists():
        text = (
            f"## Gate de evaluación: línea base generada\n\n"
            f"recall@10 = {sum(hits.values()) / len(hits):.3f} sobre {len(hits)} preguntas. "
            f"Guardar `{OUTPUT_PATH}` como `{BASELINE_PATH}`."
        )
        blocked = False

    else:
        result = compare(read_hits(BASELINE_PATH), hits)
        text = summary(result)
        blocked = result["blocked"]

    print(text)

    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(text + "\n")

    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(main())
