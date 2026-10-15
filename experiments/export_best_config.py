import subprocess
import sys
from datetime import date
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluation.run import USE_RERANKER  # noqa: E402
from experiments.reranking.run import (  # noqa: E402
    BEST_CHUNKING,
    BEST_METHOD,
    POOL_SIZE,
    TOP_K,
)
from shared.pipeline import PipelineConfig  # noqa: E402

OUT_PATH = REPO_ROOT / "app" / "config" / "pipeline.yaml"
COLLECTION_ALIAS = "corpus_demo"
CORPUS_DIR = "corpus_demo"


def git_short_sha() -> str:

    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO_ROOT
    ).stdout.strip()


def main():

    current = yaml.safe_load(OUT_PATH.read_text(encoding="utf-8")) if OUT_PATH.exists() else {}

    config = {
        **current,
        "pipeline_version": f"{date.today().isoformat()}-{git_short_sha()}",
        "collection": current.get("collection", COLLECTION_ALIAS),
        "corpus_dir": current.get("corpus_dir", CORPUS_DIR),
        "chunk_size": BEST_CHUNKING["chunk_size"],
        "chunk_overlap": BEST_CHUNKING["chunk_overlap"],
        "method": BEST_METHOD,
        "reranker": USE_RERANKER,
        "pool_size": POOL_SIZE,
        "top_k": TOP_K,
    }

    PipelineConfig.model_validate(config)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print(f"{OUT_PATH.relative_to(REPO_ROOT)}: {config}")


if __name__ == "__main__":
    main()
