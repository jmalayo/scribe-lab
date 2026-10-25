from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from shared.eval.grounding import GroundingResult
from shared.llm import LLMResult
from shared.pipeline import ABSTENTION_ANSWER, Pipeline, PipelineConfig
from shared.prompts import chunk_block

REPO_ROOT = Path(__file__).resolve().parents[2]


def make_chunk(i: int, score: float) -> dict:

    return {
        "chunk_id": f"c{i}",
        "doc_id": "music-tagger/doc.md",
        "text": f"texto {i}",
        "chunk_index": i,
        "tables": [],
        "score": score,
    }


class FakeClient:

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.calls = []

    def query_points(self, collection_name, query, limit):
        self.calls.append((collection_name, limit))
        points = [
            SimpleNamespace(payload={k: v for k, v in c.items() if k != "score"}, score=c["score"])
            for c in self.chunks[:limit]
        ]
        return SimpleNamespace(points=points)


class FakeEmbedder:

    def encode(self, text, normalize_embeddings=False):
        return np.array([0.1, 0.2, 0.3], dtype=np.float32)


class FakeLLM:

    provider = "fake"
    model = "fake-model"

    def __init__(self, text="respuesta"):
        self.text = text
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return LLMResult(text=self.text, tokens_in=11, tokens_out=3)

    def ping(self):
        return True


def make_config(**overrides) -> PipelineConfig:

    base = {
        "pipeline_version": "test",
        "collection": "corpus_demo",
        "corpus_dir": "corpus_demo",
        "chunk_size": 284,
        "chunk_overlap": 71,
        "top_k": 3,
    }
    base.update(overrides)

    return PipelineConfig.model_validate(base)


def make_pipeline(config=None, llm=None, grounder=None, chunks=None):

    chunks = chunks if chunks is not None else [make_chunk(i, 0.9 - i * 0.1) for i in range(5)]

    return Pipeline(config or make_config(), FakeClient(chunks), FakeEmbedder(), llm or FakeLLM(), grounder)


def test_run_dense_builds_prompt_and_reports_stages():

    llm = FakeLLM()
    pipeline = make_pipeline(llm=llm)

    result = pipeline.run("¿pregunta?")

    assert result.answer == "respuesta"
    assert not result.abstained
    assert [c["chunk_id"] for c in result.chunks] == ["c0", "c1", "c2"]
    assert "[music-tagger/doc.md] texto 0" in llm.prompts[0]
    assert "Pregunta: ¿pregunta?" in llm.prompts[0]
    assert set(result.latencies_ms) == {"embedding", "retrieval", "llm", "total"}
    assert (result.tokens_in, result.tokens_out) == (11, 3)
    assert (result.provider, result.model) == ("fake", "fake-model")


def test_retrieve_respects_top_k_override():

    pipeline = make_pipeline()

    assert len(pipeline.retrieve("q", top_k=5).chunks) == 5
    assert pipeline.client.calls[-1] == ("corpus_demo", 5)


def test_abstention_skips_llm_when_best_score_is_below_threshold():

    llm = FakeLLM()
    config = make_config(abstention={"enabled": True, "min_score": 0.95})
    pipeline = make_pipeline(config=config, llm=llm)

    result = pipeline.run("q")

    assert result.abstained
    assert result.answer == ABSTENTION_ANSWER
    assert llm.prompts == []
    assert "llm" not in result.latencies_ms


def test_abstention_answers_when_best_score_reaches_threshold():

    config = make_config(abstention={"enabled": True, "min_score": 0.9})

    assert not make_pipeline(config=config).run("q").abstained


def test_abstention_without_threshold_is_rejected():

    with pytest.raises(ValueError, match="min_score"):
        make_pipeline(config=make_config(abstention={"enabled": True}))


def test_grounding_receives_the_same_chunk_blocks_as_f1():

    seen = {}

    def grounder(question, answer, chunks):
        seen.update(question=question, answer=answer, chunks=chunks)
        return GroundingResult(score=0.5, unsupported_sentences=[])

    pipeline = make_pipeline(config=make_config(grounding={"enabled": True}), grounder=grounder)
    result = pipeline.run("q")

    assert result.grounding == GroundingResult(score=0.5, unsupported_sentences=[])
    assert seen["chunks"] == [chunk_block(c) for c in result.chunks]
    assert "grounding" in result.latencies_ms


def test_grounding_enabled_without_grounder_is_rejected():

    with pytest.raises(ValueError, match="grounder"):
        make_pipeline(config=make_config(grounding={"enabled": True}))


def test_unknown_prompt_version_is_rejected():

    with pytest.raises(ValueError, match="prompt_version"):
        make_pipeline(config=make_config(prompt_version="v99"))


def test_score_key_follows_method_and_reranker():

    assert make_pipeline().score_key == "score"
    assert make_pipeline(config=make_config(method="hybrid_rrf")).score_key == "rrf_score"
    assert make_pipeline(config=make_config(reranker=True)).score_key == "rerank_score"


def test_repo_pipeline_yaml_matches_the_adopted_configuration():

    config = PipelineConfig.from_yaml(REPO_ROOT / "app" / "config" / "pipeline.yaml")

    assert (config.chunk_size, config.chunk_overlap) == (293, 73)
    assert config.method == "hybrid_rrf"
    assert config.reranker is False
    assert config.top_k == 10
    assert config.abstention.enabled is False
    assert config.grounding.enabled is False
