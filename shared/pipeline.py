from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel

from shared.eval.grounding import GroundingResult, should_abstain
from shared.llm import LLMClient
from shared.prompts import ANSWER_PROMPTS, build_context_text, chunk_block
from shared.retrieval import (
    BM25Index,
    dense_search_vector,
    fetch_all_chunks,
    reciprocal_rank_fusion,
)

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

ABSTENTION_ANSWER = "No lo sé según el contexto proporcionado."

class LLMConfig(BaseModel):
    provider: Literal["ollama"] = "ollama"
    model: str = "llama3.2:3b"
    max_tokens: int = 500
    base_url: str | None = None

class AbstentionConfig(BaseModel):
    enabled: bool = False
    min_score: float | None = None

class GroundingConfig(BaseModel):
    enabled: bool = False
    threshold: float = 0.5
    entailment_threshold: float = 0.5
    contradiction_threshold: float = 0.5

class PipelineConfig(BaseModel):
    pipeline_version: str
    collection: str
    corpus_dir: str
    chunk_size: int
    chunk_overlap: int
    method: Literal["dense", "bm25", "hybrid_rrf"] = "dense"
    reranker: bool = False
    pool_size: int = 20
    top_k: int = 10
    prompt_version: str = "v1"
    llm: LLMConfig = LLMConfig()
    abstention: AbstentionConfig = AbstentionConfig()
    grounding: GroundingConfig = GroundingConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:

        import yaml

        return cls.model_validate(
            yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        )

@dataclass
class PipelineResult:
    answer: str
    chunks: list[dict]
    abstained: bool
    grounding: GroundingResult | None
    latencies_ms: dict[str, float]
    tokens_in: int | None = None
    tokens_out: int | None = None
    provider: str | None = None
    model: str | None = None

@dataclass
class RetrievalResult:
    chunks: list[dict]
    latencies_ms: dict[str, float] = field(default_factory=dict)

class Grounder(Protocol):

    def __call__(self, question: str, answer: str, chunks: list[str]) -> GroundingResult: ...

def http_grounder(detector_url: str, nli_url: str, config: GroundingConfig) -> Grounder:

    from shared.eval.grounding import (
        detect_tokens,
        ground_answer,
        nli_http_predictor,
        nli_scores,
        split_sentences,
    )

    predict = nli_http_predictor(nli_url)

    def ground(question: str, answer: str, chunks: list[str]) -> GroundingResult:

        tokens = detect_tokens(detector_url, chunks, question, answer)
        nli = nli_scores(split_sentences(answer), chunks, predict)

        result = ground_answer(
            answer=answer,
            chunks=chunks,
            question=question,
            tokens=tokens,
            nli=nli,
            threshold=config.threshold,
            entailment_threshold=config.entailment_threshold,
            contradiction_threshold=config.contradiction_threshold,
        )

        return GroundingResult.model_validate(result)

    return ground


def _get_time_ms(t0: float) -> float:

    return round((time.perf_counter() - t0) * 1000, 2)

class Pipeline:

    def __init__(
        self,
        config: PipelineConfig,
        client: QdrantClient,
        embedder,
        llm: LLMClient,
        grounder: Grounder | None = None,
    ):
        if config.prompt_version not in ANSWER_PROMPTS:
            raise ValueError(f"prompt_version desconocida: {config.prompt_version}")

        if config.abstention.enabled and config.abstention.min_score is None:
            raise ValueError("abstention.enabled requiere abstention.min_score")

        if config.grounding.enabled and grounder is None:
            raise ValueError("grounding.enabled requiere un grounder")

        self.config = config
        self.client = client
        self.embedder = embedder
        self.llm = llm
        self.grounder = grounder
        self._bm25: BM25Index | None = None

    @property
    def score_key(self) -> str:

        if self.config.reranker:
            return "rerank_score"

        return "rrf_score" if self.config.method == "hybrid_rrf" else "score"

    def _bm25_index(self) -> BM25Index:

        if self._bm25 is None:
            self._bm25 = BM25Index(fetch_all_chunks(self.client, self.config.collection))

        return self._bm25

    def retrieve(self, question: str, top_k: int | None = None) -> RetrievalResult:

        k = top_k or self.config.top_k
        pool = self.config.pool_size if self.config.reranker else k
        latencies: dict[str, float] = {}
        candidates: list[dict] = []

        if self.config.method in ("dense", "hybrid_rrf"):

            t0 = time.perf_counter()

            vector = self.embedder.encode(question, normalize_embeddings=True).tolist()
            latencies["embedding"] = _get_time_ms(t0)

            t0 = time.perf_counter()
            dense = dense_search_vector(self.client, self.config.collection, vector, pool)

            if self.config.method == "hybrid_rrf":
                candidates = reciprocal_rank_fusion(
                    rankings=[dense, self._bm25_index().search(question, pool)], 
                    top_k=pool
                )

            else:
                candidates = dense

            latencies["retrieval"] = _get_time_ms(t0)

        else:
            t0 = time.perf_counter()
            
            candidates = self._bm25_index().search(question, pool)
            latencies["retrieval"] = _get_time_ms(t0)

        if self.config.reranker:
            from shared.retrieval import rerank

            t0 = time.perf_counter()

            candidates = rerank(question, candidates, k)
            latencies["rerank"] = _get_time_ms(t0)

        return RetrievalResult(
            chunks=candidates[:k], 
            latencies_ms=latencies
        )

    def run(self, question: str, top_k: int | None = None) -> PipelineResult:

        t_total = time.perf_counter()

        retrieved = self.retrieve(question, top_k)
        latencies = dict(retrieved.latencies_ms)
        chunks = retrieved.chunks

        abstain = self.config.abstention.enabled and should_abstain(
            chunks=chunks, 
            min_score=self.config.abstention.min_score, 
            score_key=self.score_key
        )

        if abstain:
            latencies["total"] = _get_time_ms(t_total)

            return PipelineResult(
                answer=ABSTENTION_ANSWER,
                chunks=chunks,
                abstained=True,
                grounding=None,
                latencies_ms=latencies,
                provider=self.llm.provider,
                model=self.llm.model,
            )

        prompt = ANSWER_PROMPTS[self.config.prompt_version].format(
            context=build_context_text(chunks), 
            question=question
        )

        t0 = time.perf_counter()

        completion = self.llm.complete(prompt)
        latencies["llm"] = _get_time_ms(t0)

        grounding = None
        
        if self.config.grounding.enabled:
            t0 = time.perf_counter()

            grounding = self.grounder(question, completion.text, [chunk_block(c) for c in chunks])
            latencies["grounding"] = _get_time_ms(t0)

        latencies["total"] = _get_time_ms(t_total)

        return PipelineResult(
            answer=completion.text,
            chunks=chunks,
            abstained=False,
            grounding=grounding,
            latencies_ms=latencies,
            tokens_in=completion.tokens_in,
            tokens_out=completion.tokens_out,
            provider=self.llm.provider,
            model=self.llm.model,
        )
