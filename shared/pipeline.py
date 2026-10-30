from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import requests
from pydantic import BaseModel

from shared.eval.grounding import GroundingResult, should_abstain
from shared.llm import LLMClient
from shared.memory import build_messages, parse_rewrite
from shared.prompts import (
    ABSTENTION_ANSWER,
    ANSWER_PROMPTS,
    REWRITE_PROMPTS,
    build_context_text,
    chunk_block,
)
from shared.retrieval import (
    BM25Index,
    dense_search_vector,
    fetch_all_chunks,
    reciprocal_rank_fusion,
)

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

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

class MemoryConfig(BaseModel):
    enabled: bool = False
    model: str = "llama3.2:3b"
    max_tokens: int = 120
    prompt_version: str = "v2"
    rrf_k: int = 60

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
    memory: MemoryConfig = MemoryConfig()

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
    rewritten_question: str | None = None

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
        rewriter: LLMClient | None = None,
    ):
        if config.prompt_version not in ANSWER_PROMPTS:
            raise ValueError(f"prompt_version desconocida: {config.prompt_version}")

        if config.abstention.enabled and config.abstention.min_score is None:
            raise ValueError("abstention.enabled requiere abstention.min_score")

        if config.grounding.enabled and grounder is None:
            raise ValueError("grounding.enabled requiere un grounder")

        if config.memory.enabled and rewriter is None:
            raise ValueError("memory.enabled requiere un rewriter")

        if config.memory.prompt_version not in REWRITE_PROMPTS:
            raise ValueError(f"memory.prompt_version desconocida: {config.memory.prompt_version}")

        self.config = config
        self.client = client
        self.embedder = embedder
        self.llm = llm
        self.grounder = grounder
        self.rewriter = rewriter
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
            dense = [
                {**c, "dense_score": c["score"]}
                    for c in dense_search_vector(self.client, self.config.collection, vector, pool)
            ]

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

    def rewrite(self, question: str, history: Sequence[tuple[str, str]]) -> str:

        messages = build_messages(history, question, self.config.memory.prompt_version)

        try:
            raw = self.rewriter.chat(messages, json_format=True).text
        except requests.RequestException:
            return question

        return parse_rewrite(raw, history, question)["pregunta"]

    def retrieve_fused(self, question: str, rewritten: str, top_k: int | None = None) -> RetrievalResult:

        literal = self.retrieve(question, top_k)
        second = self.retrieve(rewritten, top_k)

        latencies = dict(literal.latencies_ms)

        for stage, ms in second.latencies_ms.items():
            latencies[stage] = round(latencies.get(stage, 0.0) + ms, 2)

        chunks = reciprocal_rank_fusion(
            rankings=[literal.chunks, second.chunks],
            k_constant=self.config.memory.rrf_k,
            top_k=top_k or self.config.top_k,
        )

        return RetrievalResult(chunks=chunks, latencies_ms=latencies)

    def retrieve_with_history(
        self,
        question: str,
        history: Sequence[tuple[str, str]],
        top_k: int | None = None,
    ) -> tuple[RetrievalResult, str | None]:

        if not (self.config.memory.enabled and history):
            return self.retrieve(question, top_k), None

        t0 = time.perf_counter()
        rewritten = self.rewrite(question, history)
        rewrite_ms = _get_time_ms(t0)

        if rewritten == question:
            retrieved = self.retrieve(question, top_k)
        else:
            retrieved = self.retrieve_fused(question, rewritten, top_k)

        latencies = {"rewrite": rewrite_ms, **retrieved.latencies_ms}

        return RetrievalResult(chunks=retrieved.chunks, latencies_ms=latencies), rewritten

    def run(
        self,
        question: str,
        top_k: int | None = None,
        history: Sequence[tuple[str, str]] = (),
    ) -> PipelineResult:

        t_total = time.perf_counter()

        retrieved, rewritten = self.retrieve_with_history(question, history, top_k)
        question_to_answer = rewritten or question
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
                rewritten_question=rewritten,
            )

        prompt = ANSWER_PROMPTS[self.config.prompt_version].format(
            context=build_context_text(chunks),
            question=question_to_answer,
        )

        t0 = time.perf_counter()

        completion = self.llm.complete(prompt)
        latencies["llm"] = _get_time_ms(t0)

        grounding = None
        
        if self.config.grounding.enabled:
            t0 = time.perf_counter()

            grounding = self.grounder(question_to_answer, completion.text, [chunk_block(c) for c in chunks])
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
            rewritten_question=rewritten,
        )
