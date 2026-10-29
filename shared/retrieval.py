from __future__ import annotations

import functools
import math
from collections import Counter
from typing import TYPE_CHECKING

from shared.settings import settings

# qdrant_client y shared.ingest (transformers) se cargan solo donde se usan: BM25 y RRF quedan
# importables con las dependencias mínimas de los tests unitarios
if TYPE_CHECKING:
    from qdrant_client import QdrantClient

def dense_search(client: QdrantClient, collection_name: str, query: str, k: int) -> list[dict]:

    from shared.ingest import get_embedder

    embedder = get_embedder()
    vector = embedder.encode(query, normalize_embeddings=True).tolist()

    return dense_search_vector(client, collection_name, vector, k)

def dense_search_vector(client: QdrantClient, collection_name: str, vector: list[float], k: int) -> list[dict]:

    hits = client.query_points(
        collection_name=collection_name, 
        query=vector, 
        limit=k
    ).points

    return [
        {
            **h.payload, 
            "score": h.score
        } for h in hits
    ]

def fetch_all_chunks(client: QdrantClient, collection_name: str) -> list[dict]:

    limit_batch = 200
    ls_records = []
    next_offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=None,
            limit=limit_batch,
            with_vectors=False,
            with_payload=True,
            offset=next_offset
        )

        ls_records.extend(records)

        if next_offset is None:
            
            return [
                rc.payload 
                    for rc in ls_records
            ]

class BM25Index:

    def __init__(
        self, 
        chunks: list[dict], 
        k1: float = 1.5, 
        b: float = 0.75
    ):

        self._chunks = chunks
        self._k1 = k1
        self._b = b

        doc_tokens = [
            self._tokenize(c.get("text", "")) 
                for c in chunks
        ]

        self._doc_len = [
            len(toks) 
                for toks in doc_tokens
        ]

        self._avgdl = sum(self._doc_len) / len(self._doc_len) if doc_tokens else 0.0

        self._term_freqs = [
            Counter(toks) 
                for toks in doc_tokens
        ]

        df: Counter = Counter()

        for toks in doc_tokens:
            df.update(set(toks))

        n = len(chunks)
        self._idf = {
            term: math.log(
                1 + (n - freq + 0.5) / (freq + 0.5)
            )
            for term, freq in df.items()
        }

    def _tokenize(self, text: str) -> list[str]:
        
        return text.lower().split()
        
    def search(self, query: str, k: int) -> list[dict]:

        if not self._chunks:
            return []

        query_terms = self._tokenize(query)
        scores = [0.0] * len(self._chunks)

        for i, (term_freqs, doc_len) in enumerate(zip(self._term_freqs, self._doc_len, strict=True)):

            score = 0.0
            for term in query_terms:

                freq = term_freqs.get(term)

                if not freq:
                    continue

                idf = self._idf.get(term, 0.0)

                norm = (
                    self._k1 * (1 - self._b + self._b * doc_len / self._avgdl) if self._avgdl else self._k1
                )

                score += idf * (freq * (self._k1 + 1)) / (freq + norm)

            scores[i] = score

        ranked = sorted(
            range(len(self._chunks)), 
            key=lambda i: scores[i], 
            reverse=True
        )[:k]

        return [
            {
                **self._chunks[i], 
                "score": scores[i]
            }
            for i in ranked
        ]

def reciprocal_rank_fusion(
    rankings: list[list[dict]], 
    k_constant: int = 60, 
    top_k: int | None = None
) -> list[dict]:

    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            cid = chunk["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_constant + rank + 1)
            payloads[cid] = chunk

    fused_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)

    if top_k:
        fused_ids = fused_ids[:top_k]

    return [
        {
            **payloads[cid], 
            "rrf_score": scores[cid]
        } for cid in fused_ids
    ]

class LocalReranker:

    def __init__(self, model_name: str):

        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def score(self, question: str, texts: list[str]) -> list[float]:

        pairs = [(question, t) for t in texts]

        return [float(s) for s in self._model.predict(pairs, batch_size=16, show_progress_bar=False)]


class RemoteReranker:

    def __init__(self, url: str, timeout: tuple[float, float] = (5, 30)):

        self._url = url.rstrip("/")
        self._timeout = timeout

    def score(self, question: str, texts: list[str]) -> list[float]:

        import requests

        resp = requests.post(
            f"{self._url}/rerank",
            json={"query": question, "texts": texts, "truncate": True},
            timeout=self._timeout,
        )
        resp.raise_for_status()

        scores = [0.0] * len(texts)

        for item in resp.json():
            scores[item["index"]] = item["score"]

        return scores


@functools.lru_cache(maxsize=1)
def get_rerank() -> LocalReranker | RemoteReranker:

    if settings.reranker_mode == "server":
        return RemoteReranker(settings.reranker_url)

    return LocalReranker(settings.cross_encoder_model)

def rerank(question: str, candidates: list[dict], top_k: int) -> list[dict]:

    if not candidates:
        return []
    
    scores = get_rerank().score(question, [c["text"] for c in candidates])

    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        {
            **candidates[i], 
            "rerank_score": scores[i]
        }
        for i in order
    ]