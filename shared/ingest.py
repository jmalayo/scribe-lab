from __future__ import annotations

import functools
import hashlib
import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_text_splitters import RecursiveCharacterTextSplitter

from shared.settings import settings

# numpy, requests, qdrant_client y transformers se cargan solo donde se usan: el chunking, la
# carga del corpus y corpus_hash quedan importables con las dependencias mínimas de los tests
if TYPE_CHECKING:
    from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

FRONT_MATTER_RE = re.compile(r"^---\n.*?\n---\n\n?", re.DOTALL)

@dataclass
class SourceDoc:
    doc_id: str  # p.ej. "qdrant/hybrid-search.md" -- coincide con source_doc en questions.jsonl
    library: str
    text: str

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    library: str
    text: str
    chunk_index: int
    tables: list[dict] = field(default_factory=list)

def validate_chunk(text, model_tokenizer, max_tokens):

    tokens = model_tokenizer.encode(text, add_special_tokens=False, truncation=True)

    return model_tokenizer.decode(tokens)

def load_corpus(corpus_dir: str | None = None) -> list[SourceDoc]:

    root = Path(corpus_dir or settings.corpus_dir)
    docs = []

    for path in sorted(root.rglob("*.md")):

        raw = path.read_text(encoding="utf-8")
        body = FRONT_MATTER_RE.sub("", raw, count=1)

        library = path.parent.name
        doc_id = f"{library}/{path.name}"
        
        docs.append(
            SourceDoc(
                doc_id=doc_id, 
                library=library, 
                text=body.strip()
            )
        )

    return docs

def get_tables_from_docs(doc: SourceDoc) -> tuple[list[dict], str]:

    import markdown

    blocks = re.split(r"\n{2,}", doc.text)

    tables = []

    idx = 0
    for block in blocks:

        if not block.strip():
            continue

        html = markdown.markdown(block, extensions=["tables"])

        if "<table" in html:

            tables.append({
                "text_content": block,
                "doc_id": doc.doc_id,
                "type": "Table"
            })

            idx += 1

    text_without_tables = doc.text

    for tb in tables:
        text_without_tables = text_without_tables.replace(tb["text_content"], "")

    logger.info(f"{doc.doc_id}: {idx} tabla(s) encontrada(s)")

    return tables, text_without_tables

def corpus_hash(docs: list[SourceDoc]) -> str:

    data = (
        "".join(
                f"{d.doc_id}:{d.text}" 
                    for d in sorted(docs, key=lambda d: d.doc_id)
            )
        )
    
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

def chunk_documents(
    docs: list[SourceDoc], 
    tables: list[dict], 
    chunk_size: int, 
    chunk_overlap: int
) -> list[Chunk]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    embedder = get_embedder()

    chunks: list[Chunk] = []

    for doc in docs:

        doc_tables = [
            t for t in tables
                if t["doc_id"] == doc.doc_id
        ]

        marked_text = doc.text

        for idx, t in enumerate(doc_tables):
            marked_text = marked_text.replace(t["text_content"], f"[[TABLE:{idx}]]")
    
        marked_text = re.sub(r"\n+(\[\[TABLE:\d+\]\])", r" \1", marked_text)
        marked_text = re.sub(r"(\[\[TABLE:\d+\]\])\n+", r"\1 ", marked_text)

        splits = splitter.split_text(marked_text)

        pending_tables = []

        for i, split in enumerate(splits):

            matches = re.findall(r"\[\[TABLE:(\d+)\]\]", split)

            tables_idx = []

            for m in matches:
                tables_idx.append(int(m))

            piece_tables = [doc_tables[m] for m in tables_idx]

            text = re.sub(r"\[\[TABLE:\d+\]\]", "", split).strip()

            if not text:
                pending_tables.extend(piece_tables)
                
                continue

            validated_piece = validate_chunk(text, embedder.tokenizer, embedder.max_seq_length)

            chunks.append(
                Chunk(
                    chunk_index=i,
                    chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc.doc_id}#{i}")),
                    doc_id=doc.doc_id,
                    library=doc.library,
                    text=validated_piece,
                    tables=pending_tables + piece_tables,
                )
            )

            pending_tables = []

        if pending_tables:

            doc_chunks = [c for c in chunks if c.doc_id == doc.doc_id]

            if doc_chunks:
                doc_chunks[-1].tables.extend(pending_tables)
            else:
                logger.warning(
                    f"{doc.doc_id}: el doc quedo compuesto solo por tablas, "
                    "sin ningun chunk de texto al cual adjuntarlas"
                )

    return chunks
    
class RemoteEmbedder:

    _MAX_CLIENT_BATCH = 32

    def __init__(self, url: str, model_name: str, timeout: tuple[float, float] = (5, 30)):

        self._url = url.rstrip("/")
        self._model_name = model_name
        self._timeout = timeout
        self._tokenizer = None
        self._max_seq_length = None
        self._dim = None

    @property
    def tokenizer(self):

        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)

        return self._tokenizer

    @property
    def max_seq_length(self) -> int:

        if self._max_seq_length is None:
            self._max_seq_length = self._request("get", "/info").json()["max_input_length"]

        return self._max_seq_length

    def get_dimension(self) -> int:

        if self._dim is None:
            self._dim = int(self.encode("dimension").shape[-1])

        return self._dim

    def health(self) -> bool:

        try:
            self._request("get", "/health")

        except Exception:
            return False

        return True

    def _request(self, method: str, path: str, json: dict | None = None):

        import requests

        resp = requests.request(
            method, 
            f"{self._url}{path}", 
            json=json, 
            timeout=self._timeout
        )

        resp.raise_for_status()

        return resp

    def encode(self, texts, normalize_embeddings=False):

        import numpy as np

        single = isinstance(texts, str)
        inputs = [texts] if single else list(texts)

        embeddings = []

        for start in range(0, len(inputs), self._MAX_CLIENT_BATCH):

            batch = inputs[start : start + self._MAX_CLIENT_BATCH]
            resp = self._request("post", "/embed", json={"inputs": batch, "truncate": True})
            embeddings.extend(resp.json())

        vectors = np.array(embeddings, dtype=np.float32)

        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.clip(norms, 1e-12, None)

        return vectors[0] if single else vectors


@functools.lru_cache(maxsize=1)
def get_embedder():

    if settings.text_embedder_mode == "server":
        return RemoteEmbedder(
            url=settings.text_embedder_url,
            model_name=f"sentence-transformers/{settings.embedding_model}",
        )

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        model_name_or_path=settings.embedding_model, 
        device="cpu"
    )

def qualified_collection_name(base_name: str) -> str:

    model_tag = settings.embedding_model.replace("/", "--")

    return f"{base_name}__{model_tag}"

def get_current_config(client: QdrantClient, collection_name: str) -> dict | None:

    if not client.collection_exists(collection_name):
        return None

    points, _ = client.scroll(
        collection_name=collection_name,
        limit=1,
        with_payload=True,
        with_vectors=False
    )

    if not points:
        return None

    return {
        k: v 
            for k, v in points[0].payload.items() 
                if k not in ["chunk_id", "text", "doc_id", "library", "chunk_index", "tables"]
    }

def build_qdrant_client(local_path: str | None = None) -> QdrantClient:
    from qdrant_client import QdrantClient

    if settings.qdrant_mode == "server":

        return QdrantClient(
            host=settings.qdrant_host, 
            port=settings.qdrant_port
        )
        
    return (
        QdrantClient(path=local_path) 
            if local_path else QdrantClient(":memory:")
    )

def index_chunks(
    client: QdrantClient,
    chunks: list[Chunk],
    collection_name: str,
    config: dict,
    batch_size: int = 64,
) -> int:

    from qdrant_client import models

    embedder = get_embedder()
    dim = embedder.get_dimension()

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=dim, 
            distance=models.Distance.COSINE
        ),
    )

    total = 0
    for start in range(0, len(chunks), batch_size):
        
        batch = chunks[start : start + batch_size]
        vectors = embedder.encode(
            [
                c.text for c in batch
            ], 
            normalize_embeddings=True
        )

        points = [
            models.PointStruct(
                id=c.chunk_id,
                vector=vector.tolist(),
                payload={
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "doc_id": c.doc_id,
                    "library": c.library,
                    "chunk_index": c.chunk_index,
                    "tables": c.tables,
                    **config,
                },
            )
            for c, vector in zip(batch, vectors, strict=True)
        ]

        client.upsert(
            collection_name=collection_name, 
            points=points
        )

        total += len(points)

    return total


def content_chunk_ids(chunks: list[Chunk]) -> list[Chunk]:

    seen: dict[str, int] = {}
    out = []

    for c in chunks:
        digest = hashlib.sha256(c.text.encode("utf-8")).hexdigest()
        key = f"{c.doc_id}#{digest}"
        n = seen.get(key, 0)
        seen[key] = n + 1
        name = key if n == 0 else f"{key}#{n}"
        out.append(
            Chunk(
                chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, name)),
                doc_id=c.doc_id,
                library=c.library,
                text=c.text,
                chunk_index=c.chunk_index,
                tables=c.tables,
            )
        )

    return out

def collection_behind_alias(client: QdrantClient, alias: str) -> str | None:

    for a in client.get_aliases().aliases:
        if a.alias_name == alias:
            return a.collection_name

    return None

def index_with_alias(
    client: QdrantClient,
    chunks: list[Chunk],
    alias: str,
    version: str,
    config: dict,
    batch_size: int = 64,
) -> dict:

    from qdrant_client import models

    target = f"{alias}__{version}"

    previous = collection_behind_alias(client, alias)

    if previous == target:
        return {
            "collection": target, 
            "indexed": 0, 
            "changed": False
        }

    n = index_chunks(
        client, 
        content_chunk_ids(chunks), 
        target, 
        config, 
        batch_size=batch_size
    )

    client.update_collection_aliases(
        change_aliases_operations=[
            models.CreateAliasOperation(create_alias=models.CreateAlias(collection_name=target, alias_name=alias))
        ]
    )

    if previous is not None:
        client.delete_collection(previous)

    return {
        "collection": target, 
        "indexed": n, "changed": True
    }
